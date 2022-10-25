import contextlib
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    TransformType,
    CInterpreter,
    CGradInterpreterPtr,
    CVmapInterpreterPtr,
    WithoutTop,
)


class FunctorchInterpreter:
    @contextlib.contextmanager
    def lower(self):
        # TODO: this is sketch
        try:
            guard = WithoutTop()
            yield
        finally:
            del guard

    def level(self):
        return self._cptr.level()


class VmapInterpreter(FunctorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Vmap
        self._cdata = cdata
        self._cptr = CVmapInterpreterPtr(cdata)

    def py_process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self):
        return self._cptr.batchSize()


class GradInterpreter(FunctorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Grad
        self._cdata = cdata
        self._cptr = CGradInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, self._cptr.lift, [args, kwargs])
        return args, kwargs

    # TODO: needs custom lower for GradMode interaction.

    def py_process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Grad]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FunctorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    raise RuntimeError(f"Don't know how to handle {key}")


def retrieve_current_functorch_interpreter():
    interpreter = torch._C._functorch.peek_interpreter_stack()
    assert interpreter is not None
    return coerce_cinterpreter(interpreter)


def dispatch_functorch(op, args, kwargs):
    interpreter = retrieve_current_functorch_interpreter()
    return interpreter.py_process(op, args, kwargs)
