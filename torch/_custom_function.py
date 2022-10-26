import torch
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch._src.vmap import stupid_vmap, unwrap_batched, wrap_batched
from functorch import vmap, grad
import functools
import torch.utils._pytree as pytree
from torch._C import (
    DispatchKey,
)


custom_vjp_call = PyOperator('custom_vjp_call')
custom_vjp_call.fallthrough(DispatchKey.PythonTLSSnapshot)

def unwrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return torch._C._functorch._unwrap_for_grad(t, level)
    return t


def wrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return torch._C._functorch._wrap_for_grad(t, level)
    return t


def index_of(lst, tensor):
    for idx, t in enumerate(lst):
        if tensor is t:
            return idx
    return None


def wrap_outs_and_saved(
        unwrapped_outs,
        unwrapped_saved,
        inputs,
        unwrapped_inputs,
        level):
    outs = pytree.tree_map(functools.partial(wrap_grad, level), unwrapped_outs)

    saved = []
    for s in unwrapped_saved:
        idx = index_of(unwrapped_inputs, s)
        if idx is not None:
            saved.append(inputs[idx])
            continue
        idx = index_of(unwrapped_outs, s)
        if idx is not None:
            saved.append(outs[idx])
            continue
        saved.append(wrap_grad(level, s))
    return outs, saved


def custom_vjp_call_grad_generic(maybe_interpreter, f_fwd, f_bwd, *operands):
    class Generated(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            if maybe_interpreter:
                level = maybe_interpreter.level()
                unwrapped_operands = pytree.tree_map(functools.partial(unwrap_grad, level), operands)

                with torch.enable_grad(), maybe_interpreter.lower():
                    output = custom_vjp_call(f_fwd, f_bwd, *unwrapped_operands)
                    results, saved = output

                results, out_spec = pytree.tree_flatten(results)
                outs, saved = wrap_outs_and_saved(results, saved, operands, unwrapped_operands, level)
                outs = pytree.tree_unflatten(outs, out_spec)
                ctx.save_for_backward(*saved)
            else:
                outs, saved = f_fwd(*operands)
                # TODO: the user may save things that aren't Tensors
                ctx.save_for_backward(*saved)

            flat_outs, outs_spec = pytree.tree_flatten(outs)
            outs_and_saved_and_spec = flat_outs + [saved, outs_spec]
            return tuple(outs_and_saved_and_spec)

        @staticmethod
        def backward(ctx, *grads):
            # Accounts for saved and spec
            assert grads[-1] is None
            assert grads[-2] is None
            grads = grads[:-2]

            saved = ctx.saved_tensors
            result = f_bwd(saved, grads)
            return result

    outs_and_saved_and_spec = Generated.apply(*operands)
    out_spec = outs_and_saved_and_spec[-1]
    saved = outs_and_saved_and_spec[-2]
    flat_outs = outs_and_saved_and_spec[:-2]
    return pytree.tree_unflatten(flat_outs, out_spec), saved


@custom_vjp_call.py_functorch_impl(TransformType.Grad)
def custom_vjp_call_grad(interpreter, f_fwd, f_bwd, *operands):
    return custom_vjp_call_grad_generic(interpreter, f_fwd, f_bwd, *operands)


# TODO: registering to 'Autograd' doesn't work (alias keys don't work with py_impl)
@custom_vjp_call.py_impl(DispatchKey.AutogradCPU)
def custom_vjp_call_autograd(f_fwd, f_bwd, *operands):
    return custom_vjp_call_grad_generic(None, f_fwd, f_bwd, *operands)


def reductify_leaf(tensor, tensor_bdim, desired_bdim):
    if tensor_bdim is None and desired_bdim is None:
        return tensor
    if tensor_bdim is None and desired_bdim is not None:
        raise RuntimeError('NYI: A')
    if tensor_bdim is not None and desired_bdim is None:
        return tensor.sum(tensor_bdim)
    return tensor.movedim(tensor_bdim, desired_bdim)


def reductify(tensors, tensor_bdims, desired_bdims):
    tensors, spec = pytree.tree_flatten(tensors)
    tensor_bdims, _ = pytree.tree_flatten(tensor_bdims)
    desired_bdims, _ = pytree.tree_flatten(desired_bdims)

    result = [reductify_leaf(tensor, bdim, desired_bdim)
              for tensor, bdim, desired_bdim
              in zip(tensors, tensor_bdims, desired_bdims)]
    return pytree.tree_unflatten(result, spec)


def batchify(f_fwd, f_bwd, in_dims, batch_size):
    out_dims = None

    def new_f_fwd(*args):
        nonlocal out_dims
        outs, out_dims2 = stupid_vmap(f_fwd, in_dims, batch_size)(*args)
        out_dims = out_dims2
        return outs

    def new_f_bwd(grad_outs, saved):
        assert out_dims is not None
        grad_ins, grad_ins_dims = stupid_vmap(f_bwd, out_dims, batch_size)(grad_outs, saved)
        return reductify(grad_ins, grad_ins_dims, in_dims)

    def get_out_dims():
        assert out_dims is not None
        return out_dims

    return new_f_fwd, new_f_bwd, get_out_dims


@custom_vjp_call.py_functorch_impl(TransformType.Vmap)
def custom_vjp_call_vmap(interpreter, f_fwd, f_bwd, *operands):
    current_level = interpreter.level()
    unwrapped_operands, in_dims = unwrap_batched(operands)
    new_f_fwd, new_f_bwd, get_out_dims = batchify(f_fwd, f_bwd, in_dims, interpreter.batch_size())

    with interpreter.lower():
        result = custom_vjp_call(new_f_fwd, new_f_bwd, *unwrapped_operands)

    out_dims = get_out_dims()
    return wrap_batched(current_level, result, out_dims)


class CustomVjp:
    # TODO: support kwargs (or not)
    @classmethod
    def apply(cls, *args):
        outs, saved = custom_vjp_call(cls.forward, cls.backward, *args)
        return outs

# TODO: somehow we need to raise an error if
# (1) intermediate tensors are being saved
# (2) user is computing higher order gradients. e.g. grad(grad(...
class MockCtx:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def __setitem__(self, key, value):
        raise RuntimeError("NYI")

def to_custom_vjp(af):
    class Generated(CustomVjp):
        @staticmethod
        def forward(*args):
            ctx = MockCtx()
            output = af.forward(ctx, *args)
            return output, ctx.saved_tensors

        @staticmethod
        def backward(saved, grads):
            ctx = MockCtx()
            ctx.saved_tensors = saved
            return af.backward(ctx, *grads)

    return Generated
