import torch
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch import vmap
import functools
import torch.utils._pytree as pytree
from torch._C import (
    DispatchKey,
)

mysum = PyOperator("mysum")

@mysum.py_functorch_impl(TransformType.Vmap)
def mysum_batch_rule(interpreter, x, dim):
    print("invoked")

    if not torch._C._functorch.is_batchedtensor(x): 
        with interpreter.lower():
            return mysum(x, dim)

    bdim = torch._C._functorch.maybe_get_bdim(x)
    value = torch._C._functorch.get_unwrapped(x)

    with interpreter.lower():
        value = value.movedim(bdim, 0)
        return mysum(value, dim + 1)

@mysum.py_impl(torch._C.DispatchKey.AutogradCPU)
def mysum_autograd(x, dim):
    return torch.sum(x, dim)


torch.manual_seed(0)
x = torch.randn(2, 3)
y = mysum(x, 1)
assert torch.allclose(y, x.sum(1))

def test(f, f_p, in_dims, args):
    expected = vmap(f, in_dims)(*args)
    result = vmap(f_p, in_dims)(*args)
    assert torch.allclose(result, expected)

# single vmap
test(torch.sum, mysum, (0, None), (x, 0))

# nested vmap
x = torch.randn(2, 3, 4)
test(vmap(functools.partial(torch.sum, dim=0)),
     vmap(functools.partial(mysum, dim=0)),
     (0,),
     (x,))


custom_vjp_call = PyOperator('custom_vjp_call')

# TODO: registering to 'Autograd' doesn't work (alias keys don't work with py_impl)
@custom_vjp_call.py_impl(DispatchKey.AutogradCPU)
def custom_vjp_call_autograd(f_fwd, f_bwd, *operands):
    class Generated(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
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


def reductify_leaf(tensor, tensor_bdim, desired_bdim):
    if tensor_bdim is None and desired_bdim is None:
        return tensor
    if tensor_bdim is None and desired_bdim is not None:
        raise RuntimeError('NYI: A')
    if tensor_bdim is not None and desired_bdim is None:
        return tensor.sum(tensor_bdim)
    return tensor.movedim(tensor_bdim, desired_bdim)


def reductify(tensors, tensor_bdims, desired_bdims):
    tensors, spec = tree_flatten(tensors)
    tensor_bdims, _ = tree_flatten(tensor_bdims)
    desired_bdims, _ = tree_flatten(desired_bdims)

    result = [reductify_leaf(tensor, bdim, desired_bdim)
              for tensor, bdim, desired_bdim
              in zip(tensors, tensor_bdims, desired_bdims)]
    return tree_unflatten(result, spec)


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
def custom_vjp_call_vmap(interpreter, *operands):
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


class MySin(CustomVjp):
    @staticmethod
    def forward(x):
        return x.sin(), [x]

    @staticmethod
    def backward(saved, grads):
        x, = saved
        gy, = grads
        return 2 * gy * x.cos()

x = torch.randn([], requires_grad=True)
y = MySin.apply(x)
gx, = torch.autograd.grad(y, x)
assert torch.allclose(gx, 2 * x.cos())

class MyMatmul(CustomVjp):
    @staticmethod
    def forward(x, y):
        return torch.mm(x, y), (x, y)

    @staticmethod
    def backward(saved, grads):
        gxy, = grads
        x, y = saved
        gx = gxy @ y.T
        gy = x.T @ gxy
        return 2 * gx, 2 * gy

x = torch.randn(3, 4, requires_grad=True)
y = torch.randn(4, 5, requires_grad=True)
gz = torch.randn(3, 5)

z = MyMatmul.apply(x, y)
gx, gy = torch.autograd.grad(z, (x, y), gz)
assert torch.allclose(gx, 2 * (gz @ y.T))
assert torch.allclose(gy, 2 * (x.T @ gz))
