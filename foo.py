import torch
from torch._custom_function import CustomVjp, to_custom_vjp
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch import vmap, grad
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

x = torch.randn(2, 3, 4, requires_grad=True)
y = torch.randn(2, 4, 5, requires_grad=True)

z = vmap(MyMatmul.apply)(x, y)
gx, gy = torch.autograd.grad(z, [x, y], torch.ones_like(z))

z = vmap(torch.mm)(x, y)
egx, egy = torch.autograd.grad(z, [x, y], torch.ones_like(z))
assert torch.allclose(gx / 2, egx)
assert torch.allclose(gy / 2, egy)

class MySin(CustomVjp):
    @staticmethod
    def forward(x):
        return x.sin(), [x]

    @staticmethod
    def backward(saved, grads):
        x, = saved
        gy, = grads
        return gy * x.sin()


x = torch.randn([])
y = grad(MySin.apply)(x)
assert torch.allclose(y, x.sin())

x = torch.randn([], requires_grad=True)
y = MySin.apply(x)
gx, = torch.autograd.grad(y, x, create_graph=True)
ggx, = torch.autograd.grad(gx, x)
assert torch.allclose(ggx, x.cos())

x = torch.randn([])
y = grad(grad(MySin.apply))(x)
assert torch.allclose(y, x.cos())

x = torch.randn(3)
y = vmap(grad(MySin.apply))(x)
assert torch.allclose(y, x.sin())

# Things to test:
#
# grad
# vmap
# vmap x grad
# grad x grad
# jacrev
#
# - saved {input, output, intermediate}
# - {1, 2+} x {inputs, outputs}
# - inplace operations inside body
# - returns view

# Interestingly, in JAX, they don't require gradient definition for intermediates.
class Cube(CustomVjp):
    @staticmethod
    def forward(x):
        three_x_sq = 3 * (x ** 2)
        return x ** 3, [three_x_sq]

    @staticmethod
    def backward(saved, grads):
        three_x_sq, = saved
        gy, = grads
        return three_x_sq * gy

x = torch.tensor(1., requires_grad=True)
gx = grad(Cube.apply)(x)
ggx = grad(grad(Cube.apply))(x)
print(gx)
print(ggx)


class MySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, gy):
        x, = ctx.saved_tensors
        return 2* gy * x.cos()


custom_sin = to_custom_vjp(MySin).apply
x = torch.randn([])
gx = grad(custom_sin)(x)
assert torch.allclose(gx, 2 * x.cos())

# import torch
# class MySquare(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         two_x = 2 * x
#         ctx.save_for_backward(two_x)
#         return x ** 2
# 
#     @staticmethod
#     def backward(ctx, gy):
#         two_x, = ctx.saved_tensors
#         return gy * two_x
# 
# x = torch.randn([], requires_grad=True)
# y = MySquare.apply(x)
# gy = torch.randn([], requires_grad=True)
# gx, = torch.autograd.grad(y, x, gy, create_graph=True)


import torch
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x ** 2
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gy):
        result, = ctx.saved_tensors
        return gy * 2 * result.sqrt()

x = torch.randn([], requires_grad=True)
y = MySquare.apply(x)
gy = torch.randn([], requires_grad=True)
gx, = torch.autograd.grad(y, x, create_graph=True)
ggx, = torch.autograd.grad(gx, x)
assert torch.allclose(ggx, torch.tensor(2.))

ggx = grad(grad(to_custom_vjp(MySquare).apply))(x)
assert torch.allclose(ggx, torch.tensor(2.))
