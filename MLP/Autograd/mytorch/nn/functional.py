import numpy as np
from mytorch.autograd_engine import Autograd

'''
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
'''
#3.2.1
#grad_output: dl/dw
#a_grad: dl/da = dl/dw* dw/da
#b_grad: dl/db = dl/dw* dw/db
#Assume w: NxC
def add_backward(grad_output, a, b):
    # w = a + b,  dw/da,  dw/db
    # grad_output: NxC, a: Cx1, b: Cx1
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)
    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    # TODO: implement the backward function for subtraction.
    # w = a - b,  dw/da,  dw/db
    # grad_output: NxC, a: 1xC, b: 1xC
    a_grad = grad_output * np.ones(a.shape)
    b_grad = - grad_output * np.ones(b.shape)
    return a_grad, b_grad
    raise NotImplementedError


def matmul_backward(grad_output, a, b):
    # TODO: implement the backward function for matrix product.
    # w = a @ b,  dw/da = b,  dw/db = a^T
    # grad_output: NxC, a: NxC0, b: C0xC
    # a_grad: NxC0, b_grad: C0xC
    a_grad = grad_output @ np.transpose(b)
    b_grad = np.transpose(np.transpose(grad_output) @ a)
    return a_grad, b_grad
    raise NotImplementedError


def outer_backward(grad_output, a, b):
    # w = np.outer(a, b),  dw/da,  dw/db
    # grad_output: NxC, a: 1xN, b: 1xC
    # w(i,j) = a(i)*b(j)
    assert (a.shape[0] == 1 or a.ndim == 1)
    assert (b.shape[0] == 1 or b.ndim == 1)
    # TODO: implement the backward function for outer product.
    a_grad = grad_output @ np.transpose(b)
    b_grad = a @ grad_output
    return a_grad, b_grad
    raise NotImplementedError


def mul_backward(grad_output, a, b):
    # TODO: implement the backward function for multiply.
    # w = a * b, element-wise
    # dw/da = b, dw/db = a
    # grad_output: NxC, a: NxC, b: NxC
    a_grad = grad_output * b
    b_grad = grad_output * a
    return a_grad, b_grad
    raise NotImplementedError


def div_backward(grad_output, a, b):
    # TODO: implement the backward function for division.
    # w = a / b, dw/da = 1/b, dw/db = a
    # grad_output: NxC, a: NxC, b: NxC
    a_grad = grad_output / b
    b_grad = -grad_output * a / (b**2)
    return a_grad, b_grad
    raise NotImplementedError


def log_backward(grad_output, a):
    # TODO: implement the backward function for log.
    # w = np.log(a), dw/da = 1/a
    # grad_output: NxC, a: NxC
    a_grad = grad_output / a
    return a_grad
    raise NotImplementedError


def exp_backward(grad_output, a):
    # TODO: implement the backward function for calculating exponential.
    # w = np.exp(a), dw/da = np.exp(a)
    # grad_output: NxC, a: NxC
    a_grad = grad_output * np.exp(a)
    return a_grad
    raise NotImplementedError

#
def max_backward(grad_output, a):
    # TODO: implement the backward function for max.
    # w = max(a, 0), dw/da = 1 if a > 0
    # grad_output: NxC, a: NxC
    a_grad = grad_output * np.where(a > 0, 1, 0)
    return a_grad
    pass


def sum_backward(grad_output, a):
    # TODO: implement the backward function for sum.
    # w = np.sum(a), dw/da = a
    # grad_output: Nx1, a: NxC
    a_grad = grad_output * np.ones(a.shape)
    return a_grad
    pass


def SoftmaxCrossEntropy_backward(grad_output, a):
    """
    TODO: implement Softmax CrossEntropy Loss here.
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """

    pass
