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

def conv1d_stride1_backward(grad_output, a, b, c):
    # Hint: a is input, b is weight, c is bias
    b_grad = np.zeros(b.shape)  # b.shape[0] out, b.shape[1] in, b.shape[2] kernel
    for i in range(b.shape[1]):
        for j in range(b.shape[0]):
            for k in range(b.shape[2]):
                b_grad[j, i, k] = np.sum(a[:, i, k: k + grad_output.shape[2]] * grad_output[:, j, :], axis=(0, 1))
    c_grad = np.sum(grad_output, axis=(0, 2))
    grad_output_pad = np.pad(grad_output, ((0, 0), (0, 0), (b.shape[2] - 1, b.shape[2] - 1)),
                      'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    b_flip = np.flip(b, 2)
    a_grad = np.zeros(a.shape)
    for j in range(a.shape[1]):
        for i in range(a.shape[2]):
            a_grad[:, j, i] = np.sum(grad_output_pad[:, :, i: i + b.shape[2]] * b_flip[:, j, :], axis=(1, 2))

    return a_grad, b_grad, c_grad
    raise NotImplementedError


def conv2d_stride1_backward(grad_output, a, b, c):
    # Hint: a is input, b is weight, c is bias
    b_grad = np.zeros(b.shape)  # TODO
    for i in range(b.shape[1]):
        for j in range(b.shape[0]):
            for k1 in range(b.shape[2]):
                for k2 in range(b.shape[3]):
                    b_grad[j, i, k1, k2] = np.sum(a[:, i, k1: k1 + grad_output.shape[2], k2: k2 + grad_output.shape[3]]
                                                     * grad_output[:, j, :, :], axis=(0, 1, 2))
    c_grad = np.sum(grad_output, axis=(0, 2, 3))  # TODO
    grad_output_pad = np.pad(grad_output,
                      ((0, 0), (0, 0), (b.shape[2] - 1, b.shape[2] - 1),
                       (b.shape[2] - 1, b.shape[2] - 1)),
                      'constant',
                      constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))  # TODO
    b_flip = np.flip(np.flip(b, 3), 2)
    a_grad = np.zeros(a.shape)
    for j in range(a.shape[1]):
        for i in range(a.shape[2]):
            for h in range(a.shape[3]):
                a_grad[:, j, i, h] = np.sum(grad_output_pad[:, :, i: i + b.shape[2], h: h + b.shape[2]]
                                          * b_flip[:, j, :, :], axis=(1, 2, 3))
    return a_grad, b_grad, c_grad
    raise NotImplementedError


def downsampling1d_backward(grad_output, a, downsampling_factor):
    downsampling_factor = int(downsampling_factor)
    res = (a.shape[2] - 1) % downsampling_factor
    a_grad = np.zeros((grad_output.shape[0], grad_output.shape[1],
                       grad_output.shape[2] * downsampling_factor - (downsampling_factor - 1)))  # TODO
    for i in range(grad_output.shape[2]):
        a_grad[:, :, downsampling_factor * i] = grad_output[:, :, i]
    if res > 0:
        a_grad = np.pad(a_grad, ((0, 0), (0, 0), (0, res)), 'constant',
                      constant_values=((0, 0), (0, 0), (0, 0)))
    b_grad = None
    return a_grad, b_grad
    raise NotImplementedError


def downsampling2d_backward(grad_output, a, downsampling_factor):
    upsampling_factor = int(downsampling_factor)
    res = [(a.shape[2] - 1) % upsampling_factor,
           (a.shape[3] - 1) % upsampling_factor]
    output_width = grad_output.shape[2] * upsampling_factor - (upsampling_factor - 1)
    output_height = grad_output.shape[3] * upsampling_factor - (upsampling_factor - 1)
    a_grad = np.zeros((grad_output.shape[0], grad_output.shape[1], output_width, output_height))
    for i in range(grad_output.shape[2]):
        for j in range(grad_output.shape[3]):
            a_grad[:, :, upsampling_factor * i, upsampling_factor * j] = grad_output[:, :, i, j]
    a_grad = np.pad(a_grad, ((0, 0), (0, 0), (0, res[0]), (0, res[1])),
                  'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    b_grad = None
    return a_grad, b_grad
    raise NotImplementedError


def flatten_backward(grad_output, a):
    raise NotImplementedError