import numpy as np
from mytorch.nn.functional import *

class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        output_width = A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)
        output_height = A.shape[3] * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(A.shape[2]):
            for j in range(A.shape[3]):
                Z[:, :, self.upsampling_factor * i, self.upsampling_factor * j] = A[:, :, i, j]

        return Z
        raise NotImplementedError

    def backward(self, dLdZ):
        dLdA = Downsample2d(self.upsampling_factor).forward(dLdZ)
        return dLdA
        raise NotImplementedError


class Downsample2d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        output_width = (A.shape[2] + (self.downsampling_factor - 1)) // self.downsampling_factor
        output_height = (A.shape[3] + (self.downsampling_factor - 1)) // self.downsampling_factor
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = A[:, :, i * self.downsampling_factor, j * self.downsampling_factor]
        self.autograd_engine.add_operation(inputs=[A, np.ones(1) * self.downsampling_factor],
                                           output=Z,
                                           gradients_to_update=[None, None],
                                           backward_operation=downsampling2d_backward)
        # self.autograd_engine.add_operation(inputs=[A, np.ones(1) * self.downsampling_factor],
        #                                    output=Z,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=downsampling2d_backward)
        return Z
        raise NotImplementedError


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        Z = np.zeros((A.shape[0], A.shape[1],
                      A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)))  # TODO
        for i in range(A.shape[2]):
            Z[:, :, self.upsampling_factor * i] = A[:, :, i]
        return Z
        raise NotImplementedError

    def backward(self, dLdZ):
        dLdA = Downsample1d(self.upsampling_factor).forward(dLdZ)
        raise NotImplementedError


class Downsample1d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine
    def forward(self, A):
        output_width = (A.shape[2] + (self.downsampling_factor - 1)) // self.downsampling_factor
        Z = np.zeros((A.shape[0], A.shape[1], output_width))  # TODO
        for i in range(output_width):
            Z[:, :, i] = A[:, :, i * self.downsampling_factor]
        self.autograd_engine.add_operation(inputs=[A, np.ones(1) * self.downsampling_factor],
                                           output=Z,
                                           gradients_to_update=[None, None],
                                           backward_operation=downsampling1d_backward)
        return Z

        raise NotImplementedError
