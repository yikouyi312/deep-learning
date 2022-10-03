# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        output_size = ((A.shape[2] - self.kernel_size) // 1) + 1
        Z = np.zeros((A.shape[0], self.out_channels, output_size))
        for j in range(self.out_channels):
            for i in range(output_size):
                Z[:, j, i] = np.sum(A[:, :, i: i + self.kernel_size] * self.W[j, :, :], axis=(1, 2)) + self.b[j]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))  # TODO
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                for k in range(self.kernel_size):
                    self.dLdW[j, i, k] = np.sum(self.A[:, i, k: k + dLdZ.shape[2]] * dLdZ[:, j, :], axis=(0, 1))
        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # TODO

        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)),
                          'constant', constant_values=((0, 0), (0, 0), (0, 0)))  # TODO
        w_flip = np.flip(self.W, 2)
        dLdA = np.zeros(self.A.shape)
        for j in range(self.in_channels):
            for i in range(self.A.shape[2]):
                dLdA[:, j, i] = np.sum(dLdZ_pad[:, :, i: i + self.kernel_size] * w_flip[:, j, :], axis=(1, 2))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels,
                                             out_channels,
                                             kernel_size,
                                             weight_init_fn,
                                             bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # recall conv1d_stride1
        Z = self.conv1d_stride1.forward(A)
        # downsample
        Z = self.downsample1d.forward(Z)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA = self.downsample1d.backward(dLdZ)
        # TODO
        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA)  # TODO

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_width = ((A.shape[2] - self.kernel_size) // 1) + 1
        output_height = ((A.shape[3] - self.kernel_size) // 1) + 1
        Z = np.zeros((A.shape[0], self.out_channels, output_width, output_height))
        for j in range(self.out_channels):
            for i in range(output_width):
                for h in range(output_height):
                    Z[:, j, i, h] = np.sum(A[:, :, i: i + self.kernel_size, h: h + self.kernel_size]
                                           * self.W[j, :, :], axis=(1, 2, 3)) + self.b[j]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))  # TODO
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                for k1 in range(self.kernel_size):
                    for k2 in range(self.kernel_size):
                        self.dLdW[j, i, k1, k2] = np.sum(self.A[:, i, k1: k1 + dLdZ.shape[2], k2: k2 + dLdZ.shape[3]]
                                                         * dLdZ[:, j, :, :], axis=(0, 1, 2))
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # TODO

        dLdZ_pad = np.pad(dLdZ,
                          ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                           (self.kernel_size - 1, self.kernel_size - 1)),
                          'constant',
                          constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))  # TODO
        w_flip = np.flip(np.flip(self.W, 3), 2)
        dLdA = np.zeros(self.A.shape)
        for j in range(self.in_channels):
            for i in range(self.A.shape[2]):
                for h in range(self.A.shape[3]):
                    dLdA[:, j, i, h] = np.sum(dLdZ_pad[:, :, i: i + self.kernel_size, h: h + self.kernel_size]
                                              * w_flip[:, j, :, :], axis=(1, 2, 3))
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        Z = self.conv2d_stride1.forward(A)
        # downsample
        Z = self.downsample2d.forward(Z)  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Call downsample1d backward
        dLdA = self.downsample2d.backward(dLdZ)
        # TODO
        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)  # TODO

        return dLdA


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        # TODO
        self.upsample1d = Upsample1d(upsampling_factor)# TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)  # TODO
        self.A_upsampled = A_upsampled
        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO
        # Call backward in the correct order
        delta_out = np.flip(self.conv1d_stride1.W, 2)
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0),
                                 (self.conv1d_stride1.kernel_size - 1, self.conv1d_stride1.kernel_size - 1)),
                          'constant', constant_values=((0, 0), (0, 0), (0, 0))) # TODO
        dLdA = np.zeros(self.A_upsampled.shape)
        for j in range(self.conv1d_stride1.in_channels):
            for i in range(self.A_upsampled.shape[2]):
                dLdA[:, j, i] = np.sum(dLdZ_pad[:, :, i: i + self.conv1d_stride1.kernel_size]
                                       * delta_out[:, j, :], axis=(1, 2))
        dLdA = self.upsample1d.backward(dLdA)
        return dLdA


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO
        self.upsample2d = Upsample2d(upsampling_factor)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)  # TODO
        self.A_upsampled = A_upsampled
        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        dLdZ_pad = np.pad(dLdZ,
                          ((0, 0), (0, 0), (self.conv2d_stride1.kernel_size - 1, self.conv2d_stride1.kernel_size - 1),
                           (self.conv2d_stride1.kernel_size - 1, self.conv2d_stride1.kernel_size - 1)),
                          'constant',
                          constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))  # TODO
        w_flip = np.flip(np.flip(self.conv2d_stride1.W, 3), 2)
        dLdA = np.zeros(self.A_upsampled.shape)
        for j in range(self.conv2d_stride1.in_channels):
            for i in range(self.A_upsampled.shape[2]):
                for h in range(self.A_upsampled.shape[3]):
                    dLdA[:, j, i, h] = np.sum(dLdZ_pad[:, :, i: i + self.conv2d_stride1.kernel_size,
                                                       h: h + self.conv2d_stride1.kernel_size]
                                              * w_flip[:, j, :, :], axis=(1, 2, 3))
        dLdA = self.upsample2d.backward(dLdA)
        return dLdA


class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.in_channels = A.shape[1]
        self.in_width = A.shape[2]
        Z = np.reshape(A, (A.shape[0], A.shape[1] * A.shape[2]))  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, (dLdZ.shape[0], self.in_channels, self.in_width))  # TODO

        return dLdA
