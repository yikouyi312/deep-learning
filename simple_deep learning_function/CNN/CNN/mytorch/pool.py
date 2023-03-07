import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_width = A.shape[2] - (self.kernel - 1)
        output_height = A.shape[3] - (self.kernel - 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        self.MaxIndex = np.zeros((A.shape[0], A.shape[1], output_width, output_height, 2)).astype(int)
        for bath in range(A.shape[0]):
            for channel in range(A.shape[1]):
                for i in range(output_width):
                    for j in range(output_height):
                        x, y = np.unravel_index(np.argmax(A[bath, channel, i: i + self.kernel,
                                                          j: j + self.kernel]),
                                                (self.kernel, self.kernel))
                        x, y = int(x + i), int(y + j)
                        self.MaxIndex[bath, channel, i, j] = [x, y]
                        Z[bath, channel, i, j] = A[bath, channel, x, y]
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        for bath in range(self.MaxIndex.shape[0]):
            for channel in range(self.MaxIndex.shape[1]):
                for i in range(self.MaxIndex.shape[2]):
                    for j in range(self.MaxIndex.shape[3]):
                        dLdA[bath, channel,
                             self.MaxIndex[bath, channel, i, j, 0],
                             self.MaxIndex[bath, channel, i, j, 1]] += dLdZ[bath, channel, i, j]
        # for i in range(self.MaxIndex.shape[2]):
        #     for j in range(self.MaxIndex.shape[3]):
        #         dLdA[:, :,
        #              self.MaxIndex[:, :, i, j, 0],
        #              self.MaxIndex[:, :, i, j, 1]] += dLdZ[:, :, i, j]
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_width = A.shape[2] - (self.kernel - 1)
        output_height = A.shape[3] - (self.kernel - 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.mean(A[:, :, i: i + self.kernel, j: j + self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_pad = np.pad(dLdZ,
                          ((0, 0), (0, 0), (self.kernel - 1, self.kernel - 1),
                           (self.kernel - 1, self.kernel - 1)),
                          'constant',
                          constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))  # TODO
        W = np.ones((self.kernel, self.kernel)) / (self.kernel ** 2)
        dLdA = np.zeros(self.A.shape)
        for i in range(self.A.shape[2]):
            for h in range(self.A.shape[3]):
                dLdA[:, :, i, h] = np.sum(dLdZ_pad[:, :, i: i + self.kernel, h: h + self.kernel]
                                          * W, axis=(2, 3))
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
