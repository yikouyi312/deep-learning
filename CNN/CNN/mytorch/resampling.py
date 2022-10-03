import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        Z = np.zeros((A.shape[0], A.shape[1],
                      A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)))# TODO
        for i in range(A.shape[2]):
            Z[:, :, self.upsampling_factor * i] = A[:, :, i]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = Downsample1d(self.upsampling_factor).forward(dLdZ)
        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.res = 0
    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        output_width = (A.shape[2] + (self.downsampling_factor - 1)) // self.downsampling_factor
        Z = np.zeros((A.shape[0], A.shape[1], output_width))# TODO
        for i in range(output_width):
            Z[:, :, i] = A[:, :, i * self.downsampling_factor]
        self.res = (A.shape[2] - 1) % self.downsampling_factor
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = Upsample1d(self.downsampling_factor).forward(dLdZ)
        if self.res > 0:
            dLdA = np.pad(dLdA, ((0, 0), (0, 0), (0, self.res)), 'constant',
                          constant_values=((0, 0), (0, 0), (0, 0)))
        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        output_width = A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)
        output_height = A.shape[3] * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(A.shape[2]):
            for j in range(A.shape[3]):
                Z[:, :, self.upsampling_factor * i, self.upsampling_factor * j] = A[:, :, i, j]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = Downsample2d(self.upsampling_factor).forward(dLdZ)
        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.res = [0, 0]
    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        output_width = (A.shape[2] + (self.downsampling_factor - 1)) // self.downsampling_factor
        output_height = (A.shape[3] + (self.downsampling_factor - 1)) // self.downsampling_factor
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = A[:, :, i * self.downsampling_factor, j * self.downsampling_factor]
        self.res = [(A.shape[2] - 1) % self.downsampling_factor,
                    (A.shape[3] - 1) % self.downsampling_factor]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = Upsample2d(self.downsampling_factor).forward(dLdZ)
        dLdA = np.pad(dLdA, ((0, 0), (0, 0), (0, self.res[0]), (0, self.res[1])),
                      'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
        return dLdA