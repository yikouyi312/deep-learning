import numpy as np
from mytorch.nn.modules.resampling import *
from mytorch.nn.functional import *

class Conv1D_stride1():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.autograd_engine = autograd_engine

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0,
                                      (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_size)
        Return:
            Z (np.array): (batch_size, out_channel, output_size)
        """
        output_size = ((A.shape[2] - self.kernel_size) // 1) + 1
        Z = np.zeros((A.shape[0], self.out_channel, output_size))
        for j in range(self.out_channel):
            for i in range(output_size):
                Z[:, j, i] = np.sum(A[:, :, i: i + self.kernel_size] * self.W[j, :, :], axis=(1, 2)) + self.b[j]
        self.autograd_engine.add_operation(inputs=[A, self.W, self.b],
                                           output=Z,
                                           gradients_to_update=[None, self.dW, self.db],
                                           backward_operation=conv1d_stride1_backward)
        return Z
        #raise NotImplementedError


class Conv1d():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsampling_factor,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify the variable names
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

        # Initialize Conv1D() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1D_stride1(in_channel,
                                             out_channel,
                                             kernel_size,
                                             autograd_engine,
                                             weight_init_fn,
                                             bias_init_fn) #TODO
        self.downsample1d = Downsample1d(downsampling_factor, autograd_engine)#TODO

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_size)
        Return:
            Z (np.array): (batch_size, out_channel, output_size)
        """
        # Call Conv1D_stride1
        self.Z = self.conv1d_stride1.forward(A) #TODO
        # downsample
        Z = self.downsample1d.forward(self.Z) #TODO
        return Z
        raise NotImplementedError


class Conv2D_stride1():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):

        # Do not modify this method

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.autograd_engine = autograd_engine

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size,
                                    kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.A = A
        output_width = ((A.shape[2] - self.kernel_size) // 1) + 1
        output_height = ((A.shape[3] - self.kernel_size) // 1) + 1
        Z = np.zeros((A.shape[0], self.out_channel, output_width, output_height))
        for j in range(self.out_channel):
            for i in range(output_width):
                for h in range(output_height):
                    Z[:, j, i, h] = np.sum(A[:, :, i: i + self.kernel_size, h: h + self.kernel_size]
                                           * self.W[j, :, :], axis=(1, 2, 3)) + self.b[j]
        self.autograd_engine.add_operation(inputs=[A, self.W, self.b],
                                           output=Z,
                                           gradients_to_update=[None, self.dW, self.db],
                                           backward_operation=conv2d_stride1_backward)
        return Z

class Conv2d():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsampling_factor,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify the variable names
        self.downsampling_factor = downsampling_factor

        self.autograd_engine = autograd_engine
        # Initialize Conv2D() and Downsample2d() isntance

        self.conv2d_stride1 = Conv2D_stride1(in_channel,
                                             out_channel,
                                             kernel_size,
                                             autograd_engine,
                                             weight_init_fn,
                                             bias_init_fn)  #TODO
        self.downsample2d = Downsample2d(downsampling_factor, autograd_engine) #TODO

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channel, output_width, output_height)
        """

        # Call Conv2D_stride1
        self.Z = self.conv2d_stride1.forward(A) #TODO

        # downsample
        Z = self.downsample2d.forward(self.Z) #TODO
        return Z
        raise NotImplementedError


class Flatten():
    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        raise NotImplementedError
