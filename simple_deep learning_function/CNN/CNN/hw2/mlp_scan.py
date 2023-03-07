# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1,
                       ReLU(),
                       self.conv2,
                       ReLU(),
                       self.conv3,
                       Flatten()
        ]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(w1).reshape(self.conv1.conv1d_stride1.out_channels,
                                                               self.conv1.conv1d_stride1.kernel_size,
                                                               self.conv1.conv1d_stride1.in_channels).transpose((0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2).reshape(self.conv2.conv1d_stride1.out_channels,
                                                               self.conv2.conv1d_stride1.in_channels,
                                                               self.conv2.conv1d_stride1.kernel_size)
        self.conv3.conv1d_stride1.W = np.transpose(w3).reshape(self.conv3.conv1d_stride1.out_channels,
                                                               self.conv3.conv1d_stride1.in_channels,
                                                               self.conv3.conv1d_stride1.kernel_size)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=4, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1,
                       ReLU(),
                       self.conv2,
                       ReLU(),
                       self.conv3,
                       Flatten()
                       ]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        w1 = np.nan_to_num(w1)
        w1_modify = w1[0: 24 * self.conv1.conv1d_stride1.kernel_size, 0: 4]
        w1_modify = np.hstack((w1_modify, w1_modify))
        self.conv1.conv1d_stride1.W = np.transpose(w1_modify).reshape(self.conv1.conv1d_stride1.out_channels,
                                                               self.conv1.conv1d_stride1.kernel_size,
                                                               self.conv1.conv1d_stride1.in_channels
                                                               ).transpose((0, 2, 1))
        w2 = np.nan_to_num(w2)
        w2_slice = w2[0:4, 0: 8]
        w2_modify = np.zeros((16, 16))
        w2_modify[0:4, 0:8] = w2_slice
        w2_modify[12:16, 8:16] = w2_slice
        self.conv2.conv1d_stride1.W = np.transpose(w2_modify).reshape(self.conv2.conv1d_stride1.out_channels,
                                                               self.conv2.conv1d_stride1.kernel_size,
                                                               self.conv2.conv1d_stride1.in_channels
                                                               ).transpose((0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3).reshape(self.conv3.conv1d_stride1.out_channels,
                                                               self.conv3.conv1d_stride1.in_channels,
                                                               self.conv3.conv1d_stride1.kernel_size)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
