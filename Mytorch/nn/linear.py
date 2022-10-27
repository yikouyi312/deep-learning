import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        self.W: weight
        self.b: bias
        self.dLdw, self.dLdb
        """
        self.W = np.zeros((out_features, in_features), dtype="f")
        self.b = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        self.debug = debug

    def forward(self, A):
        """
        Input: data input to be weighted: A
        Return: data ouput from the layer: Z = x*W^T + b^T
        """
        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1), dtype="f")
        Z = A @ np.transpose(self.W) + self.Ones @ np.transpose(self.b)  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Input: dLdZ: NxC1
        Output: dLdA: NxC0
        """
        dZdA = np.transpose(self.W)  # TODO, C0xC1 matrix
        dZdW = self.A  # TODO NxC0
        dZdi = None
        dZdb = self.Ones  # TODO Nx1
        dLdA = dLdZ @ np.transpose(dZdA)  # TODO NxC0
        dLdW = np.transpose(dLdZ) @ dZdW  # TODO C1xC0
        dLdi = None
        dLdb = np.transpose(dLdZ) @ dZdb  # TODO C1x1
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi

        return dLdA
