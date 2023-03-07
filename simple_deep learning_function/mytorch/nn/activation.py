import numpy as np


class Identity:

    def forward(self, Z):
        """
        Input: Z
        Output: A
        """
        self.A = Z
        return self.A

    def backward(self):
        """
        Input: None
        Returns: dAdZ = 1
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        return dAdZ


class Sigmoid:

    def forward(self, Z):
        """
        Input: Z
        Output: 1/(1+exp(-Z))
        """
        # TODO
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self):
        """
        Input: None
        Returns: dAdZ = sigmoid(Z)-sigmoid(Z)^2
        """
        dAdZ = self.A - self.A * self.A
        return dAdZ


class Tanh:

    def forward(self, Z):
        """
        Input: Z
        Output: tanh(Z)
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self):
        """
        Input: None
        Returns: dAdZ = 1 - tanh^2(Z)
        """
        dAdZ = 1 - self.A * self.A
        return dAdZ


class ReLU:

    def forward(self, Z):
        """
        Input: Z
        Output: max(Z,0)
        """
        self.A = np.maximum(Z, 0)
        return self.A

    def backward(self):
        """
        Input: None
        Returns: dAdZ = 1 if A> 0
        """
        dAdZ = np.where(self.A > 0, 1, 0)
        return dAdZ


