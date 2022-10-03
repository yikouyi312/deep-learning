import numpy as np


class MSELoss:
    def forward(self, A, Y):
        """
        Input: A = model output, Y = truth value
        N: number of observations = A.shape[0]
        C: number of features = A.shape[1]
        Returns: loss value = mse
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        se = (self.A - self.Y) * (self.A - self.Y)  # TODO
        sse = np.ones((1, N), dtype="f") @ se @ np.ones((C, 1), dtype="f")  # TODO
        mse = sse / (N * C)
        return mse[0, 0]

    def backward(self):
        """
        Input: None
        Returns: dLdA = A - y
        """
        dLdA = self.A - self.Y
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Input: A = model output, Y = truth value
        N: number of observations = A.shape[0]
        C: number of features = A.shape[1]
        Returns: loss value = crossentropyloss
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        Ones_C = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")
        self.softmax = np.exp(A) / (np.exp(A) @ Ones_C @ np.transpose(Ones_C))  # TODO
        crossentropy = -Y * np.log(self.softmax)  # TODO
        sum_crossentropy = np.transpose(Ones_N) @ crossentropy @ Ones_C  # TODO
        L = sum_crossentropy / N
        return L[0, 0]

    def backward(self):
        """
        Input: None
        Returns: dLdA = softmax - Y
        """
        dLdA = self.softmax - self.Y  # TODO
        return dLdA
