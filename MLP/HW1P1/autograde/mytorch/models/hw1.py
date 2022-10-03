import numpy as np

from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU

class MLP0:

    def __init__(self, debug=False):
    
        self.layers = [ Linear(2, 3) ]
        self.f      = [ ReLU() ]

        self.debug = debug

    def forward(self, A0):
        """
        Input: A0, input data A0 = 4x2
        Output: A1 = f(A0@W^T +b^T), output data, W 3x2, b 3x1
        """
        Z0 = A0 @ np.transpose(self.layers[0].W) \
             + np.ones((A0.shape[0], 1), dtype="f") @ np.transpose(self.layers[0].b)  # TODO
        A1 = self.f[0].forward(Z0) # TODO
        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.A0 = A0
        return A1

    def backward(self, dLdA1):
        """
        Input: dLdA1
        Output: None
        """
        dA1dZ0 = self.f[0].backward() # TODO NxC1
        dLdZ0  = dLdA1 * dA1dZ0 # TODO NxC1
        dLdA0  = dLdZ0 @ self.layers[0].W # TODO NxC0
        # dLdW0  = np.transpose(dLdZ0) @ self.A0  # C1xN @ NxC0
        # dLdb0  = np.transpose(dLdZ0) @ np.ones((self.A0.shape[0], 1), dtype="f") #
        if self.debug:
            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
            self.layers[0].forward(self.A0)
            self.layers[0].backward(dLdZ0)
            # self.layers[0].dLdW = np.transpose(dLdZ0) @ self.A0
            # self.layers[0].dldB = np.transpose(dLdZ0) @ np.ones((self.A0.shape[0], 1), dtype="f")
        return None
        
class MLP1:

    def __init__(self, debug=False):
        self.layers = [ Linear(2, 3), Linear(3, 2) ]
        self.f      = [ ReLU(), ReLU() ]
        self.debug = debug

    def forward(self, A0):
        """
        Input: A0, input data
        Output: A2: output data
        """
        Z0 = A0 @ np.transpose(self.layers[0].W) \
             + np.ones((A0.shape[0], 1), dtype="f") @ np.transpose(self.layers[0].b)  # TODO
        A1 = self.f[0].forward(Z0) # TODO
    
        Z1 = A1 @ np.transpose(self.layers[1].W) \
             + np.ones((A1.shape[0], 1), dtype="f") @ np.transpose(self.layers[1].b) # TODO
        A2 = self.f[1].forward(Z1) # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2
            self.A0 = A0
        return A2

    def backward(self, dLdA2):
        """
        Input: dLdA2
        Output: None
        """
        dA2dZ1 = self.f[1].backward() # TODO
        dLdZ1  = dLdA2 * dA2dZ1  # TODO
        dLdA1  = dLdZ1 @ self.layers[1].W # TODO
    
        dA1dZ0 = self.f[0].backward() # TODO
        dLdZ0  = dLdA1 * dA1dZ0 # TODO
        dLdA0  = dLdZ0 @ self.layers[0].W # TODO

        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1  = dLdZ1
            self.dLdA1  = dLdA1
            self.layers[1].forward(self.A1)
            self.layers[1].backward(dLdZ1)

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0

            self.layers[0].forward(self.A0)
            self.layers[0].backward(dLdZ0)
        
        return None

class MLP4:
    def __init__(self, debug=False):
        
        # Hidden Layers
        self.layers = [
            Linear(2, 4),
            Linear(4, 8),
            Linear(8, 8),
            Linear(8, 4),
            Linear(4, 2)]

        # Activations
        self.f = [
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU()]

        self.debug = debug

    def forward(self, A):
        """
        Input: A, input data
        Output: A = f(A@W^T+b^T): output data
        """
        if self.debug:

            self.Z = []
            self.A = [ A ]

        L = len(self.layers)

        for i in range(L):

            Z = A @ np.transpose(self.layers[i].W) \
             + np.ones((A.shape[0], 1), dtype="f") @ np.transpose(self.layers[i].b)  # TODO
            A = self.f[i].forward(Z)  # TODO

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return A

    def backward(self, dLdA):
        """
        Input: dLdA
        Output: None
        """
        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [ dLdA ]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = self.f[i].backward()  # TODO
            dLdZ = dLdA * dAdZ  # TODO
            dLdA = dLdZ @ self.layers[i].W # TODO

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA
                self.layers[i].forward(self.A[i])
                self.layers[i].backward(dLdZ)

        return NotImplemented
        