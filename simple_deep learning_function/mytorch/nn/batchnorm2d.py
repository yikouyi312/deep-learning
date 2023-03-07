# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        if eval:
            # TODO
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb
            # ones = np.ones((1, 1, 1, Z.shape[0], Z.shape[2], Z.shape[3]), dtype="f")
            # NZ = (Z - np.transpose(np.tensordot(ones, self.running_M, axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2))) \
            #      / np.transpose(np.tensordot(ones, self.running_V, axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2))
            # BZ = np.transpose(np.tensordot(ones, self.BW, axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2)) * NZ \
            #      + np.transpose(np.tensordot(ones, self.Bb, axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2))
            return BZ
            #raise NotImplemented
        self.Z = Z
        self.N = Z.shape[0] * Z.shape[2] * Z.shape[3]  # TODO
        # ones = np.ones((Z.shape[0], Z.shape[2], Z.shape[3], 1, 1, 1), dtype="f")
        #self.M = 1 / self.N * np.transpose(np.tensordot(ones, Z, axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2)) # TODO 1xN@NxC = 1xC
        #self.V = 1 / self.N * np.transpose(np.tensordot(ones, np.square(Z - self.M),
        #                                               axes=([0, 1, 2], [0, 2, 3])), (0, 3, 1, 2))
        self.M = 1 / self.N * np.sum(Z, axis=(0, 2, 3)).reshape(1, Z.shape[1], 1, 1)
        self.V = 1 / self.N * np.sum(np.square(Z - self.M), axis=(0, 2, 3)).reshape(1, Z.shape[1], 1, 1)
        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        self.BZ = self.BW * self. NZ + self.Bb

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M  # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO

        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3)).reshape(1, self.Z.shape[1], 1, 1)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3)).reshape(1, self.Z.shape[1], 1, 1)  # TODO

        dLdNZ = dLdBZ * self.BW  # TODO
        dLdV = - 0.5 * np.sum(dLdNZ * (self.Z - self.M) * np.power(self.V + self.eps, -1.5), axis=(0, 2, 3))\
                .reshape(1, self.Z.shape[1], 1, 1) # TODO
        dLdM = - np.sum(dLdNZ * np.power(self.V + self.eps, -0.5), axis=(0, 2, 3)).reshape(1, self.Z.shape[1], 1, 1) \
               - 2/self.N * dLdV * np.sum(self.Z - self.M, axis=(0, 2, 3)).reshape(1, self.Z.shape[1], 1, 1)  # TODO

        dLdZ = (dLdNZ * np.power(self.V + self.eps, -0.5) + dLdV * (2 / self.N * (self.Z - self.M)) + dLdM/self.N)# TODO

        return dLdZ
        raise NotImplemented
