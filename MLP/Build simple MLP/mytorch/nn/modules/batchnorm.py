import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            ones = np.ones((Z.shape[0], 1), dtype="f")
            NZ = (Z - ones @ self.running_M)/(ones @ np.sqrt(self.running_V))
            BZ = ones @ self.BW * NZ + ones @ self.Bb
            return BZ
            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        ones = np.ones((self.N, 1), dtype="f")
        self.M         = 1/self.N * np.transpose(ones) @ Z # TODO 1xN@NxC = 1xC
        self.V         = 1/self.N * np.transpose(ones) @ np.square(Z - ones @ self.M) # TODO 1xN@(NxC-1xC)=1xC
        self.NZ        = (Z - ones @ self.M) / (ones @ np.sqrt(self.V + self.eps))# TODO (NxC - Nx1 @ 1xC) / (Nx1@(1xC+1))=NxC
        self.BZ        = ones @ self.BW * self.NZ + ones @ self.Bb # TODO Nx1 @ 1xC * NxC + Nx1@1xC = NxC
        
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis= 0)  # TODO sum(NxC * NxC, axis = 0) 1xC
        self.dLdBb  = np.sum(dLdBZ, axis=0) # TODO 1xC
        ones = np.ones((self.N, 1), dtype="f")
        dLdNZ       = dLdBZ * self.BW # TODO NxC * 1xC = NxC
        dLdV        = -0.5 * np.sum(dLdNZ * ((self.Z - ones @ self.M) * (ones @ np.power(self.V + self.eps, -1.5))), axis = 0) # TODO 1xC
        dLdM        = -np.sum(dLdNZ * np.power(self.V + self.eps, -0.5), axis = 0) - 2/self.N * dLdV * np.sum(self.Z - self.M, axis=0) # TODO 1xC
        
        dLdZ        = dLdNZ * np.power(self.V + self.eps, -0.5) + dLdV * (2/self.N *(self.Z - self.M)) + dLdM/self.N # TODO 1xC * 1xC + 1xC* 1xC
        
        return  dLdZ