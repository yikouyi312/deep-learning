import numpy as np
from mytorch.nn.functional import matmul_backward, add_backward, outer_backward

class Linear():
    def __init__(self, in_features, out_features, autograd_engine):
        # Out x In
        self.W = np.random.uniform(-np.sqrt(1 / in_features), np.sqrt(1 / in_features),
                                   size=(out_features, in_features))  # flip this to out x in to mimic pytorch
        # Out x 1
        self.b = np.random.uniform(-np.sqrt(1 / in_features), np.sqrt(1 / in_features),
                                   size=(out_features, 1))  # just change this to 1-d after implementing broadcasting
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
                
        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)

        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
            Computes the affine transformation forward pass of the Linear Layer

            Args:
                - x (np.ndarray): the input array,

            Returns:
                - (np.ndarray), the output of this forward computation.
        """
        #TODO: Use the primitive operations to calculate the affine transformation
        #      of the linear layer
        # x: N x In ; self.W: Out x In; self.b: Out x 1
        # y1: N x Out
        y1 = x @ np.transpose(self.W)
        #TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation
        self.autograd_engine.add_operation(inputs=[x, np.transpose(self.W)],
                                           output=y1,
                                           gradients_to_update=[None, np.transpose(self.dW)],
                                           backward_operation=matmul_backward)
        y = y1 + np.transpose(self.b)
        self.autograd_engine.add_operation(inputs=[y1, np.transpose(self.b)],
                                           output=y,
                                           gradients_to_update=[None, np.transpose(self.db)],
                                           backward_operation=add_backward)
        #TODO: remember to return the computed value
        return y
        raise NotImplementedError
