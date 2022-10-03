import numpy as np
from mytorch.nn.functional import *

class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """
    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others
    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):
        self.autograd_engine.add_operation(inputs=[np.ones(x.shape), x],
                                           output=np.ones(x.shape) * x,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        return np.ones(x.shape) * x
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):
        x1 = np.zeros(x.shape)-x
        self.autograd_engine.add_operation(inputs=[np.zeros(x.shape), x],
                                           output=x1,
                                           gradients_to_update=[None, None],
                                           backward_operation=sub_backward)
        x2 = np.exp(x1)
        self.autograd_engine.add_operation(inputs=[x1],
                                           output=x2,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)
        x3 = np.ones(x2.shape) + x2
        self.autograd_engine.add_operation(inputs=[np.ones(x2.shape), x2],
                                           output=x3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        y = np.ones(x3.shape)/x3
        self.autograd_engine.add_operation(inputs=[np.ones(x3.shape), x3],
                                           output=y,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)
        return y
        raise NotImplementedError


class Tanh(Activation):
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):
        # tanh(x) =( e^(x) -e^(-x) ) /(e^(x) + e^(-x)) = ( e^(2x) -1 ) /(e^(2x) + 1)
        x1 = 2* np.ones(x.shape) * x
        self.autograd_engine.add_operation(inputs=[2* np.ones(x.shape), x],
                                           output=x1,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        x2 = np.exp(x1)
        self.autograd_engine.add_operation(inputs=[x1],
                                           output=x2,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)
        x3 = np.ones(x2.shape) + x2
        self.autograd_engine.add_operation(inputs=[np.ones(x2.shape),x2],
                                           output=x3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        x4 = x2 - np.ones(x2.shape)
        self.autograd_engine.add_operation(inputs=[x2, np.ones(x2.shape)],
                                           output=x4,
                                           gradients_to_update=[None, None],
                                           backward_operation=sub_backward)
        x5 = x4/x3
        self.autograd_engine.add_operation(inputs=[x4, x3],
                                           output=x5,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)
        return x5
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):
        x1 = np.maximum(x, 0)
        self.autograd_engine.add_operation(inputs=[x],
                                           output=x1,
                                           gradients_to_update=[None],
                                           backward_operation=max_backward)
        return x1
        raise NotImplementedError
