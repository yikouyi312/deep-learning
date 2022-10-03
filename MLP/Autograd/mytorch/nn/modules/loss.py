import numpy as np
from mytorch.nn.functional import matmul_backward, add_backward, sub_backward, mul_backward, div_backward, \
    exp_backward, sum_backward, log_backward
#from mytorch.autograd_engine import backward

class MSELoss:
    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine
        self.loss_val = None

    def __call__(self, y, y_hat):
        self.forward(y, y_hat)

    # TODO: Use your working MSELoss forward and add operations to 
    # autograd_engine.
    def forward(self, y, y_hat):
        """
            This class is similar to the wrapper functions for the activations
            that you wrote in functional.py with a couple of key differences:
                1. Notice that instead of passing the autograd object to the forward
                    method, we are instead saving it as a class attribute whenever
                    an MSELoss() object is defined. This is so that we can directly 
                    call the backward() operation on the loss as follows:
                        >>> mse_loss = MSELoss(autograd_object)
                        >>> mse_loss(y, y_hat)
                        >>> mse_loss.backward()

                2. Notice that the class has an attribute called self.loss_val. 
                    You must save the calculated loss value in this variable and 
                    the forward() function is not expected to return any value.
                    This is so that we do not explicitly pass the divergence to 
                    the autograd engine's backward method. Rather, calling backward()
                    on the MSELoss object will take care of that for you.

            Args:
                - y (np.ndarray) : the ground truth,
                - y_hat (np.ndarray) : the output computed by the network,

            Returns:
                - No return required
        """
        #TODO: Use the primitive operations to calculate the MSE Loss
        #TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation
        self.loss_val = 0
        dif = y - y_hat
        n = y.shape[0]
        self.autograd_engine.add_operation(inputs=[y, y_hat],
                                           output=dif,
                                           gradients_to_update=[None, None],
                                           backward_operation=sub_backward)
        square = dif * dif
        self.autograd_engine.add_operation(inputs=[dif, dif],
                                           output=square,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        loss = np.sum(square, axis=1)
        self.autograd_engine.add_operation(inputs=[square],
                                           output=loss,
                                           gradients_to_update=[None],
                                           backward_operation=sum_backward)
        avg = loss / n
        self.autograd_engine.add_operation(inputs=[loss, n],
                                           output=avg,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)
        self.loss_val = np.arrayy([avg])
        return self.loss_val


    def backward(self):
        # You can call autograd's backward here or in the mlp.
        self.autograd_engine.backward(1)
        raise NotImplementedError

# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy:
    def __init__(self, autograd_engine):
        self.loss_val = None
        self.y_grad_placeholder = None
        self.autograd_engine = autograd_engine
        self.dlda = None

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

    def forward(self, y, y_hat):
        """
            Refer to the comments in MSELoss
        """
        N = y.shape[0]
        C = y.shape[1]
        Ones_C = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")
        softmax = np.exp(y_hat) / (np.exp(y_hat) @ Ones_C @ np.transpose(Ones_C))
        crossentropy = - y * np.log(softmax)  # TODO
        sum_crossentropy = np.sum(crossentropy)  # TODO
        L = sum_crossentropy / N
        self.loss_val = np.array([L])
        self.dlda = softmax - y
        # N = y_hat.shape[0]
        # C = y_hat.shape[1]
        # expy = np.exp(y_hat)
        # self.autograd_engine.add_operation(inputs=[y_hat],
        #                                    output=expy,
        #                                    gradients_to_update=[None],
        #                                    backward_operation=exp_backward)
        # sum_expy = expy @ np.ones((C, 1))
        # self.autograd_engine.add_operation(inputs=[expy, np.ones((C, 1))],
        #                                    output=sum_expy,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=matmul_backward)
        # sum_expy_transform = sum_expy @ np.ones((1, C))
        # self.autograd_engine.add_operation(inputs=[sum_expy, np.ones((1, C))],
        #                                    output=sum_expy_transform,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=matmul_backward)
        # softmax = expy / sum_expy_transform
        # self.autograd_engine.add_operation(inputs=[expy, sum_expy_transform],
        #                                    output=softmax,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=div_backward)
        # log_softmax = np.log(softmax)
        # self.autograd_engine.add_operation(inputs=[softmax],
        #                                    output=log_softmax,
        #                                    gradients_to_update=[None],
        #                                    backward_operation=log_backward)
        # crossentropy_neg = y * log_softmax
        # self.autograd_engine.add_operation(inputs=[y, log_softmax],
        #                                    output=crossentropy_neg,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=mul_backward)
        # crossentropy = np.zeros(crossentropy_neg.shape)-crossentropy_neg
        # self.autograd_engine.add_operation(inputs=[np.zeros(crossentropy_neg.shape), crossentropy_neg],
        #                                    output=crossentropy,
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=sub_backward)
        # sum_crossentropy = np.sum(crossentropy)
        # self.autograd_engine.add_operation(inputs=[crossentropy],
        #                                    output=sum_crossentropy,
        #                                    gradients_to_update=[None],
        #                                    backward_operation=sum_backward)
        # L = sum_crossentropy / (N*np.ones(sum_crossentropy.shape))
        # self.autograd_engine.add_operation(inputs=[sum_crossentropy, N*np.ones(sum_crossentropy.shape)],
        #                                    output=np.array([L]),
        #                                    gradients_to_update=[None, None],
        #                                    backward_operation=div_backward)
        # self.loss_val = np.array([L])
        # self.dlda = softmax - y_hat
        return self.loss_val
        raise NotImplementedError


    def backward(self):
        # You can call autograd's backward here OR in the mlp.
        self.autograd_engine.backward(self.dlda)
        return
        #return self.softmax - self.y
        raise NotImplementedError
