import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.rl = self.Wrx @ x + self.brx + self.Wrh @ self.hidden + self.brh
        self.r = self.r_act(self.rl) # TODO
        self.zl = self.Wzx @ x + self.bzx + self.Wzh @ self.hidden + self.bzh
        self.z = self.z_act(self.zl) # TODO
        self.nl = self.Wnx @ x + self.bnx + self.r * (self.Wnh @ self.hidden + self.bnh)
        self.n = self.h_act(self.nl) # TODO
        h_t = (1 - self.z) * self.n + self.z * self.hidden# TODO
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        x = np.expand_dims(self.x, axis=0)
        hidden = np.expand_dims(self.hidden, axis=0)
        dx = np.zeros(x.shape)
        dh_prev_t = np.zeros(hidden.shape)

        dLdz = delta * (- self.n + hidden)
        dLdn = delta * (1 - self.z)
        dh_prev_t += self.z * delta

        self.dWnx = (dLdn * self.h_act.backward(self.n)).T @ x
        dx += (dLdn * self.h_act.backward(self.n)) @ self.Wnx
        self.dbnx = (dLdn * self.h_act.backward(self.n)).sum(axis=0)
        dLdr = dLdn * self.h_act.backward(self.n) * (self.Wnh @ self.hidden + self.bnh)
        self.dWnh = (dLdn * self.h_act.backward(self.n) * self.r).T * self.hidden
        dh_prev_t += dLdn * self.h_act.backward(self.n) * self.r @ self.Wnh
        self.dbnh = (dLdn * self.h_act.backward(self.n) * self.r).sum(axis=0)

        self.dWzx = (dLdz * self.z_act.backward()).T * x
        dx += dLdz * self.z_act.backward() @ self.Wzx
        self.dbzx = (dLdz * self.z_act.backward()).sum(axis=0)
        self.dWzh = (dLdz * self.z_act.backward()).T * self.hidden
        dh_prev_t += dLdz * self.z_act.backward() @ self.Wzh
        self.dbzh = (dLdz * self.z_act.backward()).sum(axis=0)

        self.dWrx = (dLdr * self.r_act.backward()).T * x
        dx += dLdr * self.r_act.backward() @ self.Wrx
        self.dbrx = (dLdr * self.r_act.backward()).sum(axis=0)
        self.dWrh = (dLdr * self.r_act.backward()).T * self.hidden
        dh_prev_t += dLdr * self.r_act.backward() @ self.Wrh
        self.dbrh = (dLdr * self.r_act.backward()).sum(axis=0)

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
        raise NotImplementedError
