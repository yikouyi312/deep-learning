# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):
            # TODO: Calculate updates for weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * self.l[layer_id].dLdW
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * self.l[layer_id].dLdW ** 2
            m_W = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
            v_W = self.v_W[layer_id] / (1 - self.beta2 ** self.t)
            self.l[layer_id].W = self.l[layer_id].W - self.lr * (m_W / np.sqrt(v_W + self.eps) + self.weight_decay * self.l[layer_id].W)
            # TODO: calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * self.l[layer_id].dLdb
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * self.l[layer_id].dLdb ** 2
            m_b = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
            v_b = self.v_b[layer_id] / (1 - self.beta2 ** self.t)
            self.l[layer_id].b = self.l[layer_id].b - self.lr * (m_b / np.sqrt(v_b + self.eps) + self.weight_decay * self.l[layer_id].b)

            #raise NotImplementedError("AdamW Not Implemented")
        return None