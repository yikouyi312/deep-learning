from mytorch.nn import Identity, Sigmoid, Tanh, ReLU
from mytorch.nn import MSELoss, CrossEntropyLoss
from mytorch.nn import Linear
from mytorch.nn import BatchNorm1d
from mytorch.optim import SGD
from mytorch.models import MLP0, MLP1, MLP4
from mytorch.hw1p1_autograder_flags import *
import numpy as np
class PseudoModel:
    def __init__(self):
        self.layers = [mytorch.nn.Linear(3, 2)]
        self.f = [mytorch.nn.ReLU()]

    def forward(self, A):
        return NotImplemented

    def backward(self):
        return NotImplemented


# Create Example Model
pseudo_model = PseudoModel()
pseudo_model.layers[0].W = np.ones((3, 2))
pseudo_model.layers[0].dLdW = np.ones((3, 2)) / 10
pseudo_model.layers[0].b = np.ones((3, 1))
pseudo_model.layers[0].dLdb = np.ones((3, 1)) / 10
print("W\n\n", pseudo_model.layers[0].W)
print("W\n\n", pseudo_model.layers[0].b)
# Test Example Models
optimizer = SGD(pseudo_model, lr=1)
optimizer.step()
print("W\n\n", pseudo_model.layers[0].W)
print("W\n\n", pseudo_model.layers[0].b)