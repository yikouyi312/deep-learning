## Table of contents
* [MLP](#simple) 
* [Autograd](#Autograd) 

# MLP
  - Activation Functions
    - Sigmoid `mytorch.nn.Sigmoid`
    - Tanh `mytorch.nn.Tanh`
    - ReLU `mytorch.nn.ReLU`
  - Loss Functions
    - MSE Loss `mytorch.nn.MSELoss`
    - Cross-Entropy Loss `mytorch.nn.CrossEntropyLoss`
  - Linear Layer `mytorch.nn.Linear`
  - Optimizers `mytorch.optim.SGD`
  - Regularization
    - Batch Normalization `mytorch.nn.BatchNorm1d`  
# Autograd: Automatic Differentiation
  - MemoryBuffer Class `mytorch.utils.MemoryBuffer`
    - wrapper class that allow for updating the gradients
  - AutogradEngine Class `mytorch.autograd_engine.Autograd`
    - keep track of the sequence of operations being performed, and kicking off the backprop algorithm once the forward pass is complete.

