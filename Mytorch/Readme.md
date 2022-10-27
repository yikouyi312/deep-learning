## Table of contents
* [MLP](#MLP) 
* [CNN](#CNN)

# MLP
  - Activation Functions
    - Sigmoid `mytorch.nn.Sigmoid`
    - Tanh `mytorch.nn.Tanh`
    - ReLU `mytorch.nn.ReLU`
  - Loss Functions
    - MSE Loss `MLP\mytorch.nn.MSELoss`
    - Cross-Entropy Loss `mytorch.nn.CrossEntropyLoss`
  - Linear Layer `mytorch.nn.Linear`
  - Optimizers `mytorch.optim`
    - `sgd.py`, `adam.py`, `adamW.py`
  - Regularization
    - Batch Normalization 1d
      - `mytorch.nn.batchnorm.py`
    - Batch Normalization 2d
      - `mytorch.nn.batchnorm2d.py` 
# CNN
  - Resampling `mytorch.nn.resampling`
    - Upsampling1d
    - Downsampling1d
    - Upsampling2d
    - Downsampling2d
  - CNN `mytorch.nn.conv`
    - Conv1d_stride1: stride = 1
    - Conv1d: stride > 1
      - Conv1d_stride1
      - Downsampling1d
    - Conv2d_stride1: stride = 1
    - Conv2d: stride > 1
      - Conv2d_stride1
      - Downsampling1d
    - ConvTranspose1d
      - Upsampling1d
      - Conv1d_stride1 
    - ConvTranspose2d
      - Upsampling2d
      - Conv2d_stride1
    - Flatten
    - Pooling
      - Max pooling
      - Mean pooling
  - Convert scanning MLPs to CNNs
    - CNN as a simple scanning MLP `mlp_scan.CNN_SimpleScanningMLP`
    - CNN as a distributed scanning MLP `mlp_scan.CNN_DistributedScanningMLP`

    
