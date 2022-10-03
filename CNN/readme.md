## Table of contents
* [CNN](#CNN) 


# CNN
  - Resampling `mytorch.resampling`
    - Upsampling1d
    - Downsampling1d
    - Upsampling2d
    - Downsampling2d
  - CNN `mytorch.conv`
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
    - CNN as a simple scanning MLP `hw2/mlp_scan.CNN_SimpleScanningMLP`
    - CNN as a distributed scanning MLP `hw2/mlp_scan.CNN_DistributedScanningMLP`
           
       
