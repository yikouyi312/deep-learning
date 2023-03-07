## Table of contents
* [Neural Network Model](#NeuralNetworkModel)
* [Convolution Neural Network](#ConvolutionNeuralNetwork)
* [RNNs and GRUs and Search](#RNNs&GRUs&Search)
* [Language Modeling using RNNs](#LanguageModelingusingRNNs)
* [AutomaticDifferentiation](#AutomaticDifferentiation)
* [Frame Level Classification of Speech](#FrameLevelClassificationofSpeech)
* [Face Classification & Verification using Convolutional Neural Networks](#FaceClassification&VerificationusingConvolutionalNeuralNetworks)
* [Utterance to Phoneme Mapping](#UtterancetoPhonemeMapping)
* [Attention-based End-to-End Speech-to-Text Deep Neural Network](#Attention-basedEnd-to-EndSpeech-to-TextDeepNeuralNetwork)
# Neural Network Model
* MLP.MLP.mytorch

# ConvolutionNeuralNetwork
*  CNN.CNN.mytorch

# RNNs&GRUs&Search
* RNNs and GRUs and Search

# LanguageModelingusingRNNs

# AutomaticDifferentiation
* Autograd
* a framework that allows us to calculate the derivatives of any arbitrarily complex mathematical function.
  - forward accumulation, computes the derivatives of the chain rule from inside to outside
  - reverse accumulation, computes the derivatives of the chain rule from outside to inside
*  Autograd framework keeps track of the sequence of operations that are performed on the input data leading up to the final loss calculation. It then performs backpropagation and calculates all the necessary gradients.

# FrameLevelClassificationofSpeech
  * Data
    -  MFCC data consisting of 15 features at each time step/frame
  * Model
    - MLP
    
# FaceClassification&VerificationusingConvolutionalNeuralNetworks
  * Data
    - VGGFace2 dataset
  * Goal
    - Classification: classify image with correct identity from 7000 indentities
    - Verification: map unkown identity image to known indentity
   * Model
    - CNN based architecture, ResNet, ConvNeXt
    
# UtterancetoPhonemeMapping
  * Data
    -  MFCC data consisting of 15 features at each time step/frame and 43 phoneme labels
  * Goal
    - seq-to-seq model and deal with the lack of time syschrony
    - simplify problem to one that has time syschrony by introducing /BLANK/ symbol
  * Deconding: From probbability to phoneme sequence
    - Greedy decoding
    - Beam search decoding
  * CTC: Connectionist  Temporal Classification
  * Model
    - RNN, LSTM,GRU
    
    
# Attention-basedEnd-to-EndSpeech-to-TextDeepNeuralNetwork
  * Data
    - 
