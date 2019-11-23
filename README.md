# Ensemble-Kalman-for-Neural-Network
Project for master thesis (2019)

## Introduction
Deep learning algorithms usually require a high amount of numerical computations. This typically refers to algorithms that solve the given problem by the methods of updating the estimated parameters via an iterative process, called as the optimization algorithms.

The thesis focuses on the learning methods in deep neural networks for supervised learning problems. It compares the traditional gradient based optimizations backpropagation method. Back-Propagation has two phases, the propagation step and weight update step. Errors from the output layers will propagate to the layers of neurons in a backward motion. These errors will be used to calculate gradient of the loss function with respect to the weights in the network. The gradient is then fed to optimization method, which has the task of updating the weights, with the goal of minimizing the loss function.

Over the years, we have seen many optimization methods for deep learning that are derivative- based algorithms such as Momentum (Nesterov 1983, Tseng 1998), Rprop (Riedmiller and Braun, 1993), Adagrad (Dutchi et al., 2011 [15]), RMSprop (Tieleman and Hinton, 2012), Adam (Kingma and Ba, 2014) and Adadelta (Zeiler, 2012).

However, there are many disadvatanges to Back-Propagation (BP). The most notable issues arise with scaling problems and differentiability of the gradient. As the problem complexity arises (e.g. increase of dimensionality) the performance of BP also worsen as it is very intense for a computer to compute the gradient of large number of parameters, so the training might become slower due to gradient that get stuck at local minima.

The thesis introduces the new derivative-free optimization method based on Iterative Ensemble Kalman method for inverse problem (IEnK). This method was proposed by Kocachki and Stuart (2018), who have demonstrated that IEnK method significantly outperform the traditional SGD optimization methods in training neural networks (CNN, RNN).

## Implementation details
The IEnK algorithm was written in Python version 3.6 and the training was done on Intel Core i7-5650U CPU with the MNIST dataset. In the thesis we have simplified the algorithm to the learning rate of real number. This simplified IEnK algorithm was written for the simple MLP architecture with 1 hidden layer.
