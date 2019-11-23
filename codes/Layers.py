#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:25:31 2019
Neural Network Model

@author: diepy
"""

import numpy as np

class Dense:

    def __init__(self, num_inputs, num_outputs, learning_rate, name, num_particles):
        self.sigma = np.sqrt(2/(num_outputs+num_inputs))
        self.weights = np.random.normal(loc=0.0,scale=self.sigma,size=(num_inputs, num_outputs))
        self.bias = np.random.normal(loc=0.0,scale=self.sigma,size=(num_outputs))
        self.lr = learning_rate
        self.name = name
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.k = num_particles


    def forward(self, inputs):
        self.inputs = inputs
        perturbationW = np.random.normal(loc=0.0,scale=self.sigma,size=(self.num_inputs, self.num_outputs, self.k))
        self.ensembleB = np.zeros((self.num_outputs, self.k))
        self.ensembleW = np.zeros((self.num_inputs, self.num_outputs, self.k))
        perturbationB = np.random.normal(loc=0.0,scale=self.sigma,size=(self.num_outputs, self.k))
        for k in range(self.k):
            self.ensembleW[:,:,k]= self.weights[:,:] + perturbationW[:,:,k]
            self.ensembleB[:,k]= self.bias[:] + perturbationB[:,k]
        if len(self.inputs.shape) == 1:
            out = np.zeros((self.num_outputs, self.k))
            for k in range(self.k):
                out[:,k] = np.dot(self.inputs,self.ensembleW[:,:,k]) + self.ensembleB[:,k]
        else:
            out = np.zeros((self.num_outputs, self.k))
            for k in range(self.k):
                out[:,k] = np.dot(self.inputs[:,k],self.ensembleW[:,:,k]) + self.ensembleB[:,k]
        return out

    def backward(self, ensCov,):

        self.lr = lr
        self.ensCov = ensCov


        self.ensembleW -= self.lr * (self.ensembleW @ self.ensCov)
        self.ensembleB -= self.lr * (self.ensembleB @ self.ensCov)
        
        self.weights = np.mean(self.ensembleW, axis=2)
        self.bias = np.mean(self.ensembleB,axis=1)
        return 

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        relu = inputs.copy()
        relu[relu < 0] = 0
        return relu
    def forwardtest(self, inputs):
        self.inputs = inputs
        relu = inputs.copy()
        relu[relu < 0] = 0
        return relu
    def backward(self, fin):
        return 
    def extract(self):
        return

class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = np.zeros((inputs.shape[0],inputs.shape[1]))
        for k in range(inputs.shape[1]):
            self.out[:,k] = exp[:,k]/np.sum(exp[:,k])
        return self.out
    def forwardtest(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, fin):
        return 
    def extract(self):
        return

class DenseLayer:

    def __init__(self, num_inputs, num_outputs, learning_rate, name, num_particles):
        self.sigma = np.sqrt(2/(num_outputs+num_inputs))
        self.weights = np.random.normal(loc=0.0,scale=self.sigma,size=(num_inputs, num_outputs))
        self.bias = np.zeros((num_outputs))
        self.lr = learning_rate
        self.k = num_particles
        self.name = name
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
    
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def forwardtest(self,inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T
        
    def forward(self, inputs):
        self.inputs = inputs
        self.perturbationW = np.random.normal(loc=0.0,scale=self.sigma,size=(self.num_inputs, self.num_outputs, self.k))
        self.perturbationB = np.random.normal(loc=0.0,scale=self.sigma,size=(self.num_outputs, self.k))
        self.ensembleB = np.zeros((self.num_outputs, self.k))
        self.ensembleW = np.zeros((self.num_inputs, self.num_outputs, self.k))
        for k in range(self.k):
            self.ensembleW[:,:,k]= self.weights[:,:] + self.perturbationW[:,:,k]
            self.ensembleB[:,k]= self.bias[:] + self.perturbationB[:,k]
        self.ensembleW = np.dstack((self.ensembleW,self.weights))
        self.ensembleB = np.vstack((self.ensembleB.T,self.bias)).T
        if len(self.inputs.shape) == 1:
            out = np.zeros((self.num_outputs, self.k+1))
            for k in range(self.k+1):
                out[:,k] = np.dot(self.inputs,self.ensembleW[:,:,k]) + self.ensembleB[:,k]
        else:
            out = np.zeros((self.num_outputs, self.k+1))
            for k in range(self.k+1):
                out[:,k] = np.dot(self.inputs[:,k],self.ensembleW[:,:,k]) + self.ensembleB[:,k]
        return out

    def backward(self, fin):
        self.weights -= self.lr*(self.perturbationW@fin)
        self.bias -= self.lr*(self.perturbationB@fin)


    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}


class Convolution2D:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0,self.F):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name

    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out

    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.p
        H = inputs.shape[2]+2*self.p
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)
        WW = (W - self.K)/self.s + 1
        HH = (H - self.K)/self.s + 1
        feature_maps = np.zeros((self.F, WW, HH))
        for f in range(self.F):
            for w in range(WW):
                for h in range(HH):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps

    def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Maxpooling2D:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.pool)/self.s + 1
        new_height = (H - self.pool)/self.s + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(W/self.s):
                for h in range(H/self.s):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w/self.pool, h/self.pool]
        return dx

    def extract(self):
        return 
    
class Flatten:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)
    def extract(self):
        return

