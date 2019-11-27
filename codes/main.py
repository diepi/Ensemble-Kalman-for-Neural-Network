#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:07:19 2019

@author: diepy
"""

import numpy as np
import pickle
import sys
from time import time
from Loss import cross_entropy
from Layers import DenseLayer,Softmax,ReLu


class DNN:
    def __init__(self):
        
        self.layers = []
        self.layers.append(DenseLayer(num_inputs=784, num_outputs=128, learning_rate=0.001, name='dense1',num_particles=10))
        self.layers.append(ReLu())
        self.layers.append(DenseLayer(num_inputs=128, num_outputs=10, learning_rate=0.001, name='dense2',num_particles=10))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)
        
        
    def train(self, training_data, training_label, batch_size, epoch):
        total_acc = 0
        histacc = np.array([])
        histloss= np.array([])
        for e in range(epoch):
            for batch_index in range(0, training_data.shape[0], batch_size):
                # batch input
                if batch_index + batch_size < training_data.shape[0]:
                    data = training_data[batch_index:batch_index+batch_size]
                    label = training_label[batch_index:batch_index + batch_size]
                else:
                    data = training_data[batch_index:training_data.shape[0]]
                    label = training_label[batch_index:training_label.shape[0]]
                loss = 0
                acc = 0
                start_time = time()
                for b in range(batch_size):
                    x = data[b]
                    y = label[b]

                    # forward pass
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x)
                        x = output
                    
                    ensweight = output[:,output.shape[1]-1]
                    ensError = np.zeros((output.shape[0],output.shape[1]-1))
                    for k in range(output.shape[1]-1):
                        ensError[:,k] = output[:,k] - ensweight
    
                    loss += cross_entropy(y, ensweight)
                    dloss = -y/(ensweight+0.005)
                    if np.argmax(ensweight) == np.argmax(y):
                        acc += 1
                        total_acc += 1
                    # ensemble maths
                    fin = ensError.T@dloss
                    
                    for l in range(self.lay_num-1, -1, -1):
                        self.layers[l].backward(fin)
                # time
                end_time = time()
                batch_time = end_time-start_time
                remain_time = (training_data.shape[0]*epoch-batch_index-training_data.shape[0]*e)/batch_size*batch_time
                hrs = int(remain_time)/3600
                mins = int((remain_time/60-hrs*60))
                secs = int(remain_time-mins*60-hrs*3600)
                # result
                loss /= batch_size
                batch_acc = float(acc)/float(batch_size)
                training_acc = float(total_acc)/float(training_data.shape[0]*e + (batch_index+batch_size))
                print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ==='.format(e,epoch,batch_index+batch_size,loss,batch_acc,training_acc,int(hrs),int(mins),int(secs)))
                histacc = np.append(histacc,batch_acc)
                histloss= np.append(histloss, loss)
                
        obj = []
        for i in range(self.lay_num):
            cache = self.layers[i].extract()
            obj.append(cache)
        with open("weights_file2.txt", 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return histacc, histloss
    
    
    def test_with_pretrained_weights(self, data, label, test_size):
        with open("weights_file2.txt", 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].feed(b[0]['dense1.weights'], b[0]['dense1.bias'])
        self.layers[2].feed(b[2]['dense2.weights'], b[2]['dense2.bias'])
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size)/float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size)/float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                #sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
                #sys.stdout.write("\b" * (toolbar_width-st+2))
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forwardtest(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc)/float(test_size)))
        

    def test(self, data, label, test_size):
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size)/float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size)/float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                #sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
                #sys.stdout.write("\b" * (toolbar_width-st+2))
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forwardtest(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc)/float(test_size)))


from keras.datasets import mnist
from keras.utils import np_utils   
# data: shuffled and split between train and test sets, loading and using the Keras mnist dataset
(input_X_train, output_Y_train), (input_X_test, output_Y_test) = mnist.load_data()

# use 60000 images for training, 10000 for validation test
input_X_train = input_X_train.reshape(60000, 784)
input_X_test = input_X_test.reshape(10000, 784)
input_X_train = input_X_train.astype('float32')
input_X_test = input_X_test.astype('float32')

# normalisation of the pixel values from 0-255 range to 0-1 range 
input_X_train /= 255
input_X_test /= 255

# convert class vectors to binary class matrices
output_Y_train = np_utils.to_categorical(output_Y_train, 10)
output_Y_test = np_utils.to_categorical(output_Y_test, 10)

DNN_1_EnKI = DNN()
DNN_1_EnKI.__init__()
hist10 = DNN_1_EnKI.train(input_X_train,output_Y_train,500,10)



from scipy.interpolate import interp1d
x = interp1d([0,len(hist10[0])],[0,10])
xvalues=x(np.arange(0,len(hist10[0])))

import matplotlib.pyplot as plt

plt.plot(xvalues,hist10[0])

DNN_1_EnKI.test_with_pretrained_weights(input_X_test, output_Y_test, 10000)

for k in range(1,11):
    print(hist10[0][120*k-1])
