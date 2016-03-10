"""
DNN_sigmoid.py:
Calculates the deep learning neural network for handwritten digit recognition
using sigmoid activation function
Skeleton code from neuralnetworksanddeeplearning.com
Last modified: 9/10/2015 Bill Byung Gu Cho
"""

import random
import json
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """Initialize the network object. Number of layers, Number of neurons for each layer, wieghts, and biases are initialized."""
        self.no_layers = len(sizes)
        self.no_neuron = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(x,y) for (x,y) in zip(sizes[1:], sizes[:-1])]
        # self.biases = [np.zeros((x,1)) for x in sizes[1:]]
        # self.weights = [np.zeros((x,y)) for (x,y) in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        """ Return the output of the network given the input ``a`` """
        for (b,w) in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train the network by updating the network recursively using ``update_mini_batch`` and training rate ``eta`` for each mini batch and looping this over the epochs"""
        n = len(training_data)
        for ii in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[jj:jj+mini_batch_size] for jj in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) 
            # print self.biases[-1] 
            if test_data:
                perf = [np.argmax(y)==np.argmax(self.feedforward(x)) for (x,y) in test_data]
                print "Epoch {0}: {1} / {2}".format(ii, sum(perf), len(test_data))
            else:
                print "Epoch {0} complete.".format(ii)
        

    def update_mini_batch(self, mini_batch, eta):
        """ Update the weights and biases based on the Cost function gradient respect to the weights and biases averaged over the mini batch"""
        tot_dw = [np.zeros(w.shape) for w in self.weights]
        tot_db = [np.zeros(b.shape) for b in self.biases]
        for (x,y) in mini_batch:
            (nabla_C_w, nabla_C_b) = self.backprop(x,y)
            # 1/0
            # self.weights = [w-eta/len(mini_batch)*dc_dw for (w,dc_dw) in zip(self.weights,nabla_C_w)]
            # self.biases = [b-eta/len(mini_batch)*dc_db for (b,dc_db) in zip(self.biases,nabla_C_b)]
            tot_dw = [w+eta/len(mini_batch)*dw for (w,dw) in zip(tot_dw, nabla_C_w)]
            tot_db = [b+eta/len(mini_batch)*db for (b,db) in zip(tot_db, nabla_C_b)]
        self.weights = [w-dw for (w,dw) in zip(self.weights,tot_dw)]
        self.biases = [b-db for (b,db) in zip(self.biases,tot_db)]
        # for kk in xrange(len(self.weights)):
        #     self.weights[kk] = self.weights[kk] - avg_dw[kk]
        #     self.biases[kk] = self.biases[kk] - avg_db[kk]
        

    def backprop(self, x, y):
        """ Calculate Cost function gradient with respect to the weights and biases given the input x and label y. """
        # feedforward and get the weighted input z and activation output a at each layer
        aw_b = []
        a = [x]
        for (b,w) in zip(self.biases, self.weights):
            aw_b.append(np.dot(w,a[-1])+b)
            a.append(sigmoid(aw_b[-1]))
        nabla_C_a = self.dC_da(a[-1], y)
        # output_activations = self.feedforward(x)
        # nabla_C_a = self.dC_da(output_activations, y)
        # back propagate to get the delta at each layer
        delta_layer = [nabla_C_a * sigmoid_prime(aw_b[-1])]
        for (w,z) in zip(list(reversed(self.weights))[:-1], list(reversed(aw_b))[1:]):
            delta_layer.insert(0, np.dot(w.transpose(), delta_layer[0])*sigmoid_prime(z))
        # calculate the Cost function gradient with respect to the baises and weights
        nabla_C_b = delta_layer
        nabla_C_w = []
        for (act,err) in zip(a[:-1],delta_layer):
            nabla_C_w.append(np.dot(err,act.transpose()))
        return (nabla_C_w, nabla_C_b)

    def dC_da(self, output_activations, y):
        """ \partial C / \partial output activation """
        return (output_activations-y)



#### functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # return np.exp(-z)/(1.0+np.exp(-z))**2
    return sigmoid(z)*(1-sigmoid(z))

