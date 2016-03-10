""" load_mnist.py
A module that loads the MNIST image data. load_data() loads the data
from the mnist.pkl.gz data and returns the training_data,
validation_data, and test_data.
Skeleton code from neuralnetworksanddeeplearning.com
Last modified: 9/10/2016 Bill Byung Gu Cho
"""

import numpy as np
import gzip
import cPickle
import matplotlib.pyplot as plt


def load_data():
    f = gzip.open('../data/mnist.pkl.gz', mode='rb')
    (trainig_data, validation_data, test_data) = cPickle.load(f)
    f.close()
    return (trainig_data, validation_data, test_data)

def load_data_wrapper():
    """ The loaded training_data, validation_data, and test_data are modified in a way that eases the neural network process. The input data is recompiled as a list of tuples as [(x_1, y_1), (x_2, y_2) ... (x_n, y_n) ...], where x_n is a 784x1 image array and y_n is a 10x1 label array for the n-th image. For the training_data, length of the list size is 50,000."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_labels = [vectorize_label(x) for x in tr_d[1]]
    training_data = zip(training_inputs, training_labels)
    validation_inputs = [np.reshape(y, (784,1)) for y in va_d[0]]
    validation_labels = [vectorize_label(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_labels)
    test_inputs = [np.reshape(z,(784,1)) for z in te_d[0]]
    test_labels = [vectorize_label(z) for z in te_d[1]]
    test_data = zip(test_inputs, test_labels)
    return (training_data, validation_data, test_data)


def vectorize_label(label):
    label_vec = np.zeros((10,1))
    label_vec[label] = 1.0
    return label_vec



