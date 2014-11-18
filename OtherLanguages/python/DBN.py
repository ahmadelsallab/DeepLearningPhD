#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets (DBN)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

'''

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM import RBM
from utils import *


class DBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 numpy_rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        
        assert self.n_layers > 0


        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        numpy_rng=numpy_rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)


            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in range(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost

    # def pretrain(self, lr=0.1, k=1, epochs=100):
    #     # pre-train layer-wise
    #     for i in xrange(self.n_layers):
    #         rbm = self.rbm_layers[i]
            
    #         for epoch in xrange(epochs):
    #             layer_input = self.x
    #             for j in xrange(i):
    #                 layer_input = self.sigmoid_layers[j].sample_h_given_v(layer_input)
            
    #             rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
    #             # cost = rbm.get_reconstruction_cross_entropy()
    #             # print >> sys.stderr, \
    #             #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1


    def predict(self, x):
        layer_input = x
        
        for i in range(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out

import os
import gzip
import pickle
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib.request, urllib.parse, urllib.error
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.



    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def test_dbn_MNIST(pretrain_lr=0.1, pretraining_epochs=10, k=1, \
             finetune_lr=0.1, finetune_epochs=10, dataset='mnist.pkl.gz', batch_size=10):

    '''
    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])

    
    rng = numpy.random.RandomState(123)
    '''
    
    datasets = load_data(dataset)

    train_set_x, train_set_y_ = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y_ = datasets[2]
    
    #train_set_y = numpy.zeros(50000 * 10).reshape(50000, 10)
    train_set_y = []
    for example in train_set_y_[0:99]:
        y = numpy.zeros(10)
        y[example] = 1
        train_set_y.append(y)
    train_set_y = numpy.asanyarray(train_set_y)
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    
    rng = numpy.random.RandomState(123)
    
    
    # construct DBN
    dbn = DBN(input=train_set_x[0:99], label=train_set_y, n_ins=28 * 28, hidden_layer_sizes=[1000, 1000, 1000], n_outs=10, numpy_rng=rng)
    

    # pre-training (TrainUnsupervisedDBN)
    print('Pre-train the system...')
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    print('Fine tune the system...')
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    print('Test the system...')
    print(dbn.predict(test_set_x))
    print('Done')



def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200, dataset='mnist.pkl.gz', batch_size=10):

    
    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])

    
    rng = numpy.random.RandomState(123)
    
    '''
    datasets = load_data(dataset)

    train_set_x, train_set_y_ = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y_ = datasets[2]
    
    train_set_y = []
    for example in train_set_y_:
        y = numpy.zeros(10)
        y[example] = 1
        train_set_y.append(y)
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    
    rng = numpy.random.RandomState(123)
    '''
    
    # construct DBN
    #dbn = DBN(input=train_set_x, label=train_set_y, n_ins=28 * 28, hidden_layer_sizes=[1000, 1000, 1000], n_outs=10, numpy_rng=rng)
    dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2, numpy_rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    
    x = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])
    
    
    #print(dbn.predict(test_set_x))
    print(dbn.predict(x))



if __name__ == "__main__":
    test_dbn_MNIST()
