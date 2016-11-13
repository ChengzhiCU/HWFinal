import os
import sys
import time

import numpy
import numpy as np
import cPickle


import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import bn
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from drop_m import DropoutHiddenLayer, DropconnectHiddenLayer, linear_rectified
from drop_fu import DropoutLayer

from load_data import shared_dataset
from load_ag_news import load_ag_news
theano.config.floatX = 'float32'
#the following get from the dropout_master mlp.py and drop.py from fuzhao

def PLeRU(x, alpha):
    y = T.maximum(x * alpha, x)
    return(y)
def ReLu(x):
    y = T.maximum(T.cast(0., theano.config.floatX), x)
    return(y)

rng = np.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
def drop(input, p, rng=rng): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
    """            
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask
class fcLayer(object):
    def __init__(self,rng,is_train,input,n_in,n_out,dropout_rate=0.5,W=None,
        b=None,activation = ReLu):
        self.input = input
        p=dropout_rate

        W = numpy.asarray(
            numpy.random.normal(loc=0.0,scale=0.05,size=(n_in,n_out)),
            dtype = theano.config.floatX
            )
        self.W = theano.shared(W, borrow=True)

        b = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value = b, borrow=True)

        linearOutput = T.dot(self.input,self.W) + self.b
        train_output = drop(input = np.cast[theano.config.floatX](1./p) * linearOutput, p=dropout_rate,rng=rng)
        tempOutPut = T.switch(T.neq(is_train, 0), train_output, linearOutput)

        bnOutput = bn.batch_normalization(
            inputs=tempOutPut,
            gamma=1.,
            beta=0,
            mean=T.mean(tempOutPut),
            std=T.std(tempOutPut)
        )

        self.output = activation(bnOutput)

        self.params = [self.W, self.b]


        



class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #alpha_value = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX) + 0.25
        #self.alpha = theano.shared(value=alpha_value, borrow=True)
        linearOutput = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        bnOutput = bn.batch_normalization(
            inputs=linearOutput,
            gamma=1.,
            beta=0,
            mean=T.mean(linearOutput),
            std=T.std(linearOutput)
        )
        self.output = ReLu(bnOutput)

        # store parameters of this layer
        self.params = [self.W, self.b]

class PoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, poolsize=(2, 2)):
        """
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        self.input = input
        # downsample each feature map individually, using maxpooling
        self.output = downsample.pool.pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True,
            mode='max'
        )

def cnnText(initial_learning_rate=0.0001, initial_momentum=0.5, n_epochs=100,
                    dataset='mnist.pkl.gz',
                    nkerns=[64, 64], batch_size=5):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    print 'loading data'
    datasets = load_ag_news()
    print 'finished'
    train_x, train_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    test_x, test_y = datasets[1]

    train_set_x, train_set_y = shared_dataset([train_x , train_y])
    test_set_x, test_set_y = shared_dataset([test_x, test_y])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_train_batches = 2000 / batch_size
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_valid_batches = 1000 / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_test_batches = 1000 / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    total_len = 231
    print 'total_len = ',total_len
    layer0_input = x.reshape((batch_size, 1, total_len, 70))

    layer_conv0 = ConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, total_len, 70),
        filter_shape=(256, 1, 7, 1)
    )

    layer_pool0 = PoolLayer(
        input=layer_conv0.output,
        poolsize=(3, 1)
    )

    layer_conv1 = ConvLayer(
        rng,
        input=layer_pool0.output,
        image_shape=(batch_size, 256, (total_len-6)/3, 70),
        filter_shape=(256, 256, 7, 1)
    )

    layer_pool1 = PoolLayer(
        input=layer_conv1.output,
        poolsize=(3, 1)
    )

    layer_conv20 = ConvLayer(
        rng,
        input=layer_pool1.output,
        image_shape=(batch_size, 256, ((total_len-6)/3 -6)/3, 70),
        filter_shape=(256, 256, 3, 1)
    )

    layer_conv21 = ConvLayer(
        rng,
        input=layer_conv20.output,
        image_shape=(batch_size, 256, ((total_len-6)/3 -6)/3 -2, 70),
        filter_shape=(256, 256, 3, 1)
    )

    layer_conv22 = ConvLayer(
        rng,
        input=layer_conv21.output,
        image_shape=(batch_size, 256, ((total_len-6)/3 -6)/3 -4, 70),
        filter_shape=(256, 256, 3, 1)
    )

    layer_conv23 = ConvLayer(
        rng,
        input=layer_conv22.output,
        image_shape=(batch_size, 256, ((total_len-6)/3 -6)/3 -6, 70),
        filter_shape=(256, 256, 3, 1)
    )
    
    layer_pool2 = PoolLayer(
        input=layer_conv23.output,
        poolsize=(3, 1)
    )

    layer_fc0_input = layer_pool2.output.flatten(2)    

    layer_fc0 = fcLayer(
        rng,
        is_train=is_train,
        input=layer_fc0_input,
        n_in=(((total_len-6)/3 -6)/3 -8)/3*70*256,
        n_out=1024,
        activation=ReLu,
        dropout_rate = 0.5
    )
    
    
    layer_fc1 = fcLayer(
        rng,
        is_train=is_train,
        input=layer_fc0.output,
        n_in=1024,
        n_out=1024,
        activation=ReLu,
        dropout_rate = 0.5
    )

    # classify the values of the fully-connected sigmoidal layer
    layer_softmax = LogisticRegression(input=layer_fc1.output, n_in=1024, n_out=4)

    # the cost we minimize during training is the NLL of the model
    cost = layer_softmax.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer_softmax.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )
    '''
    validate_model = theano.function(
        [index],
        layer_softmax.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)
        }
    )
    '''
    # create a list of all model parameters to be fit by gradient descent
    params = layer_conv0.params + layer_conv1.params + layer_conv20.params + layer_conv21.params + layer_conv22.params + layer_conv23.params\
        + layer_fc0.params + layer_fc1.params + layer_softmax.params

    learning_rate = theano.shared(numpy.cast[theano.config.floatX](initial_learning_rate))
    initial_learning_rate_val=initial_learning_rate
	
	# momentum method
    assert initial_momentum >= 0. and initial_momentum < 1.
    
    momentum =theano.shared(numpy.cast[theano.config.floatX](initial_momentum), name='momentum')
    
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))

    train_model = theano.function(
        [index],
        layer_softmax.errors(y),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](1)
        }
    )

    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant

                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    train=[]
    valid=[]
    test=[]
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_losses = 0.
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            train_losses = train_losses + train_model(minibatch_index)

            # test it on the test set
        test_losses = [
            test_model(i)
            for i in xrange(n_test_batches)
        ]
        test_score = numpy.mean(test_losses)
        print(('     epoch %i, minibatch %i/%i, test error of '
               ' model %f %%') %
              (epoch, minibatch_index + 1, n_train_batches,
               test_score * 100.))
        test.append((epoch, minibatch_index + 1, n_train_batches,
               test_score * 100.))            #
                    
        print('epoch %i, training error %f %%' %
            (epoch, train_losses * 100. / n_train_batches))
        train.append(train_losses * 100)
        
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # adaption of learning rate    
        new_learning_rate = learning_rate.get_value() * 0.985
        #new_learning_rate = initial_learning_rate_val*1.1 /(1+0.1*epoch)
        #get from tutorial p48
        learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print train
    print test

    f=file('log_BN1_wb.txt','wb')
    cPickle.dump((train,test),f)
    f.close()
    f1=file('log_BN1_w.txt','b')
    cPickle.dump((train,test),f1)
    f1.close()
    '''
    params = [layer_conv0.W.get_value(), layer_conv0.b.get_value(),
        layer_conv1.W.get_value(), layer_conv1.b.get_value(),
        layer_conv2.W.get_value(), layer_conv2.b.get_value(),
        layer_conv3.W.get_value(), layer_conv3.b.get_value(),
        layer_fc0.W.get_value(), layer_fc0.b.get_value(),
        layer_fc1.W.get_value(), layer_fc1.b.get_value(),
        layer_softmax.W.get_value(), layer_softmax.b.get_value(),
        random_l.W.get_value()]
    '''
    f = file('bn0.save', 'wb')
    cPickle.dump(params, f)

    spath = 'mao_drop.txt'
    fii=open(spath,"w")
    cPickle.dump(best_validation_loss,fii)
    cPickle.dump(best_iter + 1,fii)
    cPickle.dump(test_score * 100.,fii)
    cPickle.dump((end_time - start_time) / 60.,fii)

    
    f.close()
if __name__ == '__main__':
    cnnText()


def experiment(state, channel):
    cnnText(state.learning_rate, dataset=state.dataset)
