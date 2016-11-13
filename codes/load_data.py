import cPickle
import gzip
import os
import sys
import copy

import numpy
import numpy

import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')

def load_cifar10_yuv():
    
    print '... loading data & transforming to YUV'
    
    dir = os.path.join(
        os.path.split(__file__)[0],
        #"..",
        "dataset",#
        "cifar-10-python",
        "cifar-10-batches-py"
    )
    
    f = open(os.path.join(dir, 'data_batch_1'), 'rb')
    dict = cPickle.load(f)
    trainx = dict['data']
    trainy = dict['labels']
    f.close()

    #print '!!!', trainx.size

    for spIdx in xrange(trainx.shape[0]): #trainx.size
        #print spIdx,trainx[spIdx]

        sample = trainx[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        trainx[spIdx] = spOut

        #print trainx[spIdx]

    


    f = open(os.path.join(dir, 'data_batch_2'), 'rb')
    dict = cPickle.load(f)
    trainx_temp = dict['data']
    trainy_temp = dict['labels']
    f.close()

    for spIdx in xrange(trainx_temp.shape[0]): #trainx_temp.size
        #print trainx_temp[spIdx]
        sample = trainx_temp[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        trainx_temp[spIdx] = spOut

    trainx = numpy.concatenate((trainx, trainx_temp), axis = 0)
    trainy = numpy.concatenate((trainy, trainy_temp), axis = 0)
    
    


    f = open(os.path.join(dir, 'data_batch_3'), 'rb')
    dict = cPickle.load(f)
    trainx_temp = dict['data']
    trainy_temp = dict['labels']
    f.close()

    for spIdx in xrange(trainx_temp.shape[0]): #trainx_temp.size
        #print trainx_temp[spIdx]
        sample = trainx_temp[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        trainx_temp[spIdx] = spOut

    trainx = numpy.concatenate((trainx, trainx_temp), axis = 0)
    trainy = numpy.concatenate((trainy, trainy_temp), axis = 0)



    
    f = open(os.path.join(dir, 'data_batch_4'), 'rb')
    dict = cPickle.load(f)
    trainx_temp = dict['data']
    trainy_temp = dict['labels']
    f.close()

    for spIdx in xrange(trainx_temp.shape[0]): #trainx_temp.size
        #print trainx_temp[spIdx]
        sample = trainx_temp[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        trainx_temp[spIdx] = spOut

    trainx = numpy.concatenate((trainx, trainx_temp), axis = 0)
    trainy = numpy.concatenate((trainy, trainy_temp), axis = 0)
    


    f = open(os.path.join(dir, 'data_batch_5'), 'rb')
    dict = cPickle.load(f)
    trainx_temp = dict['data']
    trainy_temp = dict['labels']
    f.close()

    for spIdx in xrange(trainx_temp.shape[0]): #trainx_temp.size
        #print trainx_temp[spIdx]
        sample = trainx_temp[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        trainx_temp[spIdx] = spOut

    trainx = numpy.concatenate((trainx, trainx_temp), axis = 0)
    trainy = numpy.concatenate((trainy, trainy_temp), axis = 0)
    
    f = open(os.path.join(dir, 'test_batch'), 'rb')
    dict = cPickle.load(f)
    testx = dict['data']
    testy = dict['labels']
    f.close()

    for spIdx in xrange(testx.shape[0]): #testx.size
        #print testx[spIdx]
        sample = testx[spIdx].tolist()
        spYUV = copy.copy(sample)
        for i in range(1024):
            spYUV[i] = 0.299 * sample[i] + 0.587 * sample[i+1024] + 0.114 * sample[i+2048]
            spYUV[i+1024] = - 0.1687 * sample[i] - 0.3313 * sample[i+1024] + 0.5 * sample[i+2048] + 128
            spYUV[i+2048] = 0.5 * sample[i] - 0.4187 * sample[i+1024] - 0.0813 * sample[i+2048] + 128

        spOut = numpy.array(spYUV)
        testx[spIdx] = spOut

    rval = [(trainx[0:40000], trainy[0:40000]), (trainx[40000:50000], trainy[40000:50000]),
            (testx, testy)]
    return rval


def load_cifar10():
    
    print '... loading data'
    
    dir = os.path.join(
        os.path.split(__file__)[0],
        #"..",
        "dataset",#
        "cifar-10-python",
        "cifar-10-batches-py"
    )
    
    f = open(os.path.join(dir, 'data_batch_1'), 'rb')
    dict = cPickle.load(f)
    trainx = dict['data']
    trainy = dict['labels']
    f.close()
    
    f = open(os.path.join(dir, 'data_batch_2'), 'rb')
    dict = cPickle.load(f)
    trainx = numpy.concatenate((trainx, dict['data']), axis = 0)
    trainy = numpy.concatenate((trainy, dict['labels']), axis = 0)
    f.close()
    
    f = open(os.path.join(dir, 'data_batch_3'), 'rb')
    dict = cPickle.load(f)
    trainx = numpy.concatenate((trainx, dict['data']), axis = 0)
    trainy = numpy.concatenate((trainy, dict['labels']), axis = 0)
    f.close()
    
    f = open(os.path.join(dir, 'data_batch_4'), 'rb')
    dict = cPickle.load(f)
    trainx = numpy.concatenate((trainx, dict['data']), axis = 0)
    trainy = numpy.concatenate((trainy, dict['labels']), axis = 0)
    f.close()
    
    f = open(os.path.join(dir, 'data_batch_5'), 'rb')
    dict = cPickle.load(f)
    trainx = numpy.concatenate((trainx, dict['data']), axis = 0)
    trainy = numpy.concatenate((trainy, dict['labels']), axis = 0)
    f.close()
    
    f = open(os.path.join(dir, 'test_batch'), 'rb')
    dict = cPickle.load(f)
    testx = dict['data']
    testy = dict['labels']
    f.close()

    rval = [(trainx[0:40000], trainy[0:40000]), (trainx[40000:50000], trainy[40000:50000]),
            (testx, testy)]
    return rval

def load_mnist():
    dataset = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "dataset",
        "mnist.pkl.gz"
    )

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    rval = [train_set, valid_set, test_set]
    return rval
    
 