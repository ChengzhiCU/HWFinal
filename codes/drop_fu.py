import numpy
import theano
import theano.tensor as T

def relu(x):
    y = T.maximum(T.cast(0., theano.config.floatX), x)
    return(y)
    
class DropoutLayer(object):
    def __init__(self, rng, is_train, input, p=0.5):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        self.output = T.switch(T.neq(is_train, 0), input * mask, input * p)
