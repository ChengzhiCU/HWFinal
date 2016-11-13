import numpy
import numpy as np
import theano
import theano.tensor as T
#### the first three lines should not be ignored
def linear_rectified(x):
    y = T.maximum(T.cast(0., theano.config.floatX), x)
    return(y)

rng = np.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
def drop(input, p=0.5, rng=rng): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
    """            
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


def init_W_b(W, b, rng, n_in, n_out):
    
    if W is None:    
        W_values = numpy.asarray(
            rng.uniform(
                low=-np.sqrt(6./(n_in + n_out)),
                high=np.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function 
    if b is None:
        b_values = numpy.ones((n_out,), dtype=theano.config.floatX) * np.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b
class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, p=0.5, W=None, b=None,
                 activation=linear_rectified):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input
        # end-snippet-1

        W, b = init_W_b(W, b, rng, n_in, n_out)
        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(np.cast[theano.config.floatX](1./p) * output)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()

class DropconnectHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, p , W=None, b=None,
                 activation=linear_rectified, k=25):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of dropping NOT out a weight                   
                           
        :type k: int
        :param k: number of samples for inference                      
                           
        """
        self.input = input
        # end-snippet-1
        if k<1:
            sample = False
        else:
            sample = True
        
        W, b = init_W_b(W, b, rng, n_in, n_out)

        self.W = W
        self.b = b

        lin_output_train = T.dot(input, drop(self.W, p)) + self.b
        train_output = activation(lin_output_train)

        # here we do not drop self.b
        mean_lin_output = T.dot(input, p * self.W) + self.b
        
        if not sample:
            output = activation(mean_lin_output)
        else: #sample from a gaussian (see [Wan13])
            variance_lin_output = np.cast[theano.config.floatX](p * (1.-p)) * T.dot((input * input),(self.W * self.W)) 
            all_samples = srng.normal(avg=mean_lin_output, std=variance_lin_output, size=(k, input.shape[0], n_out), dtype=theano.config.floatX)
            mean_sample = all_samples.mean(axis=0) 
            output = activation(mean_sample)
            # This is necessary because the  
            self.consider_constant = [all_samples]
            

        #is_train is a pseudo boolean theano shared variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, output)
        
        # parameters of the model
        self.params = [self.W, self.b]



