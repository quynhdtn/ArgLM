from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano as th

from liir.ml.core.layers.Layer import Layer


__author__ = 'quynhdo'
class Input(Layer):
    '''
    Implementation of input layer.
    '''
    def __init__(self, input_dim, idx=None):
        '''

        :param input_dim: Input dimension
        :param idx: id of the layer
        :return:
        '''
        Layer.__init__(self, id=idx, input_dim=input_dim, output_dim=input_dim)

    def compile(self):
        self.output = self.input

class NoisyInput(Layer):
    '''
    Implementation of noisy input layer. This can be used in denoising models.
    Noise is automatically added to the input.
    The type of noise can be change by change the 'compile' function.
    '''
    def __init__(self, input_dim, idx=None, corruption_level=0.1, rng=None, theano_rng=None):
        '''

        :param input_dim: Input dimension
        :param idx: Id of the layer
        :param corruption_level: corruption level which is a parameter for the compile function
        :param rng: Numpy RandomState object that could be initialized in another class, then passed to this class
        :param theano_rng: Theano RandomStreams object that could be initialized in another class, then passed to this class
        :return:
        '''
        Layer.__init__(self, id=idx, input_dim=input_dim)
        self.corruption_level = corruption_level
        if rng is not None:
            self.rng=rng
        else:
            self.rng = np.random.RandomState(123456)
        if theano_rng is not None:
            self.theano_rng=theano_rng
        else:
            self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

    def compile(self):
        '''
        this function can be changed or overwitten to change the noise creation
        By default, it is binominal noise, which turn some input nodes to 0
        :return:
        '''
        self.output = self.theano_rng.binomial(size=self.input.shape, n=1, p=1 - self.corruption_level, dtype=th.config.floatX) * self.input