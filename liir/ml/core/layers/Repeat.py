from theano import tensor
from liir.ml.core.layers.Layer import Layer
import theano as th
__author__ = 'quynhdo'


class Repeat(Layer):
    '''
    A Repeat layer repeating the input x times
    '''
    def __init__(self, idx="0",  times=2):
        Layer.__init__(self, id=idx)
        self.times = times


    def compile(self):

        x = self.input.dimshuffle((0, 'x', 1))
        self.output = tensor.extra_ops.repeat(x, self.times, axis=1)





