from liir.ml.core.layers.Layer import Layer
import theano as th
__author__ = 'quynhdo'


class Reverse(Layer):
    # reverse the input
    def __init__(self, idx="0",  times=2):
        Layer.__init__(self, id=idx)
        self.times = times


    def compile(self):
        axes = [self.input.ndim - 1]
        slices = [slice(None, None, -1) if i in axes else slice(None, None, None) for i in range(self.input.ndim)]
        self.output = self.input[slices]








