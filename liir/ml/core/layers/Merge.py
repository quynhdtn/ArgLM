from theano import tensor

from liir.ml.core.layers.Layer import Layer


__author__ = 'quynhdo'


class Merge(Layer):
    '''
    A Merge layer that merge the output of the previous layers
    There are two modes of concatenation:
    - concat
    - sum
    '''
    def __init__(self, idx=None, layers=None, mode="concat", axis=None):
        Layer.__init__(self, id=idx)
        self.layers = []
        if layers is not None:
            for l in layers:
                self.layers.append(l)
        self.mode = mode
        self.axis = axis

    def add_layer(self, l):
        self.layers.append(l)

    def compile(self):

        input_vec = [l.output for l in self.layers]

        if self.mode == "concat":
            if self.axis is None:
                self.axis = -1
                self.output = tensor.concatenate(input_vec, axis=self.axis)

        if self.mode == "sum":
            if self.axis is None:
                self.axis = 0
                self.output = tensor.sum(input_vec, axis=self.axis)