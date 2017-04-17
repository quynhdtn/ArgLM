from theano import config
import numpy as np
import theano as th

from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Layer import Layer
from utils.Variant import _p


__author__ = 'quynhdo'
# an embedding layer

class Embedding(Layer):
    '''
    Implementation of Embedding layer
    '''
    def __init__(self, input_dim, output_dim,  idx="0",
                 we_dict=None , map=None):
        '''

        :param input_dim: Input dimension -- usually it is the vocabulary size
        :param output_dim: Ouput dimension -- the dimension of the output embeddings
        :param idx: id of the layer
        :param we_dict: if we_dict is not None, the layer weights are initialzed
        :param map:
        :return:
        '''
        Layer.__init__(self, id=idx, input_dim=input_dim, output_dim=output_dim)
        self.W = None
        self.we_dict = we_dict
        self.map = map

    def compile(self):
        if self.we_dict is None:
            randn = np.random.rand(self.option[Option.INPUT_DIM],
                                  self.option[Option.OUTPUT_DIM])

            self.W = th.shared((0.01 * randn).astype(config.floatX), name=_p(self.id, "W"))

        else:
            randn = self.initialize()
            self.W = th.shared((0.01 * randn).astype(config.floatX), name=_p(self.id, "W"))

        self.params[_p(self.id, "W")] = self.W
        self.output = (self.W[self.input.flatten()]).reshape([self.input.shape[0],
                                                self.input.shape[1],
                                               self.option[Option.OUTPUT_DIM]])

    def initialize(self):
        lst = [np.random.rand(self.option[Option.OUTPUT_DIM]) for i in range(self.option[Option.INPUT_DIM])]

        for w in self.map.keys():
            if w in self.we_dict.full_dict.keys():
                arr = self.we_dict.get_we(w)
                lst[self.map[w] + 1] = arr

        lst = np.asarray(lst)
        return lst





