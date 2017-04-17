import theano as th

from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Layer import Layer
from utils.Data import numpy_floatX

__author__ = 'quynhdo'
# a Time Distributed layer
# apply all the elements (as in sequential prediction) of the input in the same corelayer
#
#
#
import theano.tensor as tensor

class TimeDitributed(Layer):
    def __init__(self, idx="0", core_layer=None):
        Layer.__init__(self, id=idx)
        self.core_layer = core_layer


    def compile(self):
        #####
        if self.mask is not None:
            self.core_layer.mask = self.mask
        self.core_layer.id = self.id
        self.core_layer.init_params()
        #####
        if isinstance(self.input, list):
            time_steps = len(self.input)
        else:
            time_steps =  self.input.shape[0]


        def _step(_m, prev):
            self.core_layer.input = _m
            self.core_layer.compile(init_params=False)
            return self.core_layer.output


        result, updates = th.scan(fn=_step,
                              outputs_info=tensor.alloc(numpy_floatX(0.),
                                                               self.input.shape[1],
                                                               self.core_layer.option[Option.OUTPUT_DIM]),
                              sequences=[self.input]
                              )



        #self.output = result.swapaxes(0,1)
        self.output = result
        self.params = self.core_layer.params
        





