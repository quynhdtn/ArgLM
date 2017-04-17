import theano as th
import numpy as np

from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Layer import Layer
from utils.Functions import WeightInit, getFunction
from utils.Variant import _p


__author__ = 'quynhdo'


class Dense (Layer):
    ####
    # Dense Layer, a normal neural network layer,
    # the output is computed from the input with an activation an a transfer function (usually dot)
    ####
    def __init__(self, input_dim, output_dim,
                 w_lamda=1.0, rng=None,  idx="0", activation="sigmoid",
                 transfer="dot_transfer"):
        Layer.__init__(self, idx=idx, input_dim=input_dim, output_dim=output_dim)

        self.activation = activation
        self.transfer = transfer

        self.W = None
        self.b = None

        # params to initialize weights
        self.w_lamda = w_lamda
        self.rng = rng  # numpy random


    def init_params(self, initial_w=None, initial_b=None):
        # init params
        if initial_w is not None:
            self.W = initial_w
        else:
            self.W = th.shared(value=np.asarray(WeightInit(self.option[Option.INPUT_DIM],
                                                           self.option[Option.OUTPUT_DIM],
                                                           self.w_lamda, self.rng), dtype=th.config.floatX),
                                                           name=_p(self.id,"W"), borrow=True)
            self.params[_p(self.id, "W")] = self.W


        if initial_b is not None:
            self.b = initial_b
        else:
            self.b = th.shared(value=np.zeros(self.option[Option.OUTPUT_DIM], dtype=th.config.floatX), name=_p(self.id,"b"), borrow=True)

            self.params[ _p(self.id, "b")]=self.b

    def compute_output(self):
        # compute output
        self.output = getFunction(self.activation)(getFunction(self.transfer)(self.input, self.W, self.b))
