from theano import tensor
import numpy as np
import theano as th

from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Layer import Layer
from utils.Data import ortho_weight, numpy_floatX
from utils.Variant import _p


__author__ = 'quynhdo'


class LSTM(Layer):
    def __init__(self, input_dim, output_dim, idx="0", activation="sigmoid", return_sequences=False, need_swap_axis = False):
        Layer.__init__(self, id=id, input_dim=input_dim, output_dim=output_dim)

        self.activation = activation
        self.W = None
        self.U = None
        self.V = None
        self.b = None
        self.return_sequences = return_sequences
        self.need_swap_axis = need_swap_axis

    def recurrence(self, state_below, mask=None):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, self.U)
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, self.option[Option.OUTPUT_DIM]))
            f = tensor.nnet.sigmoid(_slice(preact, 1, self.option[Option.OUTPUT_DIM]))
            o = tensor.nnet.sigmoid(_slice(preact, 2, self.option[Option.OUTPUT_DIM]))
            c = tensor.tanh(_slice(preact, 3, self.option[Option.OUTPUT_DIM]))

            c = f * c_ + i * c
            if m_ is not None:

                c1= m_[:, None] * c
                c2= (1. - m_)[:, None] * c_
                c = c1+c2

            h = o * tensor.tanh(c)
            if m_ is not None:
                h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return tensor.cast(h, th.config.floatX), tensor.cast(c, th.config.floatX)

        state_below = (tensor.dot(state_below, self.W) +
                       self.b)

        dim_proj = self.option[Option.OUTPUT_DIM]
        rval, updates = th.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj),
                                                  tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj)],
                                    name=_p(self.id, '_layers'),
                                    n_steps=nsteps)
        return rval[0]  # return hidden values

    def compile(self):
        self.W = th.shared(np.concatenate([ortho_weight(self.option[Option.INPUT_DIM],self.option[Option.OUTPUT_DIM] ),
                           ortho_weight(self.option[Option.INPUT_DIM],self.option[Option.OUTPUT_DIM] ),
                           ortho_weight(self.option[Option.INPUT_DIM],self.option[Option.OUTPUT_DIM] ),
                           ortho_weight(self.option[Option.INPUT_DIM],self.option[Option.OUTPUT_DIM] )], axis=1),
                           name= _p(self.id, 'W'))  # W_i, W_f, W_c, W_o
        self.params[_p(self.id, 'W')] = self.W
        self.U = th.shared (np.concatenate([ortho_weight(self.option[Option.OUTPUT_DIM],self.option[Option.OUTPUT_DIM]),
                               ortho_weight(self.option[Option.OUTPUT_DIM],self.option[Option.OUTPUT_DIM] ),
                               ortho_weight(self.option[Option.OUTPUT_DIM],self.option[Option.OUTPUT_DIM] ),
                               ortho_weight(self.option[Option.OUTPUT_DIM],self.option[Option.OUTPUT_DIM] )], axis=1),
                             name=_p(self.id, 'U'))  # U_i, U_f, U_c, U_o
        self.params[_p(self.id, 'U')] = self.U
        self.b = th.shared(np.zeros((4 * self.option[Option.OUTPUT_DIM],),dtype=th.config.floatX),
                           name=_p(self.id, 'b'))   # b_i, b_f, b_c and b_o are bias vectors
        self.params[_p(self.id, 'b')] = self.b

        if self.need_swap_axis:
            self.input = self.input.swapaxes(0,1)

        self.output = self.recurrence(self.input, mask=self.mask)
        if not self.return_sequences:
            self.output = self.output[-1]


