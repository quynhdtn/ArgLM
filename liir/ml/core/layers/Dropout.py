import theano.tensor as tensor

from liir.ml.core.layers.Layer import Layer


__author__ = 'quynhdo'
# a Dropout layer

class Dropout(Layer):
    def __init__(self, input_dim,theano_rng=None, idx="0"):
        Layer.__init__(self, id=idx, input_dim=input_dim)
        self.theano_rng = theano_rng
        self.use_noise = None  #th.shared(numpy_floatX(0.))

    def compile(self):
        self.output = tensor.switch(self.use_noise,
                         (self.input *
                          self.theano_rng.binomial(self.input.shape,
                                        p=0.5, n=1,
                                        dtype=self.input.dtype)),
                         self.input * 0.5)






