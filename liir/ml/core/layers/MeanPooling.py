from liir.ml.core.layers.Layer import Layer

__author__ = 'quynhdo'

__author__ = 'quynhdo'
# a Mean Pooling layer

class MeanPooling(Layer):
    def __init__(self, idx = '0'):
        Layer.__init__(self, id=idx)
        self.mask = None

    def compile(self):
        if self.mask is not None:
            x = (self.input * self.mask[:, :, None]).sum(axis=0)
            self.output = x / self.mask.sum(axis=0)[:, None]
        else:
            self.output = self.input.sum(axis=0) / self.input.shape[0]






