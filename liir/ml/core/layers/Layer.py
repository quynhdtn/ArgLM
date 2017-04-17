from collections import OrderedDict

from liir.ml.core.options.Option import Option


__author__ = 'quynhdo'


class Layer(object):
    ####
    # Implementation of Layer basic class
    ####
    def __init__(self, idx="0", **kwargs):
        self.id = idx  # id of the layer. is important when using in a Model.
        self.input = None  # input of the layer, will be assigned when used in a model
        self.output = None  # output of the layer
        self.params = OrderedDict() # params of the layer

        self.option = Option(for_layer=True)  # options for Layer
        self.mask = None  # mask of the layer. Used for sequence data

        if kwargs is not None:
            for k in kwargs.keys():
                self.option[k] = kwargs[k]

    def init_params(self):
        # init params
        pass

    def compute_output(self):
        # compute output
        self.output = self.input

    def compile(self, init_params=True):
        # in compile function, we should initialize params, and process output from input
        #
        if init_params:
            self.init_params()
        self.compute_output()




