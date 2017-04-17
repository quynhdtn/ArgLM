__author__ = 'quynhdo'
import numpy as np

from liir.ml.core.options.Option import Option
from liir.ml.core.layers import Dense
from liir.ml.core.layers.Input import Input, NoisyInput
from liir.ml.core.layers.Model import Sequential


__author__ = 'quynhdo'


class AutoEncoder(Sequential):
    def __init__(self, input_dim, hidden_dim, loss="ce", optimizer="ada"):

        Sequential.__init__(self, prediction_type="matrix")
        self.add_layer(Input(input_dim=input_dim))
        for nd in hidden_dim:
            self.add_layer(Dense(input_dim=self.layers[-1].option[Option.OUTPUT_DIM], output_dim=nd, activation="sigmoid"))
        self.add_layer(Dense(input_dim=self.layers[-1].option[Option.OUTPUT_DIM], output_dim=input_dim, activation="sigmoid"))
        self.compile(loss=loss, optimizer=optimizer)


class StandardAutoEncoder(Sequential):
    # an autoencoder with shared weights
    def __init__(self, input_dim, hidden_dim, loss="ce", optimizer="ada"):

        Sequential.__init__(self, prediction_type="matrix")
        l1 = Dense(input_dim=input_dim, output_dim=hidden_dim, activation="sigmoid")
        l2 = Dense(input_dim=hidden_dim, output_dim=input_dim, activation="sigmoid", )
        self.add_layer(l1)
        self.add_layer(l2)
        self.compile(loss=loss, optimizer=optimizer)

    def compile_layers(self):
        print("Compiling Layer ",0)
        self.layers[0].input = self.input
        self.layers[0].compile()
        print("Compiling Layer ",1)
        self.layers[1].init_params(initial_w=self.layers[0].W.T)
        self.layers[1].input = self.layers[0].output
        self.layers[1].compile(init_params=False)


class StandardDenoisingAutoEncoder(Sequential):
    # an autoencoder with shared weights
    def __init__(self, input_dim, hidden_dim, loss="ce", optimizer="sgd"):

        Sequential.__init__(self, prediction_type="matrix")
        rng = np.random.RandomState(123)
        w_lamda = 4 * np.sqrt(6. / (input_dim + hidden_dim))
        l0 = NoisyInput(input_dim=input_dim, corruption_level=0.)
        l1 = Dense(input_dim=input_dim, output_dim=hidden_dim, activation="sigmoid", rng=rng, w_lamda=w_lamda)
        l2 = Dense(input_dim=hidden_dim, output_dim=input_dim, activation="sigmoid" )
        self.add_layer(l0)
        self.add_layer(l1)
        self.add_layer(l2)
        self.compile(loss=loss, optimizer=optimizer)

    def compile_layers(self):
        print("Compiling Layer ",0)
        self.layers[0].input = self.input
        self.layers[0].compile()
        print("Compiling Layer ",1)
        self.layers[1].input = self.layers[0].output
        self.layers[1].compile()
        print("Compiling Layer ",2)
        self.layers[2].init_params(initial_w=self.layers[1].W.T)
        self.layers[2].input = self.layers[1].output
        self.layers[2].compile(init_params=False)

if __name__=="__main__":
    from test.Data import load_Mnist
    train,valid,test=load_Mnist()
    ae = StandardDenoisingAutoEncoder(28*28, hidden_dim=500)
    ae.option[Option.BATCH_SIZE]=20
    ae.option[Option.LRATE] = 0.1
    ae.fit_normal(train[0].astype('float64'), train[0].astype('float64'), valid[0].astype('float64'), valid[0].astype('float64'))
