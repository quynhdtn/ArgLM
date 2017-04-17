from theano.tensor.shared_randomstreams import RandomStreams

from liir.ml.core.options.Option import Option
from liir.ml.core.layers import Dense
from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.MeanPooling import MeanPooling
from liir.ml.core.layers.Model import Sequential


__author__ = 'quynhdo'

class LSTMTextClassifier(Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim, loss="nl", optimizer="ada"):
        Sequential.__init__(self, use_mask=True, input_value_type="int64", hidden_dim = hidden_dim, loss=loss, optimizer=optimizer)
        l1 = Embedding(input_dim, hidden_dim)
        l2 = LSTM(hidden_dim, hidden_dim, return_sequences=True)
        l3 = MeanPooling()
        l4 = Dropout(hidden_dim,theano_rng= RandomStreams(128))
        l5 = Dense(hidden_dim, output_dim, activation="softmax")
        self.add_layer([l1,l2,l3,l4,l5])
        self.compile()

if __name__ == "__main__":
    from test.Data import load_Imdb, prepare_data_imdb

    train, valid, test = load_Imdb()
    print (len(train[0]))
    print(len(valid[0]))
    print (len(test[0][0:1000]))

    mdl = LSTMTextClassifier(10000, 2, 128)
    mdl.option[Option.MAX_EPOCHS]=20
    mdl.option[Option.BATCH_SIZE]=16
    mdl.option[Option.LRATE]=0.0001
    func = prepare_data_imdb

    mdl.fit_shuffer(train[0], train[1], X_valid=valid[0],Y_valid=valid[1], process_data_func=func)
    x, mask, y, mask_y = func(test[0][0:1000], test[1][0:1000])
    print(mdl.evaluation(x, y, mask, mask_y))
    '''
    mdl = load_model(topo="topology.pkl", params="params.pkl")
    func = prepare_data_imdb
    x, mask, y, mask_y = func(test[0][0:1000], test[1][0:1000])
    print(mdl.evaluation(x, y, mask, mask_y))
    '''
