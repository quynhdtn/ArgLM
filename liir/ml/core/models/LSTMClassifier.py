from liir.ml.core.layers.TimeDistributed import TimeDitributed

__author__ = 'quynhdo'
from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Dense import Dense
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.Model import Sequential


__author__ = 'quynhdo'

class LSTMClassifier(Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim, dep=2, loss="nl", optimizer="ada", we_dict=None, map=None):
        Sequential.__init__(self, use_mask=True, input_value_type="int32", prediction_type="vector",
                            prediction_value_type='int32')

        self.option[Option.LOSS] = loss
        self.option[Option.OPTIMIZER] = optimizer
        l1 = Embedding(input_dim, hidden_dim, we_dict=we_dict, map=map)
        self.add_layer(l1)
        for i in range(dep):
            l2 = LSTM(hidden_dim, hidden_dim, return_sequences=True)
            self.add_layer(l2)

        l4 = TimeDitributed(core_layer=Dense(hidden_dim, output_dim, activation="softmax"))

        self.add_layer(l4)
        self.option[Option.IS_SEQUENCE_WORK] = True
        self.compile()

if __name__ == "__main__":
    from test.Data import loadPOSdata, preprare_seq_seq_data

    X,Y, current_x, current_y = (loadPOSdata())

    mdl = LSTMClassifier(current_x, current_y, 32)

    func = preprare_seq_seq_data
    mdl.fit_shuffer(X,Y, process_data_func=func)

    #x, mask_x,y, mask_y = preprare_seq_seq_data(X,Y)
    #mdl.fit_normal(x,y, X_mask=mask_x, Y_mask=mask_y)
    #print (mdl.evaluation(x, y, mask_x, mask_y))


