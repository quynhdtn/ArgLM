import math

from liir.ml.core.options.Option import Option
from liir.ml.core.layers import Dense
from liir.ml.core.layers.Input import Input
from liir.ml.core.layers.Model import Sequential


__author__ = 'quynhdo'


class MLP(Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim=None, loss="nl", optimizer="ada"):
        if hidden_dim is None:
            hidden_dim = [int( math.sqrt(input_dim * (output_dim)) + 2 * math.sqrt(input_dim / (output_dim+2)))]

        Sequential.__init__(self)
        self.add_layer(Input(input_dim=input_dim))
        for nd in hidden_dim:
            self.add_layer(Dense(input_dim=self.layers[-1].option[Option.OUTPUT_DIM], output_dim=nd, activation="tanh"))
        self.add_layer(Dense(input_dim=self.layers[-1].option[Option.OUTPUT_DIM], output_dim=output_dim, activation="softmax"))
        self.compile(loss=loss, optimizer=optimizer)

if __name__ == "__main__":
    from test.Data import load_Iris
    # test a basic MLP
    import numpy as np
    from sklearn.metrics import f1_score

    iris = load_Iris()
    X,Y,X_test,Y_test = iris
    input_dim = X.shape[1]
    output_dim = len (list( np.unique(Y)))
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=[10,20])
    model.option[Option.MAX_EPOCHS]=50
    #model.option[Option.BATCH_SIZE] = 15
    model.option[Option.LRATE]=0.1

    def process (X, Y):
        return np.asarray(X), None, np.asarray(Y), None
    model.fit_shuffer(X,Y, process_data_func=process)
    #model.fit_normal(X,Y)
    y_pred = model.predict(X_test)
    print (f1_score(Y_test,y_pred))

    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X, Y)
    y_pred=clf.predict(X_test)
    print (f1_score(Y_test,y_pred))
