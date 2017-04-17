import pickle
import sys
import time

from theano import tensor
import theano as th
import numpy as np

from liir.ml.core.options import Option
from liir.ml.core.layers.Dense import Dense
from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.Model import Sequential
from liir.ml.core.layers.Repeat import Repeat
from liir.ml.core.layers.Reverse import Reverse
from liir.ml.core.layers.TimeDistributed import TimeDitributed
from optimizer.Optimizer2 import getOptimizer
from utils.Data import get_minibatches_idx
from utils.Functions import getFunction

__author__ = 'quynhdo'

# sequence to sequence model
# first encode the full input chain, then decode into another chain

class Seq2Seq(Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim, max_len_out=None, loss="nl", optimizer="ada"):
        Sequential.__init__(self, use_mask=True, input_value_type="int32", prediction_type="vector", prediction_value_type='int32')
        self.option[Option.MAX_LEN_OUT] = max_len_out
        self.option[Option.INPUT_DIM] = input_dim
        self.option[Option.OUTPUT_DIM] = output_dim
        self.option[Option.IS_SEQUENCE_WORK] = True

        self.len_out = tensor.iscalar('lo')
        l0 = Reverse()
        l1 = Embedding(input_dim, hidden_dim)
        l2 = LSTM(hidden_dim, hidden_dim)
        l3 = Repeat(times=self.len_out)
        l4 = LSTM(hidden_dim, hidden_dim, return_sequences=True, need_swap_axis=True) # need swap axis here
        l5 = TimeDitributed(core_layer=Dense(hidden_dim, output_dim, activation="softmax"))

        self.add_layer([l0, l1,l2,l3,l4, l5])
        self.compile(loss=loss, optimizer=optimizer)
    def compile(self, loss="mse", optimizer="sgd", compile_layer=True):

        # set sequential ids to all layers in the model
        for i in range(len(self.layers)):
            self.layers[i].id = str(i)

        if self.option[Option.SAVE_TOPOLOGY] != "":

            with open(self.option[Option.SAVE_TOPOLOGY], 'wb') as f:
                            pickle.dump(self, f)

        if self.use_mask:
            self.input_mask = tensor.imatrix('i_mask')
            if self.prediction_type == "vector":
                self.output_mask = tensor.ivector('o_mask')
            else:
                self.output_mask = tensor.imatrix('o_mask')

        # start passing the data
        self.layers[0].input = self.input
        if self.use_mask:
            for l in self.layers:
                l.mask = self.input_mask

        if self.use_noise is not None:
            for l in self.layers:
                if isinstance(l, Dropout):
                    l.use_noise = self.use_noise

        if compile_layer:
            self.compile_layers()

        for l in self.layers:
            for kk, pp in l.params.items():
                    self.params[kk] = pp

        self.output = self.layers[-1].output


        if self.option[Option.IS_SEQUENCE_WORK]:
            self.output = tensor.reshape(self.output,(-1, self.output.shape[-1]))

        if self.prediction_type == "vector":
            self.prediction = tensor.argmax(self.output, axis=-1)
        else:
            self.prediction = self.output

        if self.use_mask:
            self.cost = getFunction(loss)(self.output, self.gold, mask=self.output_mask)
            self.f_cost = th.function([self.input, self.gold, self.input_mask, self.output_mask, self.len_out], self.cost, name='f_cost')
            #self.f_cost = th.function([self.input, self.gold, self.input_mask], self.cost, name='f_cost')
            grads = tensor.grad(self.cost, wrt=list(self.params.values()))
            self.f_grad = th.function([self.input, self.gold, self.input_mask, self.output_mask, self.len_out], grads, name='f_grad')
            self.f_grad_shared, self.f_update = getOptimizer(optimizer)(self.lr, self.params, grads,
                                              [self.input, self.gold,  self.input_mask, self.output_mask, self.len_out], self.cost)



    def fit_shuffer(self, X, Y,  X_valid=None, Y_valid=None, process_data_func=None):
        # in this fit function, data will be shuffered
        history_errs = []
        best_p = None
        bad_count = 0
        uidx = 0  # the number of update done
        estop = False  # early stop

        start_time = time.time()

        try:
            for eidx in range(self.option[Option.MAX_EPOCHS]):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(X), self.option[Option.BATCH_SIZE], shuffle=True)
                for _, train_index in kf:
                    if self.use_noise is not None:
                        self.use_noise.set_value(1.)
                    uidx += 1


                    # Select the random examples for this minibatch
                    y = [Y[t] for t in train_index]
                    x = [X[t]for t in train_index]


                    x, mask_x, y, mask_y = process_data_func(x, y)

                    len_out = y.shape[1]

                    x, y,  mask_x,mask_y, _, _,_, _ = self.standardize_data(x, y, mask_x, mask_y, None,None, None,None)


                    '''
                    print(x.shape)
                    m  = self.get_output_layer(1,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(2,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(3,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(4,x, mask_x)
                    print (m.shape)
                    '''
                    if self.use_mask:
                        cost = self.f_grad_shared(x,  y, mask_x, mask_y, len_out)
                    else:
                        cost = self.f_grad_shared(x,  y)

                    self.f_update(self.option[Option.LRATE])
                    n_samples += x.shape[1]
                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.


                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)
                    if X_valid is not None:

                        if uidx % self.option[Option.VALID_FREQ]== 0:
                            if self.use_noise is not None:
                                self.use_noise.set_value(0.)
                            train_err = self.evaluation(x,y,mask_x,mask_y)
                            x_v, mask_x_v, y_v, mask_y_v = process_data_func(X_valid, Y_valid)
                            len_out = y_v.shape[0]
                            x_v, y_v,  mask_x_v,mask_y_v, _, _,_, _ = self.standardize_data(x_v, y_v, mask_x_v, mask_y_v, None,None, None,None)

                            valid_err = self.evaluation(x_v,y_v,mask_x_v,mask_y_v)


                            history_errs.append([valid_err])

                            if (best_p is None or
                                valid_err <= np.array(history_errs)[:,
                                                                       0].min()):

                                #with open(self.option[Option.SAVE_TO], 'wb') as f:
                                #        pickle.dump(self, f)
                                self.save_params(self.option[Option.SAVE_TO])
                                bad_counter = 0

                            print( ('Train ', train_err, 'Valid ', valid_err,
                                   ) )

                            if (len(history_errs) > self.option[Option.PATIENCE] and
                                valid_err >= np.array(history_errs)[:-self.option[Option.PATIENCE],
                                                                       0].min()):
                                bad_counter += 1
                                if bad_counter > self.option[Option.PATIENCE]:
                                    print('Early Stop!')
                                    estop = True
                                    break


                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()



        #self.use_noise.set_value(0.)
        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
                (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    from test.Data import loadPOSdata, preprare_seq_seq_data


    X,Y, current_x, current_y = (loadPOSdata())

    mdl = Seq2Seq(current_x, current_y, 32)

    func = preprare_seq_seq_data
    mdl.fit_shuffer(X,Y, process_data_func=func)

    #x, mask_x,y, mask_y = preprare_seq_seq_data(X,Y)
    #mdl.fit_normal(x,y, X_mask=mask_x, Y_mask=mask_y)
    #print (mdl.evaluation(x, y, mask_x, mask_y))
