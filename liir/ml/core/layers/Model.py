from collections import OrderedDict
import os
import timeit
import pickle
import sys
import time

from theano import config,tensor
import theano as th
import numpy as np

from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.options.Option import Option
from optimizer.Optimizer import getOptimizer
from utils.Data import get_minibatches_idx, numpy_floatX
from utils.Functions import getFunction


__author__ = 'quynhdo'

class Model(object):
    def __init__(self, idx="0", input_value_type=config.floatX, prediction_type='vector',
                 prediction_value_type=config.floatX, use_mask=False, use_noise=False):
        '''

        :param idx: id of the
        :param input_value_type:
        :param prediction_type:
        :param prediction_value_type:
        :param use_mask:
        :return:
        '''
        self.input = tensor.matrix('x', dtype=input_value_type)
        self.output = None
        self.prediction_type = prediction_type
        if prediction_type == 'vector':
            self.prediction = tensor.ivector('y')
            self.gold = tensor.ivector('g')
        if prediction_type == 'matrix':
            self.prediction = tensor.matrix('y', dtype=prediction_value_type)
            self.gold = tensor.matrix('g', dtype=prediction_value_type)
        self.params = OrderedDict()
        self.layers = []
        self.optimizer = None
        self.cost = None
        self.id = idx
        self.option = Option()  # options for the model
        if use_noise:
            self.use_noise= th.shared(numpy_floatX(0.))
        else:

            self.use_noise = None
        self.use_mask = use_mask # can be set to True when working with sequences of different sizes

        self.input_mask = None
        self.output_mask = None

        self.f_cost = None
        self.f_grad = None
        self.lr = tensor.scalar(name='lr')
        self.f_grad_shared = None
        self.f_update = None


    def add_layer(self, l):
        if isinstance(l, list):
            for ll in l:
                self.add_layer(ll)
        else:
            self.layers.append(l)
            if isinstance(l, Dropout):
                self.use_noise = th.shared(numpy_floatX(0.))

    def get_layer(self, idx):
        if idx < len(self.layers):
            return self.layers[idx]
        return None

    def compile(self):
        pass


    def standardize_data(self, X, Y=None, X_mask=None, Y_mask = None,  X_valid=None, Y_valid=None, Xvalid_mask=None, Yvalid_mask=None):
        X1 = X
        Y1 = None
        X_valid1 = None
        Y_valid1 = None
        Xvalid_mask1 = None
        Yvalid_mask1 = None
        X_mask1 = None
        Y_mask1 = None



        if self.option[Option.IS_SEQUENCE_WORK]:
            X1 = X.swapaxes(0,1)
            if Y is not None:
                Y1 = Y.swapaxes(0,1)
                Y1 = Y1.flatten()

            if X_valid is not None:
                X_valid1 = X_valid.swapaxes(0,1)
            if Y_valid is not None:
                Y_valid1 = Y_valid.swapaxes(0,1)
                Y_valid1 = Y_valid1.flatten()

            if Xvalid_mask is not None:
                Xvalid_mask1 = Xvalid_mask.swapaxes(0,1)
            if Yvalid_mask is not None:
                Yvalid_mask1 = Yvalid_mask.swapaxes(0,1)
                Yvalid_mask1 = Yvalid_mask1 .flatten()

            if X_mask is not None:
                X_mask1 = X_mask.swapaxes(0,1)

            if Y_mask is not None:
                Y_mask1 = Y_mask.swapaxes(0,1)
                Y_mask1 = Y_mask1.flatten()

        #X = th.shared(X, )
        return X1,Y1, X_mask1,Y_mask1,  X_valid1,Y_valid1,Xvalid_mask1,Yvalid_mask1


    def predict(self, x, mask = None):
        if self.use_noise is not None:
            self.use_noise.set_value(0.)
        if mask is None:
            test_model = th.function(
                    inputs=[],
                    outputs=self.prediction,
                    givens={
                        self.input: x ,

                    }
                )

            return test_model()
        else:
            test_model = th.function(
                    inputs=[],
                    outputs=self.prediction,
                    givens={
                        self.input: x,
                        self.input_mask: mask

                    }
                )

            return test_model()

    def errors_mask(self, y, output_mask):
        if y.ndim != self.prediction.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.prediction.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return tensor.mean(tensor.neq(self.prediction, y) * output_mask)
        else:
            raise NotImplementedError()



    def get_output_layer(self, layer_id, x, mask=None):
        if mask is None:
            l = self.layers[layer_id]
            output_func =  th.function([], l.output, givens={
                self.input: x
            })
            return output_func()
        else:
            l = self.layers[layer_id]

            output_func =  th.function([], l.output, givens={
                self.input: x,
                self.input_mask : mask


            },on_unused_input='warn')
            return output_func()

    def evaluation(self,x, y,  input_mask = None, output_mask=None):
        #x,y,input_mask,output_mask,_,_,_,_ = self.standardize_data(x,y,input_mask,output_mask, None, None,None,None)

        if self.use_noise is not None:
            self.use_noise.set_value(0.)
        if not self.use_mask:
            error = tensor.mean(tensor.neq(self.prediction, self.gold))
            test_model = th.function(
                    inputs=[self.input, self.gold],
                    outputs=error,
                )
            return test_model(x, y)
        else:
            error_mask =  (tensor.neq(self.prediction, self.gold) * self.output_mask).sum() / self.output_mask.sum()
            test_model = th.function(
                    inputs=[self.input, self.gold, self.input_mask, self.output_mask],
                    outputs=error_mask,
                )
            return test_model(x, y, input_mask, output_mask)




    def fit_normal(self, X, Y, X_valid=None, Y_valid=None, X_mask=None, Y_mask=None, Xvalid_mask=None, Yvalid_mask=None):

        # normal fitting
        # don't shuffer data,
        # just run over all the data of the training

        X,Y, X_mask,Y_mask, X_valid,Y_valid,Xvalid_mask,Yvalid_mask = self.standardize_data(X,Y,X_mask,Y_mask,X_valid,Y_valid,Xvalid_mask,Yvalid_mask)

        if self.option[Option.BATCH_SIZE] > X.shape[0]:
            self.option[Option.BATCH_SIZE] = X.shape[0]

        n_train_batches = (int) (X.shape[0] / self.option[Option.BATCH_SIZE])
        n_valid_batches = -1
        if X_valid is not None:
            n_valid_batches = (int)(X_valid.shape[0] / self.option[Option.BATCH_SIZE])
        start_time = timeit.default_timer()
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        done_looping = False
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        patience = self.option[Option.PATIENCE]
        validation_frequency = min(n_train_batches, patience // 2)

        epoch = 0
        while (epoch < self.option[Option.MAX_EPOCHS]) and (not done_looping):
            epoch += 1
            c = []

            for minibatch_index in range(int(n_train_batches)):
                trainX = X[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                trainY = Y[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                if X_mask is not None and Y_mask is not None:
                    maskX = X_mask[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                    maskY = Y_mask[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                    minibatch_avg_cost = self.f_grad_shared(trainX, trainY, maskX, maskY)
                if X_mask is not None and Y_mask is None:
                    maskX = X_mask[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                    minibatch_avg_cost = self.f_grad_shared(trainX, trainY, maskX, maskX)
                if X_mask is None and Y_mask is None:
                    minibatch_avg_cost = self.f_grad_shared(trainX, trainY)
                c.append(minibatch_avg_cost)
                self.f_update(self.option[Option.LRATE])
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if X_valid is not None:
                    if (iter + 1) % validation_frequency == 0:
                        validation_losses = []
                        for j in  range(int(n_valid_batches)):
                            validX = X_valid[j * self.option[Option.BATCH_SIZE]: (j + 1) * self.option[Option.BATCH_SIZE]]
                            validY = Y_valid[j * self.option[Option.BATCH_SIZE]: (j + 1) * self.option[Option.BATCH_SIZE]]
                            if Xvalid_mask is not None and Yvalid_mask is not None:
                                maskXv = Xvalid_mask[minibatch_index * self.option[Option.VALID_BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.VALID_BATCH_SIZE]]
                                maskYv = Yvalid_mask[minibatch_index * self.option[Option.VALID_BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.VALID_BATCH_SIZE]]
                                minibatch_avg_cost = self.f_grad_shared(validX, validY, maskXv, maskYv)
                            if Xvalid_mask is not None and Yvalid_mask is None:
                                maskXv = Xvalid_mask[minibatch_index * self.option[Option.BATCH_SIZE]: (minibatch_index + 1) * self.option[Option.BATCH_SIZE]]
                                minibatch_avg_cost = self.f_grad_shared(validX, validY, maskXv, maskXv)

                            if Xvalid_mask is None and Yvalid_mask is None:
                                minibatch_avg_cost = self.f_grad_shared(validX, validY)

                            validation_losses.append(self.f_grad_shared(validX,validY))
                        this_validation_loss = np.mean(np.asarray(validation_losses))

                        print(
                            'epoch %i, minibatch %i/%i, validation error %f %%' %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                this_validation_loss
                            )
                        )

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            #improve patience if loss improvement is good enough
                            if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                            ):
                                patience = max(patience, iter * patience_increase)

                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            with open(self.option[Option.SAVE_TO], 'wb') as f:
                                    pickle.dump(self, f)

                if patience <= iter:
                    done_looping = True
                    break
            print ('Training epoch %d, cost ' % epoch, np.mean(c))
        end_time = timeit.default_timer()
        if X_valid is not None:
            print(('Optimization complete. Best validation score of %f %% '
                   'obtained at iteration %i, with test performance %f %%') %
                  (best_validation_loss , best_iter + 1, test_score ))
        print ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def save_params(self, output):
        with open(output, 'wb') as f:
                    pickle.dump(self.params, f)

    def fit_shuffer(self, X, Y,  X_valid=None, Y_valid=None, process_data_func=None, continue_epoch=None, load_current_path=None, load_current_best=None):
        # in this fit function, data will be shuffered
        history_errs = []
        best_p = None
        bad_count = 0
        uidx = 0  # the number of update done
        estop = False  # early stop

        start_time = time.time()

        if load_current_best is not None:
            self.load_params(load_current_best)
            if self.use_noise is not None:
                self.use_noise.set_value(0.)
            # train_err = self.evaluation(x,y,mask_x,mask_y)
            train_err = None
            x_v, mask_x_v, y_v, mask_y_v = process_data_func(X_valid, Y_valid)

            x_v, y_v, mask_x_v, mask_y_v, _, _, _, _ = self.standardize_data(x_v, y_v, mask_x_v, mask_y_v, None, None,
                                                                             None, None)

            valid_err = self.evaluation(x_v, y_v, mask_x_v, mask_y_v)

            history_errs.append([valid_err])
            best_p = np.array(history_errs)[:,0].min()
            print (best_p)


        try:
            if load_current_path is not None:
                self.load_params(load_current_path)
            for eidx in range(self.option[Option.MAX_EPOCHS]):
                if continue_epoch is not None:
                    if eidx < continue_epoch:
                        continue
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

                    x, y,  mask_x,mask_y, _, _,_, _ = self.standardize_data(x, y, mask_x, mask_y, None,None, None,None)

                    '''
                    print(x.shape)
                    m  = self.get_output_layer(1,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(2,x, mask_x)
                    print (m.shape)
                    '''
                    if self.use_mask:
                        cost = self.f_grad_shared(x,  y, mask_x, mask_y)
                    else:
                        cost = self.f_grad_shared(x,  y)

                    self.f_update(self.option[Option.LRATE])
                    n_samples += x.shape[1]
                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.


                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                    if uidx % self.option[Option.SAVE_FREQ]== 0:
                        self.save_params(self.option[Option.SAVE_TO] + "."+str(eidx))


                    if X_valid is not None:

                        if uidx % self.option[Option.VALID_FREQ]== 0:
                            if self.use_noise is not None:
                                self.use_noise.set_value(0.)
                            #train_err = self.evaluation(x,y,mask_x,mask_y)
                            train_err = None
                            x_v, mask_x_v, y_v, mask_y_v = process_data_func(X_valid, Y_valid)

                            x_v, y_v,  mask_x_v,mask_y_v, _, _,_, _ = self.standardize_data(x_v, y_v, mask_x_v, mask_y_v, None,None, None,None)


                            valid_err = self.evaluation(x_v,y_v,mask_x_v,mask_y_v)


                            history_errs.append([valid_err])

                            if (best_p is None or
                                valid_err <= np.array(history_errs)[:,0].min()):

                                #with open(self.option[Option.SAVE_TO], 'wb') as f:
                                #        pickle.dump(self, f)
                                self.save_params(self.option[Option.SAVE_BEST_VALID_TO])
                                print ("saving best params")
                                bad_counter = 0
                                best_p = valid_err

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




class Sequential(Model):
    def __init__(self, idx="0", input_value_type=config.floatX, prediction_type='vector',
                 prediction_value_type=config.floatX, use_mask=False, use_noise=False):
        Model.__init__(self, idx, input_value_type, prediction_type, prediction_value_type, use_mask=use_mask, use_noise=use_noise)

    def compile_layers(self):
        for i in range(len(self.layers)):
            if i>0:
                self.layers[i].input = self.layers[i-1].output
            print("Compiling Layer ",i)
            self.layers[i].compile()

    def compile(self,  compile_layer=True):
        # set sequential ids to all layers in the model
        for i in range(len(self.layers)):
            self.layers[i].id = str(i)



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
            self.cost = getFunction(self.option[Option.LOSS])(self.output, self.gold, mask=self.output_mask)
            self.f_cost = th.function([self.input, self.gold, self.input_mask, self.output_mask], self.cost, name='f_cost')
            #self.f_cost = th.function([self.input, self.gold, self.input_mask], self.cost, name='f_cost')
            grads = tensor.grad(self.cost, wrt=list(self.params.values()))
            self.f_grad = th.function([self.input, self.gold, self.input_mask, self.output_mask], grads, name='f_grad')
            self.f_grad_shared, self.f_update = getOptimizer(self.option[Option.OPTIMIZER])(self.lr, self.params, grads,
                                                self.input, self.gold, self.cost, self.input_mask, self.output_mask)
        else:

            self.cost = getFunction(self.option[Option.LOSS])(self.output, self.gold)
            self.f_cost = th.function([self.input, self.gold], self.cost, name='f_cost')
            grads = tensor.grad(self.cost, wrt=list(self.params.values()))
            self.f_grad = th.function([self.input, self.gold], grads, name='f_grad')
            self.f_grad_shared, self.f_update = getOptimizer(self.option[Option.OPTIMIZER])(self.lr, self.params, grads,
                                                self.input, self.gold, self.cost)




    def load_params(self, params):
        params = pickle.load(open(params, "rb"))

        for l in self.layers:
            for kk, pp in l.params.items():
                    pp.set_value(params[kk].get_value())
                    self.params[kk] = pp



def load_model(topo, params):
    mdl = pickle.load(open(topo, "rb"))
    mdl.compile()
    mdl.load_params(params)
    return mdl
