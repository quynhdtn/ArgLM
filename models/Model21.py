import pickle
from multiprocessing.dummy import Pool

from data.DataManager2 import generate_sequential_data21, preprare_seq_seq_data, generate_vob21
from exp.ConfigReader import read_config
from liir.ml.core.layers.Dense import Dense
from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.Merge import Merge
from liir.ml.core.layers.Model import Sequential
from liir.ml.core.layers.TimeDistributed import TimeDitributed
from liir.ml.core.options.Option import Option
from optimizer.Optimizer2 import getOptimizer
import sys
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
import time

from readers.BatchReader import Conll2009BatchReader
from utils.Data import get_minibatches_idx
from utils.Functions import getFunction
import theano as th
import numpy as np

import theano.tensor as T

from we.WEDict import WEDict

__author__ = 'quynhdo'
class Model21(Sequential):

#### for this one, we produce Word_LABEL, so only one output
    def __init__(self, input_dim1, input_dim2, output_dim, hidden_dim1, hidden_dim2, dep=2, loss="nl", optimizer="ada", we_dict1=None, we_dict2=None, map1=None, map2=None, use_noise=True):
        Sequential.__init__(self, use_mask=True, input_value_type="int32", prediction_type="vector",
                            prediction_value_type='int32', use_noise=use_noise)

        self.extra_input = tensor.matrix('x_extra', dtype="int32")
        self.extra_output = None


        self.option[Option.LOSS] = loss
        self.option[Option.OPTIMIZER] = optimizer
        l1 = Embedding(input_dim1, hidden_dim1, we_dict=we_dict1, map=map1)

        l2 = Embedding(input_dim2, hidden_dim2, we_dict=we_dict2, map=map2)


        l3 = Merge(layers=[l1,l2])

        self.add_layer([l1,l2,l3])

        l4 = Dropout(hidden_dim1+hidden_dim2,theano_rng= RandomStreams(128))
        self.add_layer(l4)

        for i in range(dep):
            l5 = LSTM(hidden_dim1+hidden_dim2, hidden_dim1+hidden_dim2, return_sequences=True)
            self.add_layer(l5)

        l6 = Dropout(hidden_dim1+hidden_dim2,theano_rng= RandomStreams(128))
        self.add_layer(l6)

        l7 = TimeDitributed(core_layer=Dense(hidden_dim1+hidden_dim2, output_dim, activation="softmax"))


        self.add_layer([l7])

        self.option[Option.IS_SEQUENCE_WORK] = True

    def compile_layers(self):
        self.layers[0].compile()
        self.layers[1].compile()
        self.layers[2].compile()

        for i in range(3, len(self.layers)):
            self.layers[i].input = self.layers[i-1].output
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
        self.layers[1].input = self.extra_input

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


        if self.use_mask:
            self.cost = getFunction(self.option[Option.LOSS])(self.output, self.gold, mask=self.output_mask)
            self.f_cost = th.function([self.input, self.extra_input, self.gold, self.input_mask, self.output_mask], self.cost, name='f_cost')
            #self.f_cost = th.function([self.input, self.gold, self.input_mask], self.cost, name='f_cost')
            grads = tensor.grad(self.cost, wrt=list(self.params.values()))
            self.f_grad = th.function([self.input, self.extra_input, self.gold,  self.input_mask, self.output_mask], grads, name='f_grad')
            self.f_grad_shared, self.f_update = getOptimizer(self.option[Option.OPTIMIZER])(self.lr, self.params, grads,



                                              [self.input, self.extra_input, self.gold,  self.input_mask, self.output_mask], self.cost)


    def evaluation(self,x1,x2, y,  input_mask = None, output_mask=None):
        #x,y,input_mask,output_mask,_,_,_,_ = self.standardize_data(x,y,input_mask,output_mask, None, None,None,None)

        if self.use_noise is not None:
            self.use_noise.set_value(0.)

        error_mask =  (tensor.neq(self.prediction, self.gold) * self.output_mask).sum() / self.output_mask.sum()




        test_model = th.function(
                    inputs=[self.input,self.extra_input, self.gold, self.input_mask, self.output_mask],
                    outputs=error_mask ,
                )
        return test_model(x1,x2, y, input_mask, output_mask)

    def get_output_layer(self, layer_id, x1, x2, input_mask=None):
        if input_mask is None:
            l = self.layers[layer_id]
            output_func =  th.function([], l.output, givens={
                self.input: x1,
                self.extra_input:x2,


            })
            return output_func()
        else:
            l = self.layers[layer_id]
            print (self.input)
            print (self.extra_input)
            print (self.input_mask)
            output_func =  th.function([], l.output, givens={
                self.input: x1,
                self.extra_input:x2,
                self.input_mask:input_mask,




            },on_unused_input='warn')
            return output_func()


    def get_output(self,  x1, x2, input_mask=None):
        if input_mask is None:

            output_func =  th.function([], self.output, givens={
                self.input: x1,
                self.extra_input:x2,


            })
            return output_func()
        else:

            print (self.input)
            print (self.extra_input)
            print (self.input_mask)
            output_func =  th.function([], self.output, givens={
                self.input: x1,
                self.extra_input:x2,
                self.input_mask:input_mask,




            },on_unused_input='warn')
            return

    def get_extra_output(self,  x1, x2, input_mask=None):
        if input_mask is None:

            output_func =  th.function([], self.extra_output, givens={
                self.input: x1,
                self.extra_input:x2,


            })
            return output_func()
        else:

            print (self.input)
            print (self.extra_input)
            print (self.input_mask)
            output_func =  th.function([], self.extra_output, givens={
                self.input: x1,
                self.extra_input:x2,
                self.input_mask:input_mask,




            },on_unused_input='warn')
            return output_func

    def fit_shuffer(self, X1,X2, Y,  X_valid1=None,  X_valid2=None,Y_valid=None ,process_data_func=None, continue_epoch = None, load_path=None, load_current_best = None):
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
            # train_err = self.evaluation(x1,x2,y, mask_x,mask_y)

            nb_valid_batchs = len(X_valid1) // self.option[Option.BATCH_SIZE]

            start = 0
            end = 0
            valid_rs = []

            for g in range(nb_valid_batchs + 1):
                start = g * self.option[Option.BATCH_SIZE]
                end = start + self.option[Option.BATCH_SIZE]
                if end > len(X_valid1):
                    end = len(X_valid1)

                X_valid1t = X_valid1[start:end]
                Y_validt = Y_valid[start:end]
                X_valid2t = X_valid2[start:end]

                x_v1, mask_x_v, y_v, mask_y_v = process_data_func(X_valid1t, Y_validt)
                x_v2, mask_x_v = process_data_func(X_valid2t)

                x_v1, y_v, mask_x_v, mask_y_v, _, _, _, _ = self.standardize_data(x_v1, y_v, mask_x_v, mask_y_v, None,
                                                                                  None, None, None)

                x_v2, _, _, _, _, _, _, _ = self.standardize_data(x_v2, None, None, None, None, None, None, None)

                valid_err = self.evaluation(x_v1, x_v2, y_v, mask_x_v, mask_y_v)
                valid_rs.append(valid_err)
                # print (valid_rs)

            valid_err = np.mean(np.asarray(valid_rs))

            history_errs.append([valid_err])
            best_p = np.array(history_errs)[:,0].min()
            print (best_p)

        try:
            for eidx in range(self.option[Option.MAX_EPOCHS]):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(X1), self.option[Option.BATCH_SIZE], shuffle=True)
                for _, train_index in kf:
                    if self.use_noise is not None:
                        self.use_noise.set_value(1.)
                    uidx += 1


                    # Select the random examples for this minibatch
                    y = [Y[t] for t in train_index]
                    x1 = [X1[t]for t in train_index]
                    x2 = [X2[t]for t in train_index]


                    x1, mask_x, y, mask_y = process_data_func(x1, y)

                    x2, mask_x = process_data_func(x2, None)

                    x1, y,  mask_x,mask_y, _, _,_, _ = self.standardize_data(x1, y, mask_x, mask_y, None,None, None,None)

                    x2, _,  _,_, _, _,_, _ = self.standardize_data(x2, None, None, None, None,None, None,None)

                    '''
                    print(x.shape)
                    m  = self.get_output_layer(1,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(2,x, mask_x)
                    print (m.shape)
                    '''
                    cost = self.f_grad_shared(x1, x2,  y, mask_x, mask_y)


                    self.f_update(self.option[Option.LRATE])
                    n_samples += x1.shape[1]
                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.


                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                    if uidx % self.option[Option.SAVE_FREQ]== 0:
                        self.save_params(self.option[Option.SAVE_TO] + "."+str(eidx))
                        #train_err = self.evaluation(x1,x2,y,mask_x,mask_y)
                        #print ("train error: ", train_err)


                    if X_valid1 is not None:

                        if uidx % self.option[Option.VALID_FREQ]== 0:
                            if self.use_noise is not None:
                                self.use_noise.set_value(0.)
                            #train_err = self.evaluation(x1,x2,y, mask_x,mask_y)

                            nb_valid_batchs =  len(X_valid1) //  self.option[Option.BATCH_SIZE]

                            start = 0
                            end = 0
                            valid_rs = []




                            for g in range(nb_valid_batchs+1):
                                start = g * self.option[Option.BATCH_SIZE]
                                end =  start + self.option[Option.BATCH_SIZE]
                                if end > len(X_valid1):
                                    end = len(X_valid1)

                                X_valid1t = X_valid1[start:end]
                                Y_validt= Y_valid[start:end]
                                X_valid2t = X_valid2[start:end]

                                if len(X_valid1t) == 0:
                                    continue
                                x_v1, mask_x_v, y_v, mask_y_v = process_data_func(X_valid1t, Y_validt)
                                x_v2, mask_x_v = process_data_func(X_valid2t)


                                x_v1, y_v,  mask_x_v,mask_y_v, _, _,_, _ = self.standardize_data(x_v1, y_v, mask_x_v, mask_y_v, None,None, None,None)

                                x_v2, _,  _,_, _, _,_, _ = self.standardize_data(x_v2, None, None, None, None,None, None,None)


                                valid_err = self.evaluation(x_v1,x_v2,y_v,mask_x_v,mask_y_v)
                                valid_rs.append(valid_err)
                                #print (valid_rs)

                            valid_err = np.mean(np.asarray(valid_rs))

                            history_errs.append([valid_err])

                            if (best_p is None or
                                valid_err <= np.array(history_errs)[:,
                                                                       0].min()):

                                #with open(self.option[Option.SAVE_TO], 'wb') as f:
                                #        pickle.dump(self, f)
                                self.save_params(self.option[Option.SAVE_BEST_VALID_TO])
                                bad_counter = 0
                                best_p = valid_err
                                print ("saving best params")

                            print( ('Train ', 'Valid ', valid_err,
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


def trainSemLM21(train_texts, valid_texts, we_dict_path1=None, we_dict_path2=None, dep=1,hidden_size1=32, hidden_size2=32, batch_size=200, save_folder=".", model_name="adv", max_epochs=120, load_dt=True,
                 pos = "V",data="form"):
    X1,X2, Y, Xv1,Xv2, Yv, map_x1, map_x2, map_y1= generate_sequential_data21(train_texts, valid_texts, pos=pos,data=data)

    we_dict1 = None
    we_dict2 = None
    if we_dict_path1 is not None:
        we_dict1 = WEDict(we_dict_path1)

    if we_dict_path2 is not None:
        we_dict2 = WEDict(we_dict_path2)



    mdl =Model21(map_x1.current_index_input + 1, map_x2.current_index_input+ 1,
                 map_y1.current_index_input + 1,
                  hidden_size1, hidden_size2, dep=dep, we_dict1=we_dict1,we_dict2=we_dict2,map1=map_x1.input_key_map, map2=map_x2.input_key_map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 200
    mdl.option[Option.VALID_FREQ] = 400
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS]=max_epochs
    mdl.compile()
    func = preprare_seq_seq_data
    mdl.fit_shuffer(X1,X2, Y, Xv1, Xv2, Yv, process_data_func=func)


def trainSemLM21Batch(train_texts, valid_texts, we_dict_path1=None, we_dict_path2=None, dep=1,hidden_size1=32, hidden_size2=32, batch_size=200, save_folder=".", model_name="adv", max_epochs=120, load_dt=True,
                 pos = "V",data="form", load_dict=False):
    if not load_dict:
        print("generating vocabulary")
        map_x1, map_x2, map_y1 = generate_vob21(train_texts, valid_texts, pos=pos,data=data)
        import pickle
        with open(save_folder + "/" + "vob.mdl", 'wb') as voutput:
            pickle.dump((map_x1, map_x2, map_y1), voutput, pickle.HIGHEST_PROTOCOL)
    else:
        import pickle
        map_x1, map_x2, map_y1 = pickle.load(open(save_folder + "/" + "vob.mdl", 'rb'))


    we_dict1 = None
    we_dict2 = None
    if we_dict_path1 is not None:
        we_dict1 = WEDict(we_dict_path1)

    if we_dict_path2 is not None:
        we_dict2 = WEDict(we_dict_path2)

    mdl = Model21(map_x1.current_index_input + 1, map_x2.current_index_input + 1,
                  map_y1.current_index_input + 1,
                  hidden_size1, hidden_size2, dep=dep, we_dict1=we_dict1, we_dict2=we_dict2, map1=map_x1.input_key_map,
                  map2=map_x2.input_key_map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 100
    mdl.option[Option.VALID_FREQ] = 100
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS] = 1
    print ("compiling model...")


    mdl.compile()


    for i in range(max_epochs):
        print (" Epoch ", i )
        reader = Conll2009BatchReader(50000, train_texts)
        while True:
            txt = reader.next()
            if len(txt) == 0:
                mdl.save_params(mdl.option[Option.SAVE_TO] + "_FINAL_" + str(i) )
                break

            X1,X2, Y, Xv1,Xv2, Yv, map_x1, map_x2, map_y1= generate_sequential_data21(txt, valid_texts, pos,data, map_x1, map_x2, map_y1)


            func = preprare_seq_seq_data
            mdl.fit_shuffer(X1,X2, Y, Xv1, Xv2, Yv, process_data_func=func)









def loadSemLM21(load_path,train_texts, valid_texts, we_dict_path1=None, we_dict_path2=None, dep=1,hidden_size1=32, hidden_size2=32, batch_size=200, save_folder=".", model_name="adv", max_epochs=120,
                continue_train=False, pos="V", data="form", load_best=None, continue_epoch=None):
    X1, X2, Y, Xv1, Xv2, Yv, map_x1, map_x2, map_y1 = generate_sequential_data21(train_texts, valid_texts, pos=pos, data=data)




    we_dict1=None
    we_dict2=None
    if we_dict_path1 is not None:
        we_dict1 = WEDict(we_dict_path1)

    if we_dict_path2 is not None:
        we_dict2 = WEDict(we_dict_path2)

    mdl = Model21(map_x1.current_index_input + 1, map_x2.current_index_input + 1,
                  map_y1.current_index_input + 1,
                  hidden_size1, hidden_size2, dep=dep, we_dict1=we_dict1, we_dict2=we_dict2, map1=map_x1.input_key_map,
                  map2=map_x2.input_key_map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 20
    mdl.option[Option.VALID_FREQ] = 100
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS]=max_epochs
    mdl.compile()
    func = preprare_seq_seq_data
    mdl.load_params(load_path)
    if continue_train:
        mdl.fit_shuffer(X1, X2, Y, Xv1, Xv2, Yv, process_data_func=func, load_current_best=load_best, load_path=load_path, continue_epoch=continue_epoch)
    else:
        return mdl, map_x1, map_x2, map_y1

def loadSemLM21Batch(load_path, train_texts, valid_texts, we_dict_path1=None, we_dict_path2=None, dep=1, hidden_size1=32,
                    hidden_size2=32, batch_size=200, save_folder=".", model_name="adv", max_epochs=120,
                    continue_train=False, pos="V", data="form", load_best=None, continue_epoch=None, load_dict=False):


        if not load_dict:
            print("generating vocabulary")
            map_x1, map_x2, map_y1 = generate_vob21(train_texts, valid_texts, pos=pos, data=data)
            import pickle
            with open(save_folder + "/" + "vob.mdl", 'wb') as voutput:
                pickle.dump((map_x1, map_x2, map_y1), voutput, pickle.HIGHEST_PROTOCOL)
        else:
            import pickle
            map_x1, map_x2, map_y1 = pickle.load(open(save_folder + "/" + "vob.mdl", 'rb'))


        we_dict1 = None
        we_dict2 = None
        if we_dict_path1 is not None:
            we_dict1 = WEDict(we_dict_path1)

        if we_dict_path2 is not None:
            we_dict2 = WEDict(we_dict_path2)

        mdl = Model21(map_x1.current_index_input + 1, map_x2.current_index_input + 1,
                      map_y1.current_index_input + 1,
                      hidden_size1, hidden_size2, dep=dep, we_dict1=we_dict1, we_dict2=we_dict2,
                      map1=map_x1.input_key_map,
                      map2=map_x2.input_key_map)

        mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
        mdl.option[Option.SAVE_FREQ] = 20
        mdl.option[Option.VALID_FREQ] = 100
        mdl.option[Option.BATCH_SIZE] = batch_size
        mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
        mdl.option[Option.MAX_EPOCHS] = max_epochs
        mdl.compile()
        func = preprare_seq_seq_data
        mdl.load_params(load_path)
        if continue_train:
            for i in range(max_epochs):
                print(" Epoch ", i)
                reader = Conll2009BatchReader(50000, train_texts)
                while True:
                    txt = reader.next()
                    if len(txt) == 0:
                        mdl.save_params(mdl.option[Option.SAVE_TO] + "_FINAL_" + str(i))
                        break

                    X1, X2, Y, Xv1, Xv2, Yv, map_x1, map_x2, map_y1 = generate_sequential_data21(txt, valid_texts, pos,
                                                                                                 data, map_x1, map_x2,
                                                                                                 map_y1)

                    func = preprare_seq_seq_data
                    mdl.fit_shuffer(X1, X2, Y, Xv1, Xv2, Yv, process_data_func=func)

        else:
            return mdl, map_x1, map_x2, map_y1


######################################


def get_verb_embeddings(mdl,  map_x1, map_x2, map_y1, ofn, embedding_layer=2, fn = None, vob = None):
    '''
    extract the verb embeddings
    :param mdl : the model
    :fm : feature manager
    :param fn: file containing the verbs, verbs are separated by a space
    :return: the new file containing  the embeddings of the verbs
    '''
    print (map_x1.input_key_map)
    if fn is not None:
        vobs = set()
        f = open(fn, "r")
        for l in f.readlines():
            tmps = l.split(" ")
            for tmp in tmps:
                if tmp != "":
                    vobs.add(tmp)

        vobs = list(vobs)
    else:
        if vob is not None:
            vobs = vob
    print (vobs)

    vv = []
    print (vobs)
    for v in vobs:
        if v  in map_x1.input_key_map:
            vv.append(v)

    vobs = vv
    print (vobs)

    X1 = [ [  v ] for v in vobs]
    X2 = [ ["PRED"] for v in vobs]

    Y1 = [["EOS_EOS"] for i in range(len(X1))]




    X1 = [[map_x1.input_key_map[x]  for x in XX] for XX in X1 ]


    X2 = [[map_x2.input_key_map[x]  for x in XX] for XX in X2 ]
    Y1 = [[map_y1.input_key_map[x] for x in XX] for XX in Y1]

    x1, mask_x, y, mask_y = preprare_seq_seq_data(X1, Y1)

    x2, mask_x = preprare_seq_seq_data(X2, None)

    x1, y, mask_x, mask_y, _, _, _, _ = mdl.standardize_data(x1, y, mask_x, mask_y, None, None, None, None)

    x2, _, _, _, _, _, _, _ = mdl.standardize_data(x2, None, None, None, None, None, None, None)

    rs = mdl.get_output_layer(embedding_layer, x1, x2, mask_x)
    print (rs.shape)
    f = open (ofn, "w")
    for i in range(len(vobs)):
        w = vobs[i]
        em = rs[0][i]
        f.write(w + " ")
        for e in em:
            f.write(str(e))
            f.write(" ")
        f.write("\n")
    f.close()


def get_embeddings(mdl, fm1, fm2, ofn, embedding_layer=2, fn = None, vob = None):
    '''
    extract the verb embeddings
    :param mdl : the model
    :fm : feature manager
    :param fn: file containing the verbs, verbs are separated by a space
    :return: the new file containing  the embeddings of the verbs
    '''

    if fn is not None:
        vobs = set()
        f = open(fn, "r")
        for l in f.readlines():
            tmps = l.split(" ")
            for tmp in tmps:
                if tmp != "":
                    tmps1=tmp.split(",")
                    if len(tmps1) ==2:
                        vobs.add((tmps1[0], tmps1[1]))

        vobs = list(vobs)
    else:
        if vob is not None:
            vobs = vob

    vv = []

    for v in vobs:
        if v[0]  in fm1.f.map.keys() and v[1] in fm2.f.map.keys():
            vv.append(v)

    vobs = vv

    X1 = [ ["EOS", v[0]] for v in vobs]
    X2 = [ ["EOS", v[1]] for v in vobs]

    Y1 = [["EOS","EOS"] for i in range(len(X1))]
    Y2 = [["EOS","EOS"] for i in range(len(X1))]

    X1 = [[fm1.f.map[fm1.f.getFeatureValue(x)] +1 for x in XX] for XX in X1 ]
    Y1 = [[fm1.fY.map[fm1.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y1]

    X2 = [[fm2.f.map[fm2.f.getFeatureValue(x)] +1 for x in XX] for XX in X2 ]
    Y2 = [[fm2.fY.map[fm2.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y2]



    x1, mask_x, y1, mask_y =preprare_seq_seq_data (X1, Y1)

    x2, mask_x, y2, mask_y = preprare_seq_seq_data(X2, Y2)

    x1, y1,  mask_x,mask_y, _, _,_, _ = mdl.standardize_data(x1, y1, mask_x, mask_y, None,None, None,None)

    x2, y2,  _,_, _, _,_, _ = mdl.standardize_data(x2, y2, None, None, None,None, None,None)
    print (x1)
    print (x2)
    print (mask_x)
    rs = mdl.get_output_layer(embedding_layer, x1, x2, mask_x)
    print (rs.shape)
    f = open (ofn, "w")
    for i in range(len(vobs)):
        w = vobs[i]
        em = rs[0][i]
        f.write(w + " ")
        for e in em:
            f.write(str(e))
            f.write(" ")
        f.write("\n")
    f.close()


###### get selectional preference scores  ##########



def solve( arg):

    candidate_dictl = arg[0]
    ofnl = arg[1]
    list_pl = arg[2]
    start = arg[3]
    end= arg[4]
    cfg = arg[5]
    mdl, fm1, fm2, fm3 = loadSemLM21 (arg[6], cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'])


    for i in range(start, end):
         run_pred( candidate_dictl,list_pl[i], ofnl, mdl, fm1, fm2, fm3)

def run_pred(vob, p, ofn, mdl, fm1, fm2, fm3):
    '''

    '''
    scores = get_probability_is_argument(mdl, fm1, fm2, fm3, p)

    process_probability(fm1, fm2, fm3, scores, p, ofn + "/" + p + ".out.txt", vobs=vob[p])


def calculate_selectional_preferences_parallel(config_path, model_path, list_preds, vob, output ,num_processes = 3):
    cfg = read_config(config_path)


    mdl, fm1, fm2, fm3 = loadSemLM21 (model_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'])

    list_p = list(vob.keys())
    for p in list_p:
        if p  not in fm1.input_key_map:
            vob.pop(p)
    list_p = list(vob.keys())

    total = len(list_p)
    part = total / (num_processes - 1)
    params = []
    for i in range(num_processes - 1):
        start, end = int(i * part), int((i + 1) * part)
        params.append((vob, output, list_p, start, end, cfg, model_path))


    start, end = int((num_processes - 1) * part), total
    params.append((vob, output, list_p, start, end, cfg, model_path))
    pool = Pool()
    pool.map(solve, params)


    #for p in list_p:
    #    print (p)
    #    scores  = get_probability_is_argument(mdl, fm, p)

    #    process_probability(fm, scores,  p, output + "/" + p + ".out.txt", vobs=vob[p])

    pool.close()





def calculate_selectional_preferences(config_path, model_path, list_preds, vob, output ):
    '''

    :param config_path:
    :param model_path:
    :param list_preds:
    :param vob: dictionary with key of preds,  values are list of words
    :param output:
    :return:
    '''
    cfg = read_config(config_path)

    mdl, fm1, fm2, fm3 = loadSemLM21Batch (model_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'], pos=cfg['pos'], data=cfg['data'], load_dict=False)

    list_p = list(vob.keys())
    for p in list_p:
        if p not in fm1.input_key_map:
            vob.pop(p)

    for p in vob.keys():
        scores  = get_probability_is_argument(mdl, fm1, fm2, fm3,p)

        process_probability(fm1, fm2, fm3, scores,  p, output + "/" + p + ".out.txt", vobs=vob[p])

def process_probability(fm1,fm2,fm3,  scores, pred, output, vobs=None, levels=3):
    f = open(output, "w")
    '''
    scores =[]
    for l in range(levels):
        print ('reading ...', l)
        scores_l = read_probability_file(dir, pred, l)
        scores.append(scores_l)
    '''

    candidates = vobs
    need_to_process_lst = []
    for c in candidates:
        for lbl in fm2.input_key_map.keys():
        #for lbl in fm2.f.map:

            if  c + "_" + lbl in fm3.input_key_map:

                need_to_process_lst.append((c,lbl ))



    for arg, lbl in need_to_process_lst:
        if arg != "EOS" and lbl != "EOS":
            try:
                final_score = 0.0

                pos1 = fm3.input_key_map[arg + "_" + lbl]
                exact_prob0 =  get_score( scores[0][0],  pos1)[0] # probability at the first position
                final_score+=exact_prob0
                continue_probs0  = np.asarray( scores[0][1])


                start_probs1 = continue_probs0

                exact_prob1 =  get_score( scores[1][0],  pos1)[0] # probability at the first position
                final_score+=np.sum(exact_prob1 * start_probs1)
                continue_probs1  = np.asarray( scores[1][1])

                start_probs1 = start_probs1.flatten().repeat(5)

                start_probs2 = start_probs1 * continue_probs1.flatten()



                exact_prob2 =  get_score( scores[2][0],  pos1)[0] # probability at the first position
                final_score+=np.sum(exact_prob2 * start_probs2)

                continue_probs2  = np.asarray( scores[2][1])
                start_probs2 = start_probs2.repeat(5)

                start_probs3 = start_probs2 * continue_probs2.flatten()

                exact_prob3 =  get_score( scores[3][0],  pos1)[0] # probability at the first position

                final_score+=np.sum(exact_prob3 * start_probs3)






                prob = final_score
                print (arg + "," + lbl + ": "+ str(prob))
                f.write(arg)
                f.write("_")
                f.write(lbl)
                f.write(":")
                f.write(str(prob))
                f.write(" ")
            except Exception:
                print("there is an exception here")

    f.close()




def get_score(sl, pos1):
        rs =[]

        print('len ', len(sl))
        for allval in sl:

            for val in allval[1]:

                if val[0]==pos1 :
                    p = val[1]


                    rs.append(p)

        return np.asarray(rs)


def get_probability_is_argument(mdl, fm1, fm2, fm3,predicate_words,  pos=4):
    '''
    calculate the probability that a word is an argument of a predicate
    :param mdl:
    :param fm:
    :param argument_word:
    :param argument_label:
    :return:
    '''
    # calculate the score at pos 0
    #
    X1 = []
    X2= []

    X_new1 = []
    X_new2= []


    X1.append([] )
    X_new1.append([predicate_words])

    X2.append([] )
    X_new2.append(['PRED'])

    score = []
    for p in range(pos):
        if p == 0:
            n = 5
        if p == 1:
            n = 5
        if  p >1:
            n = 5
        rs, rs_scores,   X1, X2, my_scores = get_scores_all(mdl, fm1, fm2,fm3, X1,X2, X_new1, X_new2, num_select=n)

        X_new1 = rs[0]
        X_new2 = rs[1]

        score.append((my_scores, rs_scores))

    return score



def to_string(lst1, lst2):
    s = ""
    for i in range(len(lst1)):
        s += str(lst1[i])+ "_" + str(lst2[i]) + ","
    s = s[0:len(s)-1]
    return s


def get_scores_all(mdl, fm1, fm2,fm3, X1, X2, X_new1, X_new2,   num_select = 10):
    X11 = []
    X21 = []

        # x = X[i], we add new values to the end of x

    for j in range(len(X_new1)):

            for k in range(len(X_new1[j])):
                xx =[ xxx for xxx in X1[j]]
                xx.append(X_new1[j][k] )
                X11.append(xx)


    for j in range(len(X_new2)):

            for k in range(len(X_new2[j])):
                xx =[ xxx for xxx in X2[j]]
                xx.append(X_new2[j][k] )
                X21.append(xx)



    X1 = [[fm1.input_key_map[x]  for x in XX] for XX in X11 ]

    X2 = [[fm2.input_key_map[x]  for x in XX] for XX in X21 ]



    x1,x_mask1= preprare_seq_seq_data(X1)

    x1, _,  mask_x1,_, _, _,_, _ = mdl.standardize_data(x1, None, x_mask1, None, None,None, None,None)

    x2,x_mask2= preprare_seq_seq_data(X2)

    x2, _,  mask_x2,_, _, _,_, _ = mdl.standardize_data(x2, None, x_mask2, None, None,None, None,None)


    score_pos = mdl.get_output_layer(-1, x1, x2, mask_x1)



    score_pos=score_pos.swapaxes(0,1)
    score_pos = score_pos[:,-1]


    x = T.matrix("score")


    sort_f = th.function([x], T.argsort(x))

    sorted_values = sort_f(score_pos)
    rs1 = []
    rs2 = []
    rs_scores = []
    my_scores = []
    for i in range(sorted_values.shape[0]):
        #f.write(to_string(X1[i]) + " ")
        ss=[]
        for j in range(1,sorted_values.shape[1]):
            val = sorted_values[i][sorted_values.shape[1]-j]

            #val_map = fm.fY.map_inversed[val-1]
            score = score_pos[i][val]
            #f.write(str(val) + ":" + str(score) + " ")
            ss.append((val,score))
        #f.write("\n")
        my_scores.append(("_", ss))



        vals = []
        c = 0
        for t in range(sorted_values.shape[1]-1, -1, -1):
            if c == num_select:
                break
            v = sorted_values[i][t]

            if fm3.get_key(v)!="EOS_EOS" :
                tm = fm3.get_key(v).split("_")
                if tm[0] in fm1.input_key_map and tm[1] in fm2.input_key_map:
                    vals.append(v)
                    c+=1
        #vals = sorted_values[i][sorted_values.shape[1]-num_select:sorted_values.shape[1]]

        vals1=[]
        vals2 = []

        #val_maps = [fm1.fY.map_inversed[v-1].split("_") for v in list(vals) ]#if  fm.fY.map_inversed[v-1]!="EOS" ]
        scores = [score_pos[i][v] for v in list(vals)]# if fm.fY.map_inversed[v-1]!="EOS"]

        for  v in list(vals):
            tm = fm3.get_key(v).split("_")
            vals1.append(tm[0])
            vals2.append(tm[1])

        rs1.append(vals1)
        rs2.append(vals2)

        rs_scores.append(scores)




    return (rs1,rs2), rs_scores,   X11, X21, my_scores
