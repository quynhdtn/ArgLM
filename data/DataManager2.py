import pickle
import  numpy as np

from readers.BatchReader import Conll2009BatchReader
from readers.Conll2009Reader import read_conll2009_corpus
from representations.Word import Predicate
from utils.Data import prepare_data_seq



def generate_sequential_data11(lst, lstv = None, pos="V", data="form", count=100000):
    corpus = []
    all_data = []
    map = Mapping()
    for i in range(1,len(lst)):
        txt = read_conll2009_corpus(lst[i])
        corpus.extend(txt)
    if count is not None:
        c = 0
        corpus_select = []
        for s in corpus:
            c+=1
            if c <count:
                corpus_select.append(s)
            else:
                break
        corpus = corpus_select
    corpus.extend(read_conll2009_corpus(lst[0]))
    for sen in corpus:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w,"pos")[0]==pos:
                    lst=[map.add_value(w.form + "_PRED" )]
                    arglst = []
                    for arg in w.arguments:
                        if data=="origin":
                            hn=sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append(map.add_value(hn +"_" + w.arguments[arg]))
                    arglst.append(map.add_value("EOS"))
                    lst.extend(arglst)
                    all_data.append(lst)
    X = [[x[i] for i in range(len(x)-1) ] for x in all_data]
    Y = [[x[i+1] for i in range(len(x) - 1)] for x in all_data]
    Xv=None
    Yv=None
    if lstv is not None:
        corpusv = []
        all_datav = []
        for fn in lstv:
            txt = read_conll2009_corpus(fn)
            corpusv.extend(txt)

        for sen in corpusv:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:
                    lst = [map.add_value(w.form + "_PRED" )]
                    arglst = []
                    for arg in w.arguments:
                        if data=="origin":
                            hn=sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append(map.add_value(hn + "_" + w.arguments[arg]))
                    arglst.append(map.add_value("EOS"))
                    lst.extend(arglst)
                    all_datav.append(lst)
        Xv = [[x[i] for i in range(len(x) - 1)] for x in all_datav]
        Yv = [[x[i + 1] for i in range(len(x) - 1)] for x in all_datav]
    return X,Y,Xv,Yv, map


def generate_sequential_data21(lst, lstv = None, pos="V", data="form", count=100000):
    corpus = []
    all_data = []
    mapX1 = Mapping()
    mapX2 = Mapping()
    mapY1 = Mapping()
    for i in range(1, len(lst)):
        txt = read_conll2009_corpus(lst[i])
        corpus.extend(txt)
    if count is not None:
        c = 0
        corpus_select = []
        for s in corpus:
            c += 1
            if c < count:
                corpus_select.append(s)
            else:
                break
        corpus = corpus_select
    corpus.extend(read_conll2009_corpus(lst[0]))
    print (len(corpus))
    for sen in corpus:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w,"pos")[0]==pos:

                    lst=[(mapX1.add_value(w.form), mapX2.add_value("PRED"), mapY1.add_value(w.form+"_PRED"))]
                    arglst = []
                    for arg in w.arguments:
                        if data=="origin":
                            hn=sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append( (mapX1.add_value(hn), mapX2.add_value(w.arguments[arg]), mapY1.add_value(hn+"_"+w.arguments[arg])))
                    arglst.append((mapX1.add_value("EOS"), mapX2.add_value("EOS"), mapY1.add_value("EOS_EOS")))
                    lst.extend(arglst)
                    all_data.append(lst)

    X1 = [[x[i][0] for i in range(len(x)-1) ] for x in all_data]
    X2 = [[x[i][1] for i in range(len(x) - 1)] for x in all_data]
    Y = [[x[i+1][2] for i in range(len(x) - 1)] for x in all_data]
    Xv1=None
    Xv2=None
    Yv=None
    if lstv is not None:
        corpusv = []
        all_datav = []
        for fn in lstv:
            txt = read_conll2009_corpus(fn)
            corpusv.extend(txt)

        for sen in corpusv:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                    lst=[(mapX1.add_value(w.form), mapX2.add_value("PRED"), mapY1.add_value(w.form+"_PRED"))]
                    arglst = []
                    for arg in w.arguments:
                        if data=="origin":
                            hn=sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append( (mapX1.add_value(hn), mapX2.add_value(w.arguments[arg]), mapY1.add_value(hn+"_"+w.arguments[arg])))
                    arglst.append((mapX1.add_value("EOS"), mapX2.add_value("EOS"), mapY1.add_value("EOS_EOS")))
                    lst.extend(arglst)
                    all_datav.append(lst)
        Xv1 = [[x[i][0] for i in range(len(x) - 1)] for x in all_datav]
        Xv2= [[x[i][1] for i in range(len(x) - 1)] for x in all_datav]
        Yv = [[x[i + 1][2] for i in range(len(x) - 1)] for x in all_datav]
    return X1,X2,Y,Xv1,Xv2,Yv, mapX1,mapX2,mapY1


def generate_sequential_data21_getmap(lst, lstv=None, pos="V", data="form"):
    corpus = []
    mapX1 = Mapping()
    mapX2 = Mapping()
    mapY1 = Mapping()
    for fn in lst:
        txt = read_conll2009_corpus(fn)
        corpus.extend(txt)

    for sen in corpus:
        for w in sen:
            if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                mapX1.add_value(w.form)
                mapX2.add_value("PRED")
                mapY1.add_value(w.form + "_PRED")

                for arg in w.arguments:
                    if data == "origin":
                        hn = sen[arg].form
                    else:
                        hn = get_represented_form(sen, arg)
                    if hn is not None:
                        mapX1.add_value(hn)
                        mapX2.add_value(w.arguments[arg])
                        mapY1.add_value(hn + "_" + w.arguments[arg])
                mapX1.add_value("EOS")
                mapX2.add_value("EOS")
                mapY1.add_value("EOS_EOS")

    return mapX1, mapX2, mapY1

def generate_vob21(lst,  lstv=None, pos="V", data="form"):

    mapX1 = Mapping()
    mapX2 = Mapping()
    mapY1 = Mapping()
    if lstv is not None:
        lst.extend(lstv)

    reader =   Conll2009BatchReader(100000, lst)
    i = 0
    while True:
        txt = reader.next()
        if len(txt) == 0:
            break

        for sen in txt:
            i=i+1
            if (i%1000 == 0):
                print ("finished ", i)
            for w in sen:
                if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                    (mapX1.add_value(w.form), mapX2.add_value("PRED"), mapY1.add_value(w.form + "_PRED"))
                    for arg in w.arguments:
                        if data == "origin":
                            hn = sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            (mapX1.add_value(hn), mapX2.add_value(w.arguments[arg]),
                                           mapY1.add_value(hn + "_" + w.arguments[arg]))
                    (mapX1.add_value("EOS"), mapX2.add_value("EOS"), mapY1.add_value("EOS_EOS"))



    return mapX1, mapX2, mapY1


def generate_sequential_data21(corpus, lstv=None, pos="V", data="form", mapX1=None, mapX2=None, mapY1=None):

    all_data = []


    if mapX1 is not None:
        for sen in corpus:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                    lst = [(mapX1.get_index(w.form), mapX2.get_index("PRED"), mapY1.get_index(w.form + "_PRED"))]
                    arglst = []
                    for arg in w.arguments:
                        if data == "origin":
                            hn = sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append((mapX1.get_index(hn), mapX2.get_index(w.arguments[arg]),
                                           mapY1.get_index(hn + "_" + w.arguments[arg])))
                    arglst.append((mapX1.get_index("EOS"), mapX2.get_index("EOS"), mapY1.get_index("EOS_EOS")))
                    lst.extend(arglst)
                    all_data.append(lst)

        X1 = [[x[i][0] for i in range(len(x) - 1)] for x in all_data]
        X2 = [[x[i][1] for i in range(len(x) - 1)] for x in all_data]
        Y = [[x[i + 1][2] for i in range(len(x) - 1)] for x in all_data]
        Xv1 = None
        Xv2 = None
        Yv = None
        if lstv is not None:
            corpusv = []
            all_datav = []
            for fn in lstv:
                txt = read_conll2009_corpus(fn)
                corpusv.extend(txt)

            for sen in corpusv:
                for w in sen:
                    if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                        lst = [(mapX1.get_index(w.form), mapX2.get_index("PRED"), mapY1.get_index(w.form + "_PRED"))]
                        arglst = []
                        for arg in w.arguments:
                            if data == "origin":
                                hn = sen[arg].form
                            else:
                                hn = get_represented_form(sen, arg)
                            if hn is not None:
                                arglst.append((mapX1.get_index(hn), mapX2.get_index(w.arguments[arg]),
                                               mapY1.get_index(hn + "_" + w.arguments[arg])))
                        arglst.append((mapX1.get_index("EOS"), mapX2.get_index("EOS"), mapY1.get_index("EOS_EOS")))
                        lst.extend(arglst)
                        all_datav.append(lst)
            Xv1 = [[x[i][0] for i in range(len(x) - 1)] for x in all_datav]
            Xv2 = [[x[i][1] for i in range(len(x) - 1)] for x in all_datav]
            Yv = [[x[i + 1][2] for i in range(len(x) - 1)] for x in all_datav]
        return X1, X2, Y, Xv1, Xv2, Yv, mapX1, mapX2, mapY1

    else:
        mapX1 = Mapping()
        mapX2 = Mapping()
        mapY1 = Mapping()
        for txttrain in corpus:
            all_data.extend()
        for sen in corpus:
            for w in sen:
                if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                    lst = [(mapX1.add_value(w.form), mapX2.add_value("PRED"), mapY1.add_value(w.form + "_PRED"))]
                    arglst = []
                    for arg in w.arguments:
                        if data == "origin":
                            hn = sen[arg].form
                        else:
                            hn = get_represented_form(sen, arg)
                        if hn is not None:
                            arglst.append((mapX1.add_value(hn), mapX2.add_value(w.arguments[arg]),
                                           mapY1.add_value(hn + "_" + w.arguments[arg])))
                    arglst.append((mapX1.add_value("EOS"), mapX2.add_value("EOS"), mapY1.add_value("EOS_EOS")))
                    lst.extend(arglst)
                    all_data.append(lst)

        X1 = [[x[i][0] for i in range(len(x) - 1)] for x in all_data]
        X2 = [[x[i][1] for i in range(len(x) - 1)] for x in all_data]
        Y = [[x[i + 1][2] for i in range(len(x) - 1)] for x in all_data]
        Xv1 = None
        Xv2 = None
        Yv = None
        if lstv is not None:
            corpusv = []
            all_datav = []
            for fn in lstv:
                txt = read_conll2009_corpus(fn)
                corpusv.extend(txt)

            for sen in corpusv:
                for w in sen:
                    if isinstance(w, Predicate) and getattr(w, "pos")[0] == pos:

                        lst = [(mapX1.add_value(w.form), mapX2.add_value("PRED"), mapY1.add_value(w.form + "_PRED"))]
                        arglst = []
                        for arg in w.arguments:
                            if data == "origin":
                                hn = sen[arg].form
                            else:
                                hn = get_represented_form(sen, arg)
                            if hn is not None:
                                arglst.append((mapX1.add_value(hn), mapX2.add_value(w.arguments[arg]),
                                               mapY1.add_value(hn + "_" + w.arguments[arg])))
                        arglst.append((mapX1.add_value("EOS"), mapX2.add_value("EOS"), mapY1.add_value("EOS_EOS")))
                        lst.extend(arglst)
                        all_datav.append(lst)
            Xv1 = [[x[i][0] for i in range(len(x) - 1)] for x in all_datav]
            Xv2 = [[x[i][1] for i in range(len(x) - 1)] for x in all_datav]
            Yv = [[x[i + 1][2] for i in range(len(x) - 1)] for x in all_datav]
        return X1, X2, Y, Xv1, Xv2, Yv, mapX1, mapX2, mapY1



'''
def get_represented_form(sen, wid):
    w = sen[wid]
    pos = getattr(w, "pos")
    if pos[0] == "N" or pos[0] == "P":
        return getattr(w,"form")
    if pos == "IN" or pos == "TO":
        for i in range(len(sen)):
            c=sen[i]
            if c.head == wid:
                posc = getattr(c, "pos")
                if posc[0]=="N" or posc[0]=="P":
                    return getattr(c,"form")
    return None
'''


def get_represented_form(sen, wid):
    w = sen[wid]

    return getattr(w,"form")



class Mapping:
    def __init__(self):
        self.input_key_map = {}

        self.current_index_input = 1

    def add_value(self, invalue):
        inv = None
        if invalue in self.input_key_map:
            inv = self.input_key_map[invalue]
        else:
            inv = self.current_index_input
            self.current_index_input+=1
            self.input_key_map[invalue]=inv
        return inv

    def get_key(self,val):
        for k in self.input_key_map:
            if self.input_key_map[k] == val:
                return k

    def get_index(self,val):
        return self.input_key_map[val]

def preprare_seq_seq_data(X, Y=None):
            x, x_mask = prepare_data_seq(X)
            # xaa= np.asarray(x)
            # print (xaa.shape)
            x = np.asarray(x, dtype='int32')

            x_mask = np.asarray(x_mask, dtype='int32')
            if Y is not None:
                y, y_mask = prepare_data_seq(Y)
                # xaa= np.asarray(y)
                # print (xaa.shape)
                y = np.asarray(y, dtype='int32')
                y_mask = np.asarray(y_mask, dtype='int32')

                # y = y.flatten()
                # y_mask = y_mask.flatten()

                return x, x_mask, y, y_mask

            return x, x_mask

if __name__ == "__main__":
    lst = ["/home/quynh/working/Data/conll2009/ood.conll2009.pp.txt"]
    X1, X2, Y, Xv1, Xv2, Yv, mapX1, mapX2, mapY1 = generate_sequential_data21(lst)
    print (X1)
    print (X2)

