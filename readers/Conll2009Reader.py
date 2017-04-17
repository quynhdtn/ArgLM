import resource
from representations.Word import Word, Predicate
from representations.Sentence import Sentence

__author__ = 'quynhdo'

def get_predicate_list(list_paths):
    lst = []
    corpus =[]
    for lp in list_paths:
        txt = read_conll2009_corpus(lp)
        corpus.extend(txt)
    for s in corpus:
        for p in s.get_predicates():
            lst.append(p.form)
    return lst


def read_conll2009_corpus(path, read_annotation=True, use_gold=False, filter_list_predicates=None):
    f = open(path, 'r')
    sens=[]
    words=[]
    append = list.append
    strip = str.strip

    for line in f:
        line = strip(line)
        if not line:
            if words:
                append(sens,words)
                words = []
        else:
            append(words, line)

    if words:
        append(sens,words)

    corpus = []
    for sen in sens:
        corpus.append(read_conll2009_sentence(sen, read_annotation, use_gold))
    return corpus


def read_conll2009_corpus(path, read_annotation=True, use_gold=False):
    f = open(path, 'r')
    sens=[]
    words=[]
    append = list.append
    strip = str.strip

    for line in f:
        line = strip(line)
        if not line:
            if words:
                append(sens,words)
                words = []
        else:
            append(words, line)

    if words:
        append(sens,words)

    corpus = []
    for sen in sens:
        corpus.append(read_conll2009_sentence(sen, read_annotation, use_gold))
    return corpus


def read_unannotated_conll2009_corpus(path,  use_gold=False):
    f = open(path, 'r')
    sens=[]
    words=[]
    append = list.append
    strip = str.strip

    for line in f:
        line = strip(line)
        if not line:
            if words:
                append(sens,words)
                words = []
        else:
            append(words, line)

    if words:
        append(sens,words)

    corpus = []
    for sen in sens:
        corpus.append(read_unannotated_conll2009_sentence(sen,  use_gold))
    return corpus


def read_conll2009_sentence(sentence_lines, read_annotation=True, use_gold=False):
    sen_conll2009 = Sentence()
    arguments_store = []
    predicates = []
    if not use_gold:
            for line in sentence_lines:
                temps = line.split('\t')
                word = Word(int(temps[0])-1, temps[1].lower(), temps[3], temps[5], int(temps[9])-1, temps[11])
                if temps[12] == "Y":
                    word.make_predicate(temps[13].split(".")[1])
                    predicates.append(word.id)
                arguments_store.append(temps[14:])
                sen_conll2009.append(word)
    else:
            for line in sentence_lines:
                temps = line.split('\t')
                word = Word(int(temps[0])-1, temps[1].lower(), temps[2], temps[4], int(temps[8])-1, temps[10])
                if temps[12] == "Y":

                    word.make_predicate(temps[13].split(".")[1])
                    predicates.append(word.id)
                arguments_store.append(temps[14:])
                sen_conll2009.append(word)

    if read_annotation:
        for i in range(len(predicates)):
                p_id = predicates[i]
                for j in range(len(arguments_store)):
                    lbl = arguments_store[j][i]
                    if lbl != "_":
                        sen_conll2009[p_id].arguments[j]=lbl

    return sen_conll2009

def read_unannotated_conll2009_sentence(sentence_lines, use_gold=False):
    sen_conll2009 = Sentence()

    if not use_gold:
            for line in sentence_lines:
                temps = line.split('\t')
                word = Word(int(temps[0])-1, temps[1].lower(), temps[3], temps[5], int(temps[9])-1, temps[11])
                sen_conll2009.append(word)
    else:
            for line in sentence_lines:
                temps = line.split('\t')
                word = Word(int(temps[0])-1, temps[1].lower(), temps[2], temps[4], int(temps[8])-1, temps[10])

                sen_conll2009.append(word)


    return sen_conll2009



if __name__ == "__main__":
    import time

    print (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    corpus = read_conll2009_corpus("/home/quynh/working/old/Desktop/data_split/201.train.conll2009.pp.txt-brown2-eval.out")

    for i in range(100):
        print (corpus[i].to_string())
    #print (len(corpus))

    #print (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #tic = time.clock()
    #c = 0

    #f = open("/home/quynh/working/Data/conll2009/predicate.train.txt","w")
    #for s in corpus:
    #    for w in s:
    #        if isinstance(w, Predicate):
    #            f.write(w.form+" ")

    #f.close()
    #toc = time.clock()


    #print (toc - tic)








