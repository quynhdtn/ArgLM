import re

from readers.Conll2009Reader import read_conll2009_sentence,  read_conll2009_corpus


class Reader(object):
    def __init__(self, input_file):
        self.input_file=input_file

    def readAll(self):
        raise NotImplementedError("Subclasses should implement this!")

class BatchReader(Reader):
    def __init__(self, batch_size, input_file):
        Reader.__init__(self,input_file)
        self.batch_size = batch_size
        self.current_position = 0
        self.current_file = 0



    def readAll(self):
        raise NotImplementedError("Subclasses should implement this!")

    def next(self):
        raise NotImplementedError("Subclasses should implement this!")

    def reset(self):
        self.current_position = 0
        self.current_file = 0


class SimulateReader(BatchReader):
    def __init__(self, batch_size, data, input_file=None):
        BatchReader.__init__(self,batch_size,input_file)

        self.data = data

    def readAll(self):
        return self.data




class Conll2009BatchReader(BatchReader):

    def __init__(self, batch_size, input_file, read_label=True, use_gold=False):
        BatchReader.__init__(self,batch_size,input_file)

        self.read_label = read_label
        self.use_gold = use_gold

    def next(self):
        txt = []
        if self.current_file >= len(self.input_file):
            return txt

        txt.extend(self.readConll2009SentencesRange(self.input_file[self.current_file], self.current_position, self.current_position + self.batch_size,
                                        self.read_label, self.use_gold))
        self.current_position += len(txt)

        while len(txt) < self.batch_size:
            self.current_position = 0
            self.current_file +=1
            if self.current_file >= len(self.input_file):
                return txt
            s1 = len(txt)
            txt.extend(self.readConll2009SentencesRange(self.input_file[self.current_file], self.current_position, self.current_position + self.batch_size - len(txt),
                                        self.read_label, self.use_gold))

            self.current_position += len(txt) - s1




        return txt

    def readAll(self):
        txt = []
        for f in self.input_file:
            txt=read_conll2009_corpus(f)
        return txt

    def readConll2009SentencesRange(self, path, start, end=None, read_label=True, use_gold=False):
        txt = []
        f = open(path, 'r')
        sens = []
        words = []
        idx = 0
        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    if end is None:
                        if idx >= start:
                            sens.append(words)

                    else:

                        if idx >= start:
                            if idx < end:
                                sens.append(words)
                            else:
                                break
                    idx += 1
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            if end is None:
                if idx >= start:
                    sens.append(words)

            else:

                if idx >= start:
                    if idx < end:
                        sens.append(words)

        for sen in sens:
            conll2009sen = read_conll2009_sentence(sen, read_label, use_gold=use_gold)
            txt.append(conll2009sen)
        return txt

if __name__ == "__main__":
    lst = ["/home/quynh/working/Data/conll2009/train.conll2009.pp.txt"]
    reader = Conll2009BatchReader(1000, lst)
    count = 0
    while  True:
        txt =  reader.next()
        if len(txt) == 0:
            break
        count+=len(txt)

    print (count)

    txt =  read_conll2009_corpus("/home/quynh/working/Data/conll2009/train.conll2009.pp.txt")
    print (len(txt))