__author__ = 'quynhdo'

class Word(object):

    def __init__(self,id=None, form=None, lemma=None, pos=None, head=None, deprel=None):
            self.id = id
            self.form = form
            self.lemma = lemma
            self.pos = pos
            self.head = head
            self.deprel = deprel

    def make_predicate(self, sense="01"):
        '''
        Convert a word to a predicate
        '''
        self.__class__ = Predicate
        self.sense = sense
        self.arguments = {}


class Predicate(Word):

    def clear(self):
        self.sense = None
        self.arguments = {}

