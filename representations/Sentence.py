from representations.Word import Predicate

__author__ = 'quynhdo'

class Sentence(list):

    def __init__(self):   # value should be  a list of Word
        list.__init__(self)

    def get_predicates(self):
        return [w for w in self if isinstance(w, Predicate)]

    def to_string(self):
        return " ".join([w.form for w in self])


