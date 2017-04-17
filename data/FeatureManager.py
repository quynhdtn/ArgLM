import pickle
from liir.nlp.features.IndexedFeature import IndexedFeature

__author__ = 'quynhdo'


# this class to control the features



class FeatureManager:
    def __init__(self):
        self.f = IndexedFeature('simple', self.getForm, is1Hot=False)
        self.fY = IndexedFeature('simple', self.getForm, is1Hot=False)


    def getForm(self, str, kargs=None):
            return str


    def extract_features(self, seqX, seqY):
        '''

        :param seqX: list : data seq
        :param seqY: list : label
        :return:
        '''

        for x in seqX:
            self.f.addFeatureValueToMap(self.f.getFeatureValue(x))

        for y in seqY:
            self.fY.addFeatureValueToMap(self.fY.getFeatureValue(y))


    def get_representation(self, seqX, seqY):
        return [self.f.map[self.f.getFeatureValue(x)] for x in seqX], [self.fY.map[self.fY.getFeatureValue(y)] for y in seqY]


    def save(self, dir):
        pickle.dump(self, open(dir, "wb"))

    def load(dir):
        return pickle.load(open(dir, "rb"))





