__author__ = 'quynhdo'

import numpy as np
import re

class WEDict:

    def __init__(self, full_dict_path=None, do_scale=False):
        f = open(full_dict_path, "r")
        self.full_dict = {}
        self.we_size = 0
        for l in f.readlines(): # read the full dictionary
            l = l.strip()
            tmps = re.split('\s+', l)
            if len(tmps) > 1:
                we = []
                for i in range(1, len(tmps)):
                    we.append(float(tmps[i]))
                self.we_size = len(we)
                self.full_dict[tmps[0]] = np.asarray(we)

        f.close()
        if do_scale:
            self.scale()

    def merge_WEdict(self, d):
        '''
        merge the current dict with another wedict
        :param d:
        :return:
        '''
        for k in self.full_dict.keys():
            arr1 = self.full_dict[k]
            arr2 = d.full_dict[k]
            arr= np.concatenate((arr1,arr2))
            self.full_dict[k]=arr
        self.we_size += d.we_size


    def get_we(self, w):
        if w in self.full_dict:
            we = self.full_dict[w]
        else:
            we = np.zeros(self.we_size)
        return we

    def extract_we_for_vob(self, vob, output):
        f = open(output, "w")
        c = 0
        for w in vob:
            if w in self.full_dict.keys():
                f.write(w)
                f.write(" ")
                we = self.full_dict[w]
                c += 1
                for val in we:
                    f.write(str(val))
                    f.write(" ")
                f.write("\n")
        f.close()
        print ( "Words in WE dict: ")
        print (str(c) + "/" + str(len(vob)))

    def write_to_file(self, output):
        f = open(output, "w")
        for w in self.full_dict.keys():
            f.write(w)
            f.write(" ")
            for v in self.full_dict[w]:
                f.write(str(v))
                f.write(" ")
            f.write("\n")
        f.close()

    def scale(self):
        k,t = self.getFullVobWEAndKeys()

        t = 0.1 * t / np.std(t)
        for kk in range(k.size):
            self.full_dict[k[kk]] = t[kk,:]



