from we.WEDict import WEDict

__author__ = 'quynhdo'

class WEManager(dict):
    def __init__(self, we_list_file=None):
        dict.__init__(self)
        if we_list_file is not None:
            f = open(we_list_file, 'r')
            for line in f.readlines():
                line=line.strip()
                tmps = line.split(" ")
                if len(tmps) == 2:
                    wed = WEDict(tmps[1])
                    self[tmps[0]]=wed
                if len(tmps) > 2:
                    wed = WEDict(tmps[1])
                    for j in range(2, len(tmps)):
                        owed = WEDict(tmps[j])
                        wed.mergeWEDict(owed)
                    self[tmps[0]]=wed