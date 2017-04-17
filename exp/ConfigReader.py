__author__ = 'quynhdo'
## to read an experiment configuration


def read_config(config_path):
    config = {'train':[], 'valid':[],'data':'form','pos':'V','eos':True, 'model':'11',
              'batch_size':200, 'hidden_size':16, 'dep':1,
              'we_dict':None, 'we_dict1':None,  'we_dict2':None, "load_data":True}

    f = open(config_path, 'r')
    for l in f.readlines():
        tmps= l.strip().split(" ")
        if len(tmps)==2:
            if tmps[0] in ['train', 'valid']:
                lst = config[tmps[0]]
                lst.append(tmps[1])
                config[tmps[0]] = lst

            elif tmps[0]=='eos':
                config[tmps[0]] = bool(tmps[1])
            elif tmps[0] in ['dep', 'batch_size', 'hidden_size', 'hidden_size1', 'hidden_size2', 'max_epochs']:
                config[tmps[0]] = int(tmps[1])

            else:
                config[tmps[0]] = tmps[1]

    return config


print (bool(1))