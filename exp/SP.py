

from exp.ConfigReader import read_config
from models.Model11 import loadSemLM11, get_verb_embeddings
from models.Model21 import loadSemLM21

__author__ = 'quynhdo'


#### compute word embeddings
import argparse


__author__ = 'quynhdo'




def read_iSRL_candidates_new(fn):
    f = open(fn,  "r")
    data = {}
    for l in f.readlines():
        tmps = l.split(" ")
        fr = tmps[0]

        if fr in data.keys():
            v = data[fr]
        else:
            v = set()
        for i in range(1, len(tmps)):
            v.add(tmps[i])

        data[fr] = v
    return data

def loadModel11( config_path, pred_file, output, parrallel=False):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/best_simple.pkl"
    vob=read_iSRL_candidates_new(pred_file)
    from models.Model11 import calculate_selectional_preferences_parallel, calculate_selectional_preferences

    if parrallel:
        calculate_selectional_preferences_parallel(config_path,load_path, list(vob.keys()), vob, output)
    else:

        calculate_selectional_preferences(config_path,load_path, list(vob.keys()), vob, output)

def loadModel21(config_path, pred_file, output, parrallel=False):
    cfg = read_config(config_path)
    #load_path = cfg['save_folder'] + "/best_adv.pkl"
    load_path = cfg['save_folder'] + "/adv.pkl_FINAL_29"
    vob = read_iSRL_candidates_new(pred_file)
    from models.Model21 import calculate_selectional_preferences_parallel, calculate_selectional_preferences

    if parrallel:
        calculate_selectional_preferences_parallel(config_path, load_path, list(vob.keys()), vob, output)
    else:

        calculate_selectional_preferences(config_path, load_path, list(vob.keys()), vob, output)

def loadModel22(config_path, pred_file, output):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/hybrid.pkl"
    mdl, fm1,fm2 = loadSemLM22(load_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],load_dt=cfg['load_data'], continue_train=False)



if __name__=="__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("exp_config", help="exp configuration file")
    parser.add_argument("pred_file", help="pred file")
    parser.add_argument("output_file", help="output file")



    parser.add_argument('--11', dest='mdl11', action='store_true')
    parser.add_argument('--21', dest='mdl21', action='store_true')
    parser.add_argument('--22', dest='mdl22', action='store_true')
    parser.add_argument('--p', dest='parallel', action='store_true')


    args = parser.parse_args()


    if args.mdl11:


            loadModel11(args.exp_config, args.pred_file, args.output_file, args.parallel)
    if args.mdl21:
        loadModel21(args.exp_config, args.pred_file, args.output_file, args.parallel)

    if args.mdl22:
        loadModel22(args.exp_config, args.pred_file, args.output_file)

