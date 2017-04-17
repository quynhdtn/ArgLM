

from exp.ConfigReader import read_config
from models.Model11 import loadSemLM11
from models.Model21 import loadSemLM21, get_verb_embeddings

__author__ = 'quynhdo'


#### compute word embeddings
import argparse


__author__ = 'quynhdo'




def loadModel11( config_path, pred_file, output):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/best_simple.pkl"

    mdl, fm = loadSemLM11( load_path, cfg['train'], cfg['valid'],  cfg['dep'],
                 cfg['hidden_size'], cfg['batch_size'], cfg['save_folder'], continue_train=False)
    get_verb_embeddings(mdl, fm, output, fn=pred_file, embedding_layer=2)

def loadModel21(config_path, pred_file, output):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/best_adv.pkl"

    mdl, map_x1, map_x2, map_y1 = loadSemLM21(load_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],continue_train=False)

    #mdl, fm1,fm2 = None,None,None
    get_verb_embeddings(mdl, map_x1, map_x2, map_y1, output, fn=pred_file, embedding_layer=4)

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



    args = parser.parse_args()


    if args.mdl11:
        loadModel11(args.exp_config, args.pred_file, args.output_file)
    if args.mdl21:
        loadModel21(args.exp_config, args.pred_file, args.output_file)

    if args.mdl22:
        loadModel22(args.exp_config, args.pred_file, args.output_file)

