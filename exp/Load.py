from exp.ConfigReader import read_config
from models.Model11 import loadSemLM11
from models.Model21 import loadSemLM21

__author__ = 'quynhdo'
import argparse


__author__ = 'quynhdo'




def loadModel11( config_path):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/simple.pkl.4"
    load_best=cfg['save_folder'] + "/best_simple.pkl"


    loadSemLM11( load_path, cfg['train'], cfg['valid'], cfg['dep'],
                 cfg['hidden_size'], cfg['batch_size'], cfg['save_folder'],  continue_train=True, load_best=load_best, continue_epoch=5)

def loadModel21(config_path):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/adv.pkl.29"
    load_best = cfg['save_folder'] + "/best_adv.pkl"

    loadSemLM21(load_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'], continue_train=True, load_best=load_best, continue_epoch=29)


def loadModel22(config_path):
    cfg = read_config(config_path)
    load_path = cfg['save_folder'] + "/hybrid.pkl"
    loadSemLM22(load_path, cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],load_dt=cfg['load_data'], continue_train=True)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("exp_config", help="exp configuration file")

    parser.add_argument('--11', dest='mdl11', action='store_true')
    parser.add_argument('--21', dest='mdl21', action='store_true')
    parser.add_argument('--22', dest='mdl22', action='store_true')



    args = parser.parse_args()


    if args.mdl11:
        loadModel11(args.exp_config)
    if args.mdl21:
        loadModel21(args.exp_config)

    if args.mdl22:
        loadModel22(args.exp_config)




    #trainModel22("/Users/quynhdo/Documents/PhDFinal/workspace/NewSemLM/config/exp.config")