import argparse

import time

from exp.ConfigReader import read_config
from models.Model11 import trainSemLM11
from models.Model21 import trainSemLM21, trainSemLM21Batch

__author__ = 'quynhdo'



def trainModel11(config_path):
    cfg = read_config(config_path)

    trainSemLM11(cfg['train'], cfg['valid'], cfg['dep'],
                 cfg['hidden_size'], cfg['batch_size'], cfg['save_folder'], pos=cfg['pos'], max_epochs=cfg['max_epochs'], data=cfg["data"])

def trainModel21(config_path):
    cfg = read_config(config_path)


    trainSemLM21Batch(cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],
                  pos=cfg['pos'],max_epochs=cfg["max_epochs"], data=cfg["data"])


def trainModel22(config_path):
    cfg = read_config(config_path)

    #trainSemLM22(cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
    #             cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],load_dt=cfg['load_data'])

if __name__=="__main__":
    tic = time.clock()
    parser = argparse.ArgumentParser()

    parser.add_argument("exp_config", help="exp configuration file")

    parser.add_argument('--11', dest='mdl11', action='store_true')
    parser.add_argument('--21', dest='mdl21', action='store_true')
    parser.add_argument('--22', dest='mdl22', action='store_true')



    args = parser.parse_args()


    if args.mdl11:
        trainModel11(args.exp_config)
    if args.mdl21:
        trainModel21(args.exp_config)

    if args.mdl22:
        trainModel22(args.exp_config)

    toc = time.clock()

    print(tic-toc)




    #trainModel22("/Users/quynhdo/Documents/PhDFinal/workspace/NewSemLM/config/exp.config")