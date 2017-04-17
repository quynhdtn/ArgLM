__author__ = 'quynhdo'

class Option(dict):

    INPUT_DIM = 'input_dim'
    INPUT_LENGTH = 'input_length'

    OUTPUT_DIM = 'output_dim'
    OUTPUT_LENGTH = 'output_length'

    MAX_EPOCHS = 'max_epochs'
    PATIENCE = 'patience'

    SAVE_FREQ = 'saveFreq'
    VALID_FREQ = 'validFreq'
    LRATE = 'lrate'
    BATCH_SIZE = 'batch_size'
    VALID_BATCH_SIZE = 'valid_batch_size'


    OUTPUT_TYPE = 'output_type'  # 2D: OUTPUT IS VECTOR, 3D: MAXTRIX
    SAVE_TO = 'save_to'
    SAVE_TOPOLOGY = 'save_topology'

    IS_SEQUENCE_WORK = 'is_sequence_work'

    MAX_LEN_OUT = 'max_len_out'
    HIDDEN_DIM = 'hidden_dim'

    DEP = 'dep'

    LOSS = 'loss'

    OPTIMIZER = 'optimizer'

    SAVE_BEST_VALID_TO = 'save_best_valid_to'

    def __init__(self, for_layer=False):
        dict.__init__(self)

        self[Option.INPUT_DIM] = None
        self[Option.INPUT_LENGTH] = None
        self[Option.OUTPUT_DIM] = None
        self[Option.OUTPUT_LENGTH] = None

        if not for_layer:
            self[Option.MAX_EPOCHS] = 300
            self[Option.PATIENCE] = 10000
            self[Option.SAVE_FREQ] = 10
            self[Option.LRATE] = 0.1

            self[Option.BATCH_SIZE] = 100
            self[Option.SAVE_TO] = "./params.pkl"
            self[Option.SAVE_BEST_VALID_TO] = "./params_best.pkl"
            self[Option.VALID_BATCH_SIZE] = 100
            self[Option.VALID_FREQ] = 10
            self[Option.SAVE_TOPOLOGY] = "./topology.pkl"
            self[Option.IS_SEQUENCE_WORK] = False
            self[Option.HIDDEN_DIM] = 32
            self[Option.DEP] = 2
            self[Option.LOSS] = 'nl'
            self[Option.OPTIMIZER] = 'ada'



