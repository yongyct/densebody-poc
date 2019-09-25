import argparse
import logging
import json

from densebody_poc.utils.constants import JOB_CONF_KEY, MODEL_CONF_KEY, OPS_CONF_KEY, \
    DATA_DIR_KEY, BATCH_SIZE_KEY, MODEL_KEY, MAX_DATASET_SIZE_KEY, PHASE_KEY, \
    CHECKPOINT_DIR_KEY, NAME_KEY, CONTINUE_TRAIN_KEY, LOAD_EPOCH_KEY, VERBOSE_KEY, \
    IM_SIZE_KEY, NZ_KEY, NCHANNELS_KEY, NET_E_KEY, NET_D_KEY, NDOWN_KEY, NUP_KEY, \
    NORM_KEY, NL_E_KEY, NL_D_KEY, INIT_TYPE_KEY, DEVICE_KEY, UV_MAP_KEY, RESULTS_DIR_KEY


def get_user_conf():
    """
    Sets logging configurations + retrieve config from user provided json
    :return: UserConfig object containing user config
    """
    # Logging configs
    logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

    # Program args
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
    program_args = parser.parse_args()

    logging.info('Json config provided: {}'.format(program_args.filename))

    return UserConfig(get_json_conf(program_args.filename))


def get_json_conf(filename):
    """
    Parses json information in the filename, and return it as a dictionary
    :param filename: absolute filename of user json config file
    :return: dictionary from json configuration
    """
    with open(filename) as json_file:
        conf_dict = json.load(json_file)
    return conf_dict


class UserConfig:
    """
    Object to hold configuration values
    """
    def __init__(self, conf):
        
        job_conf = conf[JOB_CONF_KEY]
        model_conf = conf[MODEL_CONF_KEY]
        ops_conf = conf[OPS_CONF_KEY]

        self.NAME = job_conf[NAME_KEY]
        self.DATA_DIR = job_conf[DATA_DIR_KEY]
        self.BATCH_SIZE = job_conf[BATCH_SIZE_KEY]
        self.CHECKPOINT_DIR = job_conf[CHECKPOINT_DIR_KEY]
        self.PHASE = job_conf[PHASE_KEY]
        self.MAX_DATASET_SIZE = job_conf[MAX_DATASET_SIZE_KEY]
        self.DEVICE = job_conf[DEVICE_KEY]
        self.UV_MAP = job_conf[UV_MAP_KEY]

        self.MODEL = model_conf[MODEL_KEY]
        self.IM_SIZE = model_conf[IM_SIZE_KEY]
        self.NZ = model_conf[NZ_KEY]
        self.NCHANNELS = model_conf[NCHANNELS_KEY]
        self.NET_E = model_conf[NET_E_KEY]
        self.NET_D = model_conf[NET_D_KEY]
        self.NDOWN = model_conf[NDOWN_KEY]
        self.NUP = model_conf[NUP_KEY]
        self.NORM = model_conf[NORM_KEY]
        self.NL_E = model_conf[NL_E_KEY]
        self.NL_D = model_conf[NL_D_KEY]
        self.INIT_TYPE = model_conf[INIT_TYPE_KEY]

        self.CONTINUE_TRAIN = ops_conf[CONTINUE_TRAIN_KEY]
        self.LOAD_EPOCH = ops_conf[LOAD_EPOCH_KEY]
        self.VERBOSE = ops_conf[VERBOSE_KEY]
        self.RESULTS_DIR = ops_conf[RESULTS_DIR_KEY]
