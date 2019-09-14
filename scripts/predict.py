import os
import sys
import argparse
import logging

from densebody_poc.utils import config_util, validation_util
from densebody_poc.utils.densebody_dataset import DenseBodyDataset
from densebody_poc.utils.constants import JOB_CONF_KEY, BATCH_SIZE_KEY

# Logging configs
logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

# Program args
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
program_args = parser.parse_args()

logging.info('Json config provided: {}'.format(program_args.filename))


if __name__ == '__main__':
    
    conf = config_util.get_json_conf(program_args.filename)
    # TODO: Add validation -> validation_util.validate_user_conf(conf)
    validation_util.validate_user_conf(conf)
    dataset = DenseBodyDataset(conf)
    batches_per_epoch = max(len(dataset) // conf[JOB_CONF_KEY][BATCH_SIZE_KEY], 1)
    logging.info('Predicting {} images'.format(len(dataset)))
    
    # TODO: implement model objects
    # model = create_model()
