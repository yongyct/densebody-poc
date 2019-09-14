import os
import sys
import argparse
import logging

from densebody_poc.utils import config_util, validation_util
from densebody_poc.utils.densebody_dataset import DenseBodyDataset

# Logging configs
logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

# Program args
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
opts = parser.parse_args()

logging.info('Json config provided: {}'.format(opts.filename))


if __name__ == '__main__':
    
    conf = config_util.get_json_conf(opts.filename)
    # TODO: Add validation -> validation_util.validate_user_conf(conf)
    validation_util.validate_user_conf(conf)
    dataset = DenseBodyDataset(conf)
