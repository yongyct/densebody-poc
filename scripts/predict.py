import os
import sys
import argparse
import logging
from tqdm import tqdm

import torch

from densebody_poc.utils import config_util, validation_util, model_util
from densebody_poc.datasets.densebody_dataset import DenseBodyDataset
from densebody_poc.datasets.visualizer import Visualizer
from densebody_poc.exceptions.model_error import ModelNotFoundError
from densebody_poc.exceptions.conf_error import InvalidJsonConfigError

from densebody_poc.utils.constants import JOB_CONF_KEY, BATCH_SIZE_KEY, MAX_DATASET_SIZE_KEY


# Logging configs
logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

# Program args
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
program_args = parser.parse_args()

logging.info('Json config provided: {}'.format(program_args.filename))


def handle_error(e):
    '''
    Error handling during the prediction dataflow
    '''
    logging.error(str(e) + '\n...Exiting program...')
    sys.exit(0)


if __name__ == '__main__':
    
    conf = config_util.get_json_conf(program_args.filename)
    # TODO: Add validation -> validation_util.validate_user_conf(conf)
    try:
        validation_util.validate_user_conf(conf)
    except InvalidJsonConfigError as e:
        handle_error(e)

    try:
        dataset = DenseBodyDataset(conf)
    except FileNotFoundError as e:
        handle_error(e)

    batches_per_epoch = max(len(dataset) // conf[JOB_CONF_KEY][BATCH_SIZE_KEY], 1)
    logging.info('Predicting {} images'.format(len(dataset)))
    
    # TODO: implement model objects
    # model = create_model()
    try:
        model = model_util.get_model(conf)
        model.initialize(conf)
        model.setup(conf)
    except ModelNotFoundError as e:
        handle_error(e)
        
    # TODO: implement visualizer
    visualizer = Visualizer(conf)
    
    with torch.no_grad():
        dataset_iter = tqdm(range(len(dataset)), ncols=80)
        for i in dataset_iter:
            dataset_iter.set_description('Predicting case {}'.format(i))
            try:
                data = dataset[i]
            except TypeError as e:
                logging.error('Issue processing data {} in dataset\n'.format(i) + str(e))
                continue
            
            # TODO implement setting of input data to model, and passing data into encoder/decoder
            model.set_input(data)
            model.fake_UV = model.decoder(model.encoder(model.real_input))

            # TODO: Implement saving of results
            # visualizer.save_results(model.get_current_visuals(), opt.load_epoch, i)
            
            print('Processed data {}'.format(i))
