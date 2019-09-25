import logging
import torch

from densebody_poc.utils import model_util
from densebody_poc.exceptions.conf_error import InvalidJsonConfigError

from densebody_poc.utils.constants import PREDICT, TRAIN, TEST, CPU, CUDA


def validate_user_conf(conf):
    logging.info('Validating user config...')
    logging.info(conf)
    # TODO: Add validation logic

    # Mandatory config checks
    ## TODO: Add mandatory fields

    # Validate phase type
    valid_phases = [PREDICT, TRAIN, TEST]
    if conf.PHASE not in valid_phases:
        raise InvalidJsonConfigError('Invalid phase specified, available phases are: {}'
                                     .format(valid_phases))

    # Validate device type
    valid_devices = [CPU, CUDA]
    if conf.DEVICE not in valid_devices:
        raise InvalidJsonConfigError('Invalid device specified, available devices are: {}'
                                     .format(valid_devices))
    if not torch.cuda.is_available() and conf.DEVICE == CUDA:
        raise InvalidJsonConfigError('Invalid device specified, cuda selected is not available')

    # Validate availability of model selected
    all_model_classes = model_util.get_all_model_classes()
    all_model_class_names = list(map(lambda x: x.__name__.lower(), all_model_classes))
    conf_model_class_name = conf.MODEL.lower()
    if conf_model_class_name not in all_model_class_names:
        raise InvalidJsonConfigError('Selected model {} not available, available models are: {}'
                                     .format(conf_model_class_name, all_model_class_names))
