import logging

from densebody_poc.utils import model_util
from densebody_poc.exceptions.conf_error import InvalidJsonConfigError

from densebody_poc.utils.constants import MODEL_CONF_KEY, MODEL_KEY


def validate_user_conf(conf):
    logging.info('Validating user config...')
    logging.info(conf)
    # TODO: Add validation logic
    
    # Mandatory config checks
    
    # Validate availability of model selected
    all_model_classes = model_util.get_all_model_classes()
    all_model_class_names = list(map(lambda x: x.__name__.lower(), all_model_classes))
    conf_model_class_name = conf[MODEL_CONF_KEY][MODEL_KEY].lower()
    if conf_model_class_name not in all_model_class_names:
        raise InvalidJsonConfigError('Selected model {} not available, available models are: {}'\
             .format(conf_model_class_name, all_model_class_names))
