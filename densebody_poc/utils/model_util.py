import sys
import inspect
import importlib
import logging

import densebody_poc.models
from densebody_poc.models.base_model import BaseModel
from densebody_poc.exceptions.model_error import ModelNotFoundError

from densebody_poc.utils.constants import MODEL_CONF_KEY, MODEL_KEY


def get_all_model_classes():
    '''
    Retrieves the name of all model classes available for use.
    Can be used in validation of json config
    '''
    model_classes = []
    model_modules = inspect.getmembers(
        sys.modules[densebody_poc.models.__name__], 
        inspect.ismodule
    )
    for _, module_obj in model_modules:
        module_classes = inspect.getmembers(sys.modules[module_obj.__name__], inspect.isclass)
        for _, module_class_obj in module_classes:
            if issubclass(module_class_obj, BaseModel) and module_class_obj != BaseModel:
                model_classes.append(module_class_obj)
    return model_classes


def get_model(conf):
    '''
    Retrieves model class based on name provided in json config
    '''
    ### TODO: This part should be added to validation utils
    all_model_classes = get_all_model_classes()
    all_model_class_names = list(map(lambda x: x.__name__.lower(), all_model_classes))
    conf_model_class_name = conf[MODEL_CONF_KEY][MODEL_KEY].lower()
    if conf_model_class_name not in all_model_class_names:
        raise ModelNotFoundError('Selected model {} not available, available models are: {}'\
             .format(conf_model_class_name, all_model_class_names))

    for model_class_name, model_class_obj in zip(all_model_class_names, all_model_classes):
        if conf_model_class_name == model_class_name:
            return model_class_obj()
