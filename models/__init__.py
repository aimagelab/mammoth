# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from argparse import Namespace
from typing import Dict, List, Callable
from torch import nn
import importlib
import inspect

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)

from models.utils.continual_model import ContinualModel
from utils import register_dynamic_module_fn
from utils.conf import warn_once

REGISTERED_MODELS = dict()  # dictionary containing the registered models. Template: {name: {'class': class, 'parsable_args': parsable_args}}

def register_model(name: str) -> Callable:
    """
    Decorator to register a ContinualModel. The decorator should be used on a class that inherits from `ContinualModel`.
    The registered model can be accessed using the `get_model` function and can include additional keyword arguments to be set during parsing.

    Differently from the `register_dataset` and `register_backbone` functions, this decorator does not infer the arguments from the signature of the class.
    Instead, to define model-specific arguments, you should define the `get_parser` function in the model class, which should return a parser with the additional arguments.

    Args:
        name: the name of the model
    """
    return register_dynamic_module_fn(name, REGISTERED_MODELS, ContinualModel)


def get_all_models_legacy() -> List[dict]:
    """
    Returns the list of all the available models in the models folder that follow the model naming convention (see :ref:`models`).
    """
    return {model.split('.')[0].replace('_', '-'): model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and not os.path.isdir('models/' + model)}


def get_model(args: Namespace, backbone: nn.Module, loss, transform, dataset) -> ContinualModel:
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--model` attribute
        backbone (nn.Module): the backbone of the model
        loss: the loss function
        transform: the transform function
        dataset: the instance of the dataset

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the continual model instance
    """
    model_name = args.model.replace('_', '-')
    names = get_model_names()
    assert model_name in names
    return get_model_class(args)(backbone, loss, args, transform, dataset)


def get_model_class(args: Namespace) -> ContinualModel:
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--model` attribute

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the continual model class
    """
    names = get_model_names()
    model_name = args.model.replace('_', '-')
    assert model_name in names
    if isinstance(names[model_name], Exception):
        raise names[model_name]
    return names[model_name]


def get_model_names() -> Dict[str, ContinualModel]:
    """
    Return the available continual model names and classes.

    Returns:
        A dictionary containing the names of the available continual models and their classes.
    """
    names = {}  # key: model name, value: model class
    for model, model_conf in REGISTERED_MODELS.items():
        # names[model.replace('_', '-')] = model_conf['class']
        names[model.replace('_', '-')] = model_conf['class']

    for model in get_all_models_legacy():  # for the models that follow the old naming convention, load the model class and check for errors
        if model in names:  # model registered with the new convention has priority
            continue

        try:
            mod = importlib.import_module('models.' + model.replace('-', '_'))
            model_classes_name = [x for x in mod.__dir__() if inspect.isclass(getattr(mod, x))
                                    and 'ContinualModel' in str(inspect.getmro(getattr(mod, x))[1:])
                                    and not inspect.isabstract(getattr(mod, x))]
            for d in model_classes_name:
                c = getattr(mod, d)
                names[c.NAME.replace('_', '-')] = c

        except Exception as e:  # if an error is detected, raise the appropriate error message
            warn_once(f'Error in model {model}')
            warn_once(e)
            names[model.replace('_', '-')] = e
    return names
