# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from argparse import Namespace
from typing import Dict, List
from torch import nn
import importlib
import inspect
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)
from models.utils.continual_model import ContinualModel
from utils.conf import warn_once


def get_all_models() -> List[dict]:
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

    def _get_names():
        names: Dict[str, ContinualModel] = {}
        for model_name, model in get_all_models().items():
            try:
                mod = importlib.import_module('models.' + model)
                model_classe_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
                                     and 'ContinualModel' in str(inspect.getmro(getattr(mod, x))[1:])][-1]
                c = getattr(mod, model_classe_name)
                names[c.NAME.replace('_', '-')] = c
            except Exception as e:
                warn_once("Error in model", model)
                names[model.replace('_', '-')] = e
        return names

    if not hasattr(get_model_names, 'names'):
        setattr(get_model_names, 'names', _get_names())
    return getattr(get_model_names, 'names')
