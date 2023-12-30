# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from argparse import Namespace
from typing import List
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


NAMES = {}
for model_name, model in get_all_models().items():
    try:
        mod = importlib.import_module('models.' + model)
        model_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
                                and 'ContinualModel' in str(inspect.getmro(getattr(mod, x))[1:])]
        for d in model_classes_name:
            c = getattr(mod, d)
            NAMES[c.NAME.replace('_', '-')] = c
    except Exception as e:
        warn_once("Error in model", model)
        warn_once(e)
        NAMES[model.replace('_', '-')] = e


def get_model(args: Namespace, backbone: nn.Module, loss, transform) -> ContinualModel:
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--model` attribute
        backbone (nn.Module): the backbone of the model
        loss: the loss function
        transform: the transform function

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the continual model instance
    """
    model_name = args.model.replace('_', '-')
    assert model_name in NAMES
    return get_model_class(args)(backbone, loss, args, transform)

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
    model_name = args.model.replace('_', '-')
    assert model_name in NAMES
    if isinstance(NAMES[model_name], Exception):
        raise NAMES[model_name]
    return NAMES[model_name]
