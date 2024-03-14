# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import string
import random


def custom_str_underscore(value):
    return str(value).replace("_", '-').strip()


def smart_joint(*paths):
    return os.path.join(*paths).replace("\\", "/")


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.

    Args:
        path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


def random_id(length=8, alphabet=string.ascii_letters + string.digits):
    """
    Returns a random string of the specified length.

    Args:
        length: the length of the string
        alphabet: the alphabet to be used

    Returns:
        the random string
    """
    return ''.join(random.choices(alphabet, k=length))
