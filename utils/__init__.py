from argparse import Namespace
from collections.abc import Iterable
import inspect
import os
import sys
import string
import random
import logging
from typing import Callable, Dict, Type, TypeVar, Union, get_args, get_origin, Literal
import torch
import numpy as np
T = TypeVar("T")

def in_notebook():
    # implementation from tqdm autonotebook
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False # running in console mode
        # running in notebook mode
        return True
    except Exception:
        return False

def check_fn_dynamic_type(fn: T, tp: Type[T], strict=True) -> bool:
    """
    Controls if the function respects the type `tp`.
    The function must have the same number of arguments as the type `tp` and the same type for each argument.

    Args:
        fn: the function to be checked
        tp: the type to be respected
        strict: if True, raises an error if the function does not respect the type `tp`
    """
    type_args = [str(arg).split("'")[1].split("'")[0] for arg in get_args(tp)[0]]
    fn_args = [v._annotation if v._annotation != inspect._empty else str(type(v.default)).split("'")[1].split("'")[0]
               for k, v in inspect.signature(fn).parameters.items()]
    if not all([f == t for f, t in zip(fn_args, type_args)]):
        if strict:
            raise ValueError(f'{fn} does not respect type {tp}')
        return False
    return True


def setup_logging():
    """
    Configures the logging module.
    """

    # check if logging has already been configured
    if hasattr(setup_logging, 'done'):
        return
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger('root')
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)
    logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
    logger.addHandler(handler)
    setattr(setup_logging, 'done', True)


def field_with_aliases(choices: dict) -> str:
    """
    Build a data type where for each key in `choices` there are a set of aliases.

    Example:
        Given the following dictionary:

        .. code-block:: python

            choices = {
                'a': ['a', 'alpha'],
                'b': ['b', 'beta']
            }

        The values 'a' and 'alpha' will be converted to 'a', and 'b' and 'beta' will be converted to 'b'.

    Args:
        choices: the dictionary containing the aliases

    Returns:
        the data type for argparse
    """

    def _parse_field(value: str) -> str:
        if not isinstance(value, str):
            value = str(value)

        for key, aliases in choices.items():
            if value in aliases:
                return key
        raise ValueError(f'Value `{value}` does not match the provided choices `{choices}`')
    return _parse_field


def binary_to_boolean_type(value: str) -> bool:
    """
    Converts a binary string to a boolean type.

    Args:
        value: the binary string

    Returns:
        the boolean type
    """
    if not isinstance(value, str):
        value = str(value)

    value = value.lower()
    true_values = ['true', '1', 't', 'y', 'yes']
    false_values = ['false', '0', 'f', 'n', 'no']

    assert value in true_values + false_values

    return value in true_values


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


def infer_args_from_signature(signature: inspect.Signature, excluded_signature: inspect.Signature = None, ignore_args: list = None) -> dict:
    """
    Load the arguments of a function from its signature.

    Args:
        signature: the signature of the function
        excluded_signature: the signature of the function to be excluded from the arguments
        ignore_args: a list of arguments to be ignored when inferring the arguments from the signature

    This function will return a dictionary with the arguments of the function, their type, and whether they are required or not.
    If an argument has a default value, it will be included in the dictionary as well.

    Returns:
        the inferred arguments
    """
    excluded_args = [] if excluded_signature is None else list(excluded_signature.parameters.keys())
    parsable_args = {}

    if ignore_args is None:
        ignore_args = []
    else:
        print(ignore_args)
    
    excluded_args += ignore_args
    n_ignored_args = len(ignore_args)
    for i, (arg_name, value) in enumerate(signature.parameters.items()):
        if arg_name in excluded_args:
            continue
        if arg_name != 'self' and not arg_name.startswith('_') and i>=n_ignored_args: 
            default = value.default
            tp = str
            if value.annotation is not inspect._empty:
                tp = value.annotation
            elif default is not inspect.Parameter.empty:
                tp = type(default)
            
            choices = None
            if get_origin(tp) == Literal:
                choices = get_args(tp)
                tp = str

            if default is inspect.Parameter.empty and arg_name != 'num_classes':
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': True
                }
            else:
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': False,
                    'default': default if default is not inspect.Parameter.empty else None
                }
            if choices is not None:
                parsable_args[arg_name]['choices'] = choices
    return parsable_args


def register_dynamic_module_fn(name: str, register: dict, tp: Type[T], ignore_args: list = None) -> Callable[[Union[T, Callable]], T]:
    """
    Register a dynamic module in the specified dictionary.

    Args:
        name: the name of the module
        register: the dictionary where the module will be registered
        cls: the class to be registered
        tp: the type of the class, used to dynamically infer the arguments
        ignore_args: a list of arguments to be ignored when inferring the arguments from the signature
    """
    name = name.replace('_', '-').lower()

    def register_network_fn(target: Union[T, Callable]) -> T:
        # check if the name is already registered
        if name in register:
            if not in_notebook():
                raise ValueError(f"Name {name} already registered!")
            else:
                logging.warning(f"Name {name} already registered, overwriting it.")

        # check if `cls` is a subclass of `T`
        if inspect.isfunction(target):
            signature = inspect.signature(target)
        elif isinstance(target, tp) or issubclass(target, tp):
            signature = inspect.signature(target.__init__)
            if not hasattr(target, 'NAME'):
                setattr(target, 'NAME', name)  # set the name of the class
        else:
            raise ValueError(f"The registered class must be a subclass of {tp.__class__.__name__} or a function returning {tp.__class__.__name__}")

        parsable_args = infer_args_from_signature(signature, ignore_args=ignore_args)
        register[name] = {'class': target, 'parsable_args': parsable_args}
        return target

    return register_network_fn


class disable_logging:
    """
    Wrapper for disabling logging for a specific block of code.
    """

    def __init__(self, min_level=logging.CRITICAL):
        self.logger = logging.getLogger()
        self.min_level = min_level

    def __enter__(self):
        self.old_logging_level = self.logger.level
        logging.disable(self.min_level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(self.old_logging_level)


def to_parsable_obj(r: Union[Dict, Namespace, list, torch.Tensor, np.ndarray]) -> Union[Dict, list, str, int, float, bool]:
    """
    Convert a non-builtin object to a parsable (and loadable with `weights_only=True`) object.
    Looking at you, Namespace.
    """

    if isinstance(r, Namespace):
        return to_parsable_obj(vars(r))
    if isinstance(r, list):
        return [to_parsable_obj(x) for x in r]
    if isinstance(r, dict):
        return {k: to_parsable_obj(v) for k, v in r.items()}
    else:
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy().tolist()
        elif isinstance(r, np.ndarray):
            r = r.tolist()
        if not isinstance(r, str) and isinstance(r, Iterable) and len(r) > 1:
            return [to_parsable_obj(x) for x in r]
        # check if type of r is builtin
        if isinstance(r, (int, float, str, bool)):
            try:
                r = r.item()  # could be numpy scalar
            except BaseException:
                return r
        if isinstance(r, (torch.device)):
            return str(r)
        if r is not None:
            logging.warning(f"Object {r} is not parsable, returning it as str.")
            return str(r)  # return as str if not parsable
        
        return None
