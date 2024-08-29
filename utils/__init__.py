import os
import sys
import string
import random
import logging


def setup_logging():
    """
    Configures the logging module.
    """

    # check if logging has already been configured
    if hasattr(setup_logging, 'done'):
        return
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] (%(name)s) %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    setattr(setup_logging, 'done', True)


def field_with_aliases(choices: dict) -> str:
    """
    Build a data type where for each key in `choices` there are a set of aliases.

    Example:
        Given the following dictionary:
        ```
        choices = {
            'a': ['a', 'alpha'],
            'b': ['b', 'beta']
        }
        ```
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
