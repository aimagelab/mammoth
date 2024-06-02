"""
This package contains utility functions used by all datasets, including the base dataset class (ContinualDataset).
"""

import functools
import inspect

DEFAULT_ARGS = {}

def is_static_call(*args):
    return len(args) == 0

def set_default_from_args(arg_name):
    global DEFAULT_ARGS
    caller = inspect.currentframe().f_back
    caller_name = caller.f_locals['NAME']
    if caller_name not in DEFAULT_ARGS:
        DEFAULT_ARGS[caller_name] = {}

    def decorator_set_default_from_args(func):
        DEFAULT_ARGS[caller_name][arg_name] = func(None)

        @functools.wraps(func)
        def wrapper(*args):
            # global DEFAULT_ARGS
            
            if is_static_call(*args):
                # if no arguments are passed, return the function
                return func(None)
            
            return func(*args)
        return wrapper
    return decorator_set_default_from_args