import os
import decorator


def init_test_environ(func):
    def wrapper(func, *args, **kwargs):
        os.environ['MAMMOTH_TEST'] = '1'
        return func(*args, **kwargs)
    return decorator.decorator(wrapper, func)
