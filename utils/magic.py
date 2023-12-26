# persistent_locals has been co-authored with Andrea Maffezzoli

import sys
from typing import Callable


class persistent_locals:
    """
    A decorator class that allows access to the local variables of a function
    after it has been called.

    Usage:
    @persistent_locals
    def my_function():
        ...

    my_function()
    print(my_function.locals)  # Access the local variables of my_function
    """

    def __init__(self, func: Callable):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == 'return':
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        """
        Clears the stored local variables.
        """
        self._locals = {}

    @property
    def locals(self):
        """
        Returns the stored local variables.
        """
        return self._locals
