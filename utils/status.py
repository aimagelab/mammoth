# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from datetime import datetime
from time import time
from typing import Union


class ProgressBar:
    def __init__(self, verbose=True):
        self.old_time = 0
        self.running_sum = 0
        self.verbose = verbose

    def prog(self, i: int, max_iter: int, epoch: Union[int, str],
                     task_number: int, loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.
        :param i: the current iteration
        :param max_iter: the maximum number of iteration
        :param epoch: the epoch
        :param task_number: the task index
        :param loss: the current value of the loss function
        """
        if not self.verbose:
            if i == 0:
                print('[ {} ] Task {} | epoch {}\n'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    task_number + 1 if isinstance(task_number, int) else task_number,
                    epoch
                ), file=sys.stderr, end='', flush=True)
            else:
                return
        if i == 0:
            self.old_time = time()
            self.running_sum = 0
        else:
            self.running_sum = self.running_sum + (time() - self.old_time)
            self.old_time = time()
        if i:  # not (i + 1) % 10 or (i + 1) == max_iter:
            progress = min(float((i + 1) / max_iter), 1)
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            print('\r[ {} ] Task {} | epoch {}: |{}| {} ep/h | loss: {} |'.format(
                datetime.now().strftime("%m-%d | %H:%M"),
                task_number + 1 if isinstance(task_number, int) else task_number,
                epoch,
                progress_bar,
                round(3600 / (self.running_sum / i * max_iter), 2),
                round(loss, 8)
            ), file=sys.stderr, end='', flush=True)

def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 task_number: int, loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    global static_bar

    if i == 0:
        static_bar = ProgressBar()
    static_bar.prog(i, max_iter, epoch, task_number, loss)
