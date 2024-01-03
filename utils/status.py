# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from datetime import datetime
from time import time
from typing import Union


class ProgressBar:
    def __init__(self, joint=False, verbose=True):
        """
        Initializes a ProgressBar object.

        Args:
            joint: a boolean indicating whether the progress bar is for a joint task
            verbose: a boolean indicating whether to display the progress bar
        """
        self.joint = joint
        self.old_time = 0
        self.running_sum = 0
        self.verbose = verbose

    def prog(self, i: int, max_iter: int, epoch: Union[int, str],
             task_number: int, loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.

        Args:
            i: the current iteration
            max_iter: the maximum number of iteration. If None, the progress bar is not printed.
            epoch: the epoch
            task_number: the task index
            loss: the current value of the loss function
        """
        if not self.verbose:
            if i == 0:
                if self.joint:
                    print('[ {} ] Joint | epoch {}\n'.format(
                        datetime.now().strftime("%m-%d | %H:%M"),
                        epoch
                    ), file=sys.stderr, end='', flush=True)
                else:
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
            self.running_sum = self.running_sum + (time() - self.old_time) + 1e-8
            self.old_time = time()
        if i:  # not (i + 1) % 10 or (i + 1) == max_iter:
            progress = min(float((i + 1) / max_iter), 1) if max_iter else 0
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress))) if max_iter else '~N/A~'
            if self.joint:
                print('\r[ {} ] Joint | epoch {}: |{}| {} ep/h | loss: {} |'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    epoch,
                    progress_bar,
                    round(3600 / (self.running_sum / i * max_iter), 2) if max_iter else 'N/A',
                    round(loss, 8)
                ), file=sys.stderr, end='', flush=True)
            else:
                print('\r[ {} ] Task {} | epoch {}: |{}| {} ep/h | loss: {} |'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    task_number + 1 if isinstance(task_number, int) else task_number,
                    epoch,
                    progress_bar,
                    round(3600 / (self.running_sum / i * max_iter), 2) if max_iter else 'N/A',
                    round(loss, 8)
                ), file=sys.stderr, end='', flush=True)


def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 task_number: int, loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.

    Args:
        i: the current iteration
        max_iter: the maximum number of iteration
        epoch: the epoch
        task_number: the task index
        loss: the current value of the loss function
    """
    global static_bar

    if i == 0:
        static_bar = ProgressBar()
    static_bar.prog(i, max_iter, epoch, task_number, loss)
