# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from datetime import datetime
from time import time
from typing import Union
import shutil


def padded_print(string: str, max_width: int, **kwargs) -> None:
    """
    Prints a string with blank spaces to reach the max_width.

    Args:
        string: the string to print
        max_width: the maximum width of the string
    """
    pad_len = max(0, max_width - len(string))
    print(string + ' ' * pad_len, **kwargs)


class ProgressBar:
    def __init__(self, joint=False, verbose=True, update_every=1):
        """
        Initializes a ProgressBar object.

        Args:
            joint: a boolean indicating whether the progress bar is for a joint task
            verbose: a boolean indicating whether to display the progress bar
            update_every: the number of iterations after which the progress bar is updated
        """
        self.joint = joint
        self.update_every = update_every
        self.verbose = verbose
        self.old_time = None

        self.reset()

        assert self.update_every > 0

    def reset(self) -> None:
        """
        Resets the progress bar.
        """
        if self.old_time is not None:
            max_width = shutil.get_terminal_size((80, 20)).columns
            padded_print(f'\n\t- Took: {round(self.running_sum, 2)} s', max_width=max_width, file=sys.stderr, flush=True)

        self.old_time = time()
        self.running_sum = 0
        self.current_task_iter = 0

    def prog(self, current_epoch_iter: int, max_epoch_iter: int, epoch: Union[int, str],
             task_number: int, loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.

        Args:
            current_epoch_iter: the current iteration of the epoch
            max_epoch_iter: the maximum number of iteration for the task. If None, the progress bar is not printed.
            epoch: the epoch
            task_number: the task index
            loss: the current value of the loss function
        """
        max_width = shutil.get_terminal_size((80, 20)).columns
        if not self.verbose:
            if current_epoch_iter == 0:
                if self.joint:
                    padded_print('[ {} ] Joint | epoch {}\n'.format(
                        datetime.now().strftime("%m-%d | %H:%M"),
                        epoch
                    ), max_width=max_width, file=sys.stderr, end='', flush=True)
                else:
                    padded_print('[ {} ] Task {} | epoch {}\n'.format(
                        datetime.now().strftime("%m-%d | %H:%M"),
                        task_number + 1 if isinstance(task_number, int) else task_number,
                        epoch
                    ), max_width=max_width, file=sys.stderr, end='', flush=True)
            else:
                return

        timediff = time() - self.old_time
        self.running_sum += timediff + 1e-8

        # Print the progress bar every update_every iterations
        if (current_epoch_iter and current_epoch_iter % self.update_every == 0) or (max_epoch_iter is not None and current_epoch_iter == max_epoch_iter - 1):
            progress = min(float((current_epoch_iter + 1) / max_epoch_iter), 1) if max_epoch_iter else 0
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress))) if max_epoch_iter else '~N/A~'
            if self.joint:
                padded_print('\r[ {} ] Joint | epoch {} | iter {}: |{}| {} ep/h | loss: {} | Time: {} ms/it'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    epoch,
                    self.current_task_iter + 1,
                    progress_bar,
                    round(3600 / (max_epoch_iter * timediff), 2) if max_epoch_iter else 'N/A',
                    round(loss, 8),
                    round(1000 * timediff / self.update_every, 2)
                ), max_width=max_width, file=sys.stderr, end='', flush=True)
            else:
                padded_print('\r[ {} ] Task {} | epoch {} | iter {}: |{}| {} ep/h | loss: {} | Time: {} ms/it'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    task_number + 1 if isinstance(task_number, int) else task_number,
                    epoch,
                    self.current_task_iter + 1,
                    progress_bar,
                    round(3600 / (max_epoch_iter * timediff), 2) if max_epoch_iter else 'N/A',
                    round(loss, 8),
                    round(1000 * timediff / self.update_every, 2)
                ), max_width=max_width, file=sys.stderr, end='', flush=True)

        self.current_task_iter += 1
        self.old_time = time()


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
