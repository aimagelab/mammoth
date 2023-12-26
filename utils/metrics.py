# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def backward_transfer(results):
    """
    Calculates the backward transfer metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.

    Returns:
        float: The mean backward transfer value.
    """
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    """
    Calculates the forward transfer metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.
        random_results (list): A list of results from a random baseline.

    Returns:
        float: The mean forward transfer value.
    """
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    """
    Calculates the forgetting metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.

    Returns:
        float: The mean forgetting value.
    """
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)
