# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


# multi-label metrics

def jaccard_sim(predictions, target):
    '''
    Jaccard similarity (intersection over union)
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    jac_sim = torch.sum(target & predictions, dim=-1) * 1.0 / torch.sum(target | predictions, dim=-1)
    jac_sim = torch.sum(jac_sim)
    jac_sim /= target.shape[0]
    return jac_sim.item()


def modified_jaccard_sim(predictions, target):
    '''
    Jaccard similarity multiplied by percentage of correct predictions (per sample precision)
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    jac_sim = torch.sum(target & predictions, dim=-1) * 1.0 / torch.sum(target | predictions, dim=-1)
    correct_pred_pct = torch.sum(target & predictions, dim=-1) * 1.0 / (torch.sum(predictions, dim=-1) + 1e-8)
    modified_jac_sim = jac_sim * correct_pred_pct
    modified_jac_sim = torch.sum(modified_jac_sim)
    modified_jac_sim /= target.shape[0]
    return modified_jac_sim.item()


def strict_accuracy(predictions, target):
    '''
    The accuracy measure where if some of the labels for a sample are predicted correctly, and some are wrong, it gives
    a score of zero to the accuracy of that sample
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    acc = torch.sum((target == predictions).all(dim=-1)) * 1.0
    acc /= target.shape[0]
    return acc.item()


def recall(predictions, target):
    '''
    The recall measure
    Args:
        predictions (torch.BoolTensor): Model predictions of size batch_size x number of classes
        target (torch.BoolTensor): target predictions of size batch_size x number of classes
    '''
    recall = torch.sum(predictions[target]) * 1.0 / torch.sum(target)
    return recall.item()
