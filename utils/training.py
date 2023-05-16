# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

from utils.wandbsc import WandbLogger
from utils import metrics
from tqdm import tqdm


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    valid_metrics_list = {'jaccard_sim': [], 'modified_jaccard': [], 'strict_acc': [], 'recall': []}
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    cpt = dataset.N_CLASSES // dataset.N_TASKS
    num_seen_classes = len(dataset.test_loaders) * cpt

    for k, test_loader in enumerate(tqdm(dataset.test_loaders)):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for i, data in enumerate(test_loader):
            if model.args.debug_mode == 1 and i > 3:
                break
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                if dataset.SETTING == 'multi-label':
                    predictions = outputs > 0.0
                    labels = labels[:, :num_seen_classes]
                    labels = labels.bool()
                    valid_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * inputs.shape[0]
                    valid_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * inputs.shape[0]
                    valid_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * inputs.shape[0]
                    valid_metrics['recall'] += metrics.recall(predictions, labels) * inputs.shape[0]
                    data_len += inputs.shape[0]
                else:
                    _, pred = torch.max(outputs.data, 1)
                    correct += torch.sum(pred == labels).item()
                    total += labels.shape[0]
                    if dataset.SETTING == 'class-il':
                        mask_classes(outputs, dataset, k)
                        _, pred = torch.max(outputs.data, 1)
                        correct_mask_classes += torch.sum(pred == labels).item()

        if dataset.SETTING == 'multi-label':
            valid_metrics['jaccard_sim'] /= data_len
            valid_metrics['modified_jaccard'] /= data_len
            valid_metrics['strict_acc'] /= data_len
            valid_metrics['recall'] /= data_len
            for k,v in valid_metrics.items():
                valid_metrics_list[k].append(v)
        else:
            accs.append(correct / total * 100
                        if 'class-il' in model.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)

    if dataset.SETTING == 'multi-label':
        return valid_metrics_list
    else:
        return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    run_name = args.wandb_name
    if hasattr(model, 'get_name') and run_name is None:
        run_name = model.get_name()
    wandb_logger = WandbLogger(args, prj=args.wandb_project, entity=args.wandb_entity, name=run_name)

    model.to(model.device)
    results, results_mask_classes = [], []
    multi_label_results = {
        'jaccard_sim': [],
        'modified_jaccard': [],
        'strict_acc': [],
        'recall': []
    }

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        model.train()
        for t in range(dataset.N_TASKS):
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if 'joint' in args.model:
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if isinstance(not_aug_inputs, torch.Tensor):
                        not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if ((args.model != 'joint' and args.model != 'joint_webvision') or t == dataset.N_TASKS - 1):

            accs = evaluate(model, dataset)
            if dataset.SETTING == 'multi-label':
                for k, v in accs.items():
                    multi_label_results[k].append(v)
                mean_results = {k: np.mean(v) for k, v in accs.items()}
                print_multi_label_results(mean_results, t + 1)
                if not args.disable_log:
                    logger.log_multilabel(mean_results)
                    logger.log_full_multilabel(accs)
                if not args.nowand:
                    d2={
                        **{f'RESULT_mean_{k}': v for k, v in mean_results.items()},
                    }
                    for k,v in accs.items():
                        for i, v2 in enumerate(v):
                            d2[f'RESULT_{k}_{i}'] = v2
                    wandb_logger(d2)

            else:
                results.append(accs[0])
                results_mask_classes.append(accs[1])
                mean_acc = np.mean(accs, axis=1)
                print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
                if not args.disable_log:
                    logger.log(mean_acc)
                    logger.log_fullacc(accs)
                if not args.nowand:
                    d2={
                        'RESULT/class_mean_accs': mean_acc[0],
                        'RESULT/task_mean_accs': mean_acc[1],
                        **{f'RESULT/class_acc_{i}': a for i, a in enumerate(accs[0])},
                        **{f'RESULT/task_acc_{i}': a for i, a in enumerate(accs[1])},
                        'RESULT/task': t,
                    }
                    wandb_logger(d2)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            # if dataset.SETTING == 'multi-label':
            #     d = logger.dump_multilabel()
            # else:
            d = logger.dump()
            d['wandb_url'] = wandb_logger.wandb_url
            wandb_logger(d)

    wandb_logger.finish()
