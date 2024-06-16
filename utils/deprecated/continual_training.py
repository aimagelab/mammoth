# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from argparse import Namespace

import torch
from datasets import get_dataset
from models import get_model
from models.utils.continual_model import ContinualModel
from utils.loggers import Logger

from utils.stats import track_system_stats
from utils.status import progress_bar

try:
    import wandb
except ImportError:
    wandb = None


def evaluate(model: ContinualModel, dataset) -> float:
    """
    Evaluates the final accuracy of the model.

    Args:
        model: the model to be evaluated
        dataset: the GCL dataset at hand

    Returns:
        a float value that indicates the accuracy
    """
    model.net.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.

    Args:
        args: the arguments loaded from the command line
    """

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:
        epoch, i = 0, 0
        model.net.train()

        while not dataset.train_over:
            inputs, labels, not_aug_inputs = dataset.get_train_data()
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            loss = model.observe(inputs, labels, not_aug_inputs)
            progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)
            system_tracker()
            i += 1

        if model.NAME == 'joint_gcl':
            model.end_task(dataset)

        acc = evaluate(model, dataset)
        print('Accuracy:', acc)

    if not args.disable_log:
        logger.log(acc)
        logger.write(vars(args))

    if not args.nowand:
        wandb.log({'Accuracy': acc})
        wandb.finish()
