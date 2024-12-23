from argparse import Namespace
import math
from typing import Union
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel


def get_scheduler(model: ContinualModel, args: Namespace, reload_optim=True) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Returns the scheduler to be used for the current dataset.
    If `reload_optim` is True, the optimizer is reloaded from the model. This should be done at least ONCE every task
    to ensure that the learning rate is reset to the initial value.
    """
    if args.lr_scheduler is not None:
        if reload_optim or not hasattr(model, 'opt'):
            model.opt = model.get_optimizer()
        # check if lr_scheduler is in torch.optim.lr_scheduler
        supported_scheds = [sched_name.lower() for sched_name in ContinualDataset.AVAIL_SCHEDS]
        sched = None
        if args.lr_scheduler.lower() in supported_scheds:
            if args.lr_scheduler.lower() == 'multisteplr':
                assert args.lr_milestones is not None, 'MultiStepLR requires `--lr_milestones`'
                sched = MultiStepLR(model.opt, milestones=args.lr_milestones, gamma=args.sched_multistep_lr_gamma)
            elif args.lr_scheduler.lower() == 'cosine':
                sched = CosineAnnealingLR(model.opt, T_max=args.n_epochs, verbose=True)

        if sched is None:
            raise ValueError('Unknown scheduler: {}'.format(args.lr_scheduler))
        return sched
    return None


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        """
        Apply cosine learning rate schedule to all the parameters in the optimizer.
        """
        assert K > 1, "K must be greater than 1"
        self.K = K
        super().__init__(optimizer)

    def cosine(self, base_lr):
        if self.last_epoch == 0:
            return base_lr
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


class CosineSchedulerWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer: Optimizer, base_lrs: Union[list, float], warmup_length: int, steps: int):
        """
        Apply cosine learning rate schedule with warmup to all the parameters in the optimizer.
        If more than one param_group is passed, the learning rate must either be a list of the same length or a float.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to which the learning rate will be applied.
            base_lrs (list | float): Initial learning rate.
            warmup_length (int): Number of warmup steps. The learning rate will linearly increase from 0 to base_lr during this period.
            steps (int): Total number of steps.
        """
        self.warmup_length = warmup_length
        self.steps = steps
        self.base_lrs = base_lrs
        super().__init__(optimizer)

        if not isinstance(base_lrs, list):
            base_lrs = [base_lrs for _ in optimizer.param_groups]
        assert len(base_lrs) == len(optimizer.param_groups)

    def get_lr(self):
        ret_lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_length:
                lr = base_lr * (self.last_epoch + 1) / self.warmup_length
            else:
                e = self.last_epoch - self.warmup_length
                es = self.steps - self.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            ret_lrs.append(lr)
        return ret_lrs
