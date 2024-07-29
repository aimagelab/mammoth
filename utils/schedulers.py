import math
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        """
        Apply cosine learning rate schedule to all the parameters in the optimizer.
        """
        self.K = K
        super().__init__(optimizer)

    def cosine(self, base_lr):
        if self.last_epoch == 0:
            return base_lr
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


class CosineSchedulerWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer: Optimizer, base_lrs: list | float, warmup_length: int, steps: int):
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
