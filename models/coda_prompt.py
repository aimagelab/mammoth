import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from utils import none_or_float
from datasets import get_dataset
from models.coda_prompt_utils.zoo_ours import ResNetZoo
from utils.schedulers import CosineSchedule
import re
import sys
import os

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--mu', type=float, default=0.1, help='weight of prompt loss')
    parser.add_argument('--prompt_flag', type=str, default='dual', choices=['dual', 'reservoir', 'reservoir_new', 'specific'], help='type of prompt')
    parser.add_argument('--pool_size_coda', type=int, default=100, help='pool size')
    return parser


class CodaPrompt(ContinualModel):
    NAME = 'coda_prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.n_tasks = self.dataset.N_TASKS
        backbone = ResNetZoo(num_classes=self.n_classes, pt=True, mode=0, prompt_flag=args.prompt_flag, prompt_param=[self.n_tasks, [args.pool_size_coda, 8, 0, 0, 0]])
        super().__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.net.task_id = 0
        self.opt = self.get_optimizer()

    def get_optimizer(self):
        params_to_opt = list(self.net.prompt.parameters()) + list(self.net.last.parameters())
        optimizer_arg = {'params':params_to_opt,
                        'lr':self.args.lr,
                        'weight_decay':self.args.optim_wd}
        if self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(**optimizer_arg)
        elif self.args.optimizer == 'adam':
            opt = torch.optim.Adam(**optimizer_arg)
        else:
            raise ValueError('Optimizer not supported for this method')
        return opt

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task != 0:
            self.net.task_id = self.current_task
            self.net.prompt.process_frequency()
            self.opt = self.get_optimizer()

        self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
        self.old_epoch = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.scheduler and self.old_epoch != epoch:
            self.scheduler.step()
            self.old_epoch = epoch
        labels = labels.long()
        self.opt.zero_grad()
        logits, loss_prompt = self.net(inputs, train=True)
        loss_prompt = loss_prompt.sum()
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        logits = logits[:, :offset_2]
        labels = labels[:, :offset_2]
        loss_bce = self.loss(logits, labels.float())
        loss = loss_bce + self.args.mu * loss_prompt
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task - 1)
        logits, _ = self.net(x)
        return logits[:, :offset_2]

