"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from datasets import get_dataset
from models.coda_prompt_utils.model import Model
from utils.schedulers import CosineSchedule


class CodaPrompt(ContinualModel):
    NAME = 'coda_prompt'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual Learning via'
                                ' CODA-Prompt: COntinual Decomposed Attention-based Prompting')
        parser.add_argument('--mu', type=float, default=0.0, help='weight of prompt loss')
        parser.add_argument('--pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--prompt_len', type=int, default=8, help='prompt length')
        parser.add_argument('--virtual_bs_iterations', type=int, default=1, help="virtual batch size iterations")
        return parser

    def __init__(self, backbone, loss, args, transform):
        del backbone
        print("-" * 20)
        print(f"WARNING: CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES
        self.n_tasks = self.dataset.N_TASKS
        backbone = Model(num_classes=self.n_classes, pt=True, prompt_param=[self.n_tasks, [args.pool_size, args.prompt_len, 0]])
        super().__init__(backbone, loss, args, transform)
        self.net.task_id = 0
        self.opt = self.get_optimizer()

    def get_optimizer(self):
        params_to_opt = list(self.net.prompt.parameters()) + list(self.net.last.parameters())
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.args.lr,
                         'weight_decay': self.args.optim_wd}
        if self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(**optimizer_arg)
        elif self.args.optimizer == 'adam':
            opt = torch.optim.Adam(**optimizer_arg)
        else:
            raise ValueError('Optimizer not supported for this method')
        return opt

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self.dataset.get_offsets(self.current_task)

        if self.current_task != 0:
            self.net.task_id = self.current_task
            self.net.prompt.process_task_count()
            self.opt = self.get_optimizer()

        self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
        self.old_epoch = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.scheduler and self.old_epoch != epoch:
            self.scheduler.step()
            self.old_epoch = epoch
            self.iteration = 0
        labels = labels.long()
        self.opt.zero_grad()
        logits, loss_prompt = self.net(inputs, train=True)
        loss_prompt = loss_prompt.sum()
        logits = logits[:, :self.offset_2]
        logits[:, :self.offset_1] = -float('inf')
        loss_ce = self.loss(logits, labels)
        loss = loss_ce + self.args.mu * loss_prompt
        if self.task_iteration == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        loss.backward()
        if self.task_iteration > 0 and self.task_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()

    def forward(self, x):
        return self.net(x)[:, :self.offset_2]
