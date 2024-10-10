"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import logging
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from datasets import get_dataset
from models.coda_prompt_utils.model import Model
from utils.schedulers import CosineSchedule


class CodaPrompt(ContinualModel):
    """Continual Learning via CODA-Prompt: COntinual Decomposed Attention-based Prompting."""
    NAME = 'coda_prompt'
    COMPATIBILITY = ['class-il', 'task-il']
    net: Model

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0.001, optimizer='adam', optim_mom=0.9)
        parser.add_argument('--mu', type=float, default=0.0, help='weight of ortho prompt loss')
        parser.add_argument('--pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--prompt_len', type=int, default=8, help='prompt length')
        parser.add_argument('--virtual_bs_iterations', '--virtual_bs_n', dest='virtual_bs_iterations',
                            type=int, default=1, help="virtual batch size iterations")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        logging.info(f"CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224.augreg_in21k_ft_in1k`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        assert args.lr_scheduler is None, "CODA-Prompt uses a custom scheduler: cosine. Ignoring --lr_scheduler."

        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES
        self.n_tasks = self.dataset.N_TASKS
        backbone = Model(num_classes=self.n_classes, pt=True, prompt_param=[self.n_tasks, [args.pool_size, args.prompt_len, 0]])
        super().__init__(backbone, loss, args, transform, dataset=dataset)
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

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        labels = labels.long()
        self.opt.zero_grad()
        logits, loss_prompt = self.net(inputs, train=True)
        loss_prompt = loss_prompt.sum()
        logits = logits[:, :self.offset_2]
        logits[:, :self.offset_1] = -float('inf')
        loss_ce = self.loss(logits, labels)
        loss = loss_ce + self.args.mu * loss_prompt
        if self.epoch_iteration == 0:
            self.opt.zero_grad()

        (loss / float(self.args.virtual_bs_iterations)).backward()
        if (self.epoch_iteration > 0 or self.args.virtual_bs_iterations == 1) and self.epoch_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()

    def forward(self, x):
        return self.net(x)[:, :self.offset_2]
