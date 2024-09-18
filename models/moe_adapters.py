"""
Implementation of MoE-Adapters from the CVPR 2024 paper "Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters"
Paper: https://arxiv.org/abs/2403.11549
Original code: https://github.com/JiazuoYu/MoE-Adapters4CL
"""

import logging
import torch
from torch.optim.optimizer import Optimizer as Optimizer
from argparse import ArgumentParser

from models.moe_adapters_utils import clip
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.future_model import FutureModel
from utils.schedulers import CosineSchedulerWithLinearWarmup


class Model(torch.nn.Module):
    def __init__(self, args, dataset: ContinualDataset, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.dataset = dataset

        self.prompt_template = args.prompt_template
        self.device = device
        self.classes_names = self.dataset.get_class_names()
        self.model, self.clip_preprocess, _ = clip.load(args.clip_backbone, device=self.device, jit=False)

        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.classes_names]
        ).to(self.device)

        for k, v in self.model.named_parameters():
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

    def forward(self, images: torch.Tensor, n_past_classes=0, n_seen_classees=None, train=False) -> torch.Tensor:
        if train:
            n_seen_classees = self.text_tokens.shape[0] if n_seen_classees is None else n_seen_classees
            logits, _ = self.model(images, self.text_tokens[n_past_classes:n_seen_classees], 0, is_train=True)
        else:
            with torch.no_grad():
                logits, _ = self.model(images, self.text_tokens, 0, is_train=False)
                logits = logits.softmax(dim=-1)
        return logits


class MoEAdapters(FutureModel):
    """MoE Adapters -- Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters."""
    # https://arxiv.org/pdf/2403.11549v1
    NAME = 'moe-adapters'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    net: Model

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Frozen hyperparameters
        parser.set_defaults(batch_size=64, n_epochs=1, optimizer='adamw', lr=1e-3, eval_future=1)
        parser.add_argument("--virtual_bs_n", type=int, default=1, help="Virtual batch size iterations")

        # Tunable hyperparameters
        parser.add_argument("--clip_backbone", type=str, default='ViT-B/16', help="Clip backbone")

        parser.add_argument("--prompt_template", type=str, default='a bad photo of a {}.', help="Template string")

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert args.lr_scheduler is None, "MoE Adapters does not require a learning rate scheduler and will use a custom one."

        if args.optimizer != 'adamw':
            logging.warning("MoE Adapters should use AdamW optimizer.")

        logging.info("MoE Adapters redefines the tokenizer of CLIP. Check out the changes in models/moe_adapters_utils/tokenizer.py .")

        del backbone
        logging.info("MoE Adapters will override the backbone model.")
        super().__init__(None, loss, args, transform, dataset=dataset)
        self.net = Model(args, self.dataset, device=self.device)
        self.opt = self.get_optimizer()

    def get_parameters(self):
        return [v for k, v in self.net.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k]

    def get_optimizer(self) -> Optimizer:
        return super().get_optimizer()

    def begin_task(self, dataset):
        self.change_transform(dataset)

        self.opt = self.get_optimizer()

        num_batches = len(dataset.train_loader)
        total_iterations = self.args.n_epochs * num_batches
        self.custom_scheduler = CosineSchedulerWithLinearWarmup(self.opt, self.args.lr, 30, total_iterations)

    def change_transform(self, dataset):
        dataset.train_loader.dataset.transform = self.net.clip_preprocess
        dataset.test_loaders[-1].dataset.transform = self.net.clip_preprocess

    def forward(self, x):
        logits = self.net(x, n_seen_classees=self.n_seen_classes)
        return logits[:, :self.n_seen_classes]

    def future_forward(self, x):
        return self.net(x)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        logits = self.net(inputs, n_past_classes=self.n_past_classes, n_seen_classees=self.n_seen_classes, train=True)
        # -- cross entropy loss --
        loss = self.loss(logits, labels - self.n_past_classes)

        if self.task_iteration == 0:
            self.opt.zero_grad()

        (loss / self.args.virtual_bs_n).backward()
        if (self.task_iteration > 0 or self.args.virtual_bs_n == 1) and self.task_iteration % self.args.virtual_bs_n == 0:
            self.opt.step()
            self.opt.zero_grad()
            self.custom_scheduler.step()

        return loss.item()
