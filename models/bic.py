# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay

# based on https://github.com/sairin1202/BIC


class BiC(ContinualModel):
    """Bias Correction."""
    NAME = 'bic'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--bic_epochs', type=int, default=250,
                            help='bias injector.')
        parser.add_argument('--temp', type=float, default=2.,
                            help='softmax temperature')
        parser.add_argument('--valset_split', type=float, default=0.1,
                            help='bias injector.')
        parser.add_argument('--wd_reg', type=float, default=None,
                            help='bias injector.')
        parser.add_argument('--distill_after_bic', type=binary_to_boolean_type, default=1)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

        self.lamda = 0

    def begin_task(self, dataset):
        if self.current_task > 0:

            self.old_net = deepcopy(self.net.eval())
            if hasattr(self, 'corr_factors'):
                self.old_corr = deepcopy(self.corr_factors)
            self.net.train()
            self.lamda = 1 / (self.current_task + 1)

            icarl_replay(self, dataset, val_set_split=self.args.valset_split)

        if hasattr(self, 'corr_factors'):
            del self.corr_factors

    def evaluate_bias(self, fprefx):
        resp = torch.zeros((self.current_task + 1) * self.cpt).to(self.device)
        with torch.no_grad():
            with bn_track_stats(self, False):
                for data in self.val_loader:

                    inputs, labels = data[0], data[1]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    resp += self.forward(inputs, anticipate=fprefx == 'post')[:, :(self.current_task + 1) * self.cpt].sum(0)
        resp /= len(self.val_loader.dataset)

        if fprefx == 'pre':
            self.oldresp = resp.cpu()

    def end_task(self, dataset):
        if self.current_task > 0:
            self.net.eval()

            print("EVAL PRE", dataset.evaluate(self, dataset))

            self.evaluate_bias('pre')

            corr_factors = torch.tensor([0., 1.], device=self.device, requires_grad=True)
            self.biasopt = Adam([corr_factors], lr=0.001)

            for l in range(self.args.bic_epochs):
                for data in self.val_loader:

                    inputs, labels = data[0], data[1]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.biasopt.zero_grad()
                    with torch.no_grad():
                        out = self.forward(inputs)

                    start_last_task = self.n_past_classes
                    end_last_task = self.n_seen_classes
                    tout = out + 0
                    tout[:, start_last_task:end_last_task] *= corr_factors[1].repeat_interleave(end_last_task - start_last_task)
                    tout[:, start_last_task:end_last_task] += corr_factors[0].repeat_interleave(end_last_task - start_last_task)

                    loss_bic = self.loss(tout[:, :end_last_task], labels)
                    loss_bic.backward()
                    self.biasopt.step()

            self.corr_factors = corr_factors
            print(self.corr_factors, file=sys.stderr)

            self.evaluate_bias('post')

            self.net.train()

        self.build_buffer(dataset)

    def forward(self, x, anticipate=False):
        ret = super().forward(x)
        if ret.shape[0] > 0:
            if hasattr(self, 'corr_factors'):
                start_last_task = (self.current_task - 1 + (1 if anticipate else 0)) * self.cpt
                end_last_task = (self.current_task + (1 if anticipate else 0)) * self.cpt
                ret[:, start_last_task:end_last_task] *= self.corr_factors[1].repeat_interleave(end_last_task - start_last_task)
                ret[:, start_last_task:end_last_task] += self.corr_factors[0].repeat_interleave(end_last_task - start_last_task)
        return ret

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)

        dist_loss = torch.tensor(0.)
        if self.current_task > 0:
            with torch.no_grad():

                old_outputs = self.old_net(inputs)
                if self.args.distill_after_bic:
                    if hasattr(self, 'old_corr'):
                        start_last_task = (self.current_task - 1) * self.cpt
                        end_last_task = (self.current_task) * self.cpt
                        old_outputs[:, start_last_task:end_last_task] *= self.old_corr[1].repeat_interleave(end_last_task - start_last_task)
                        old_outputs[:, start_last_task:end_last_task] += self.old_corr[0].repeat_interleave(end_last_task - start_last_task)

            pi_hat = F.log_softmax(outputs[:, :self.current_task * self.cpt] / self.args.temp, dim=1)
            pi = F.softmax(old_outputs[:, :self.current_task * self.cpt] / self.args.temp, dim=1)

            dist_loss = -(pi_hat * pi).sum(1).mean()

        class_loss = self.loss(outputs[:, :(self.current_task + 1) * self.cpt], labels, reduction='none')
        loss = (1 - self.lamda) * class_loss.mean() + self.lamda * dist_loss.mean() * self.args.temp * self.args.temp

        if self.args.wd_reg:
            loss += self.args.wd_reg * torch.sum(self.net.module.get_params() ** 2)

        loss.backward()

        self.opt.step()

        return loss.item()

    def build_buffer(self, dataset):

        examples_per_task = self.buffer.buffer_size // self.current_task if self.current_task > 0 else self.buffer.buffer_size

        if self.current_task > 1:
            # shrink buffer
            buf_x, buf_y, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, lab, tasklab = buf_x[idx], buf_y[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        counter = 0
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                labels, not_aug_inputs = data[1], data[2]
                not_aug_inputs = not_aug_inputs.to(self.device)
                if examples_per_task - counter > 0:
                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                         labels=labels[:(examples_per_task - counter)],
                                         task_labels=(torch.ones(self.args.batch_size) *
                                                      (self.current_task - 1))[:(examples_per_task - counter)])
                    counter += len(not_aug_inputs)
