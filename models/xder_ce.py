# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer


class XDerCE(ContinualModel):
    NAME = 'xder_ce'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' eXtended Dark Experience Replay with cross-entropy on future heads.')

        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')

        parser.add_argument('--gamma', type=float, default=0.85)
        parser.add_argument('--eta', type=float, default=0.1)
        parser.add_argument('--m', type=float, default=0.3)

        parser.add_argument('--past_constraint', type=int, default=1, choices=[0,1], help='Enable past constraint')
        parser.add_argument('--future_constraint', type=int, default=1, choices=[0,1], help='Enable future constraint')
        parser.add_argument('--align_bn', type=int, default=0, choices=[0,1], help='Use BatchNorm alignment')

        return parser

    def __init__(self, backbone, loss, args, transform):
        super(XDerCE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.update_counter = torch.zeros(self.args.buffer_size)

        if not hasattr(self.args, 'start_from'):
            self.args.start_from = 0

    def end_task(self, dataset):

        tng = self.training
        self.train()

        if self.args.start_from is None or self.current_task >= self.args.start_from:
            # Reduce Memory Buffer
            if self.current_task > 0:
                examples_per_class = self.args.buffer_size // self.n_seen_classes
                buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                self.buffer.empty()
                for tl in buf_lab.unique():
                    idx = tl == buf_lab
                    ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                    first = min(ex.shape[0], examples_per_class)
                    self.buffer.add_data(
                        examples=ex[:first],
                        labels=lab[:first],
                        logits=log[:first],
                        task_labels=tasklab[:first]
                    )

            # To make this baseline as strong as can be,
            # this model uses reservoir at task 0 (just like DER++)
            # and then switches to task-stratified later (like X-DER)
            if self.current_task == 0:
                self.buffer.empty()

            # Add new task data
            examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
            examples_per_class = examples_last_task // self.n_classes_current_task
            ce = torch.tensor([examples_per_class] * self.n_classes_current_task).int()
            ce[torch.randperm(self.n_classes_current_task)[:examples_last_task - (examples_per_class * self.n_classes_current_task)]] += 1

            with torch.no_grad():
                with bn_track_stats(self, False):
                    if self.args.start_from is None or self.args.start_from <= self.current_task:
                        for data in dataset.train_loader:
                            inputs, labels, not_aug_inputs = data
                            inputs = inputs.to(self.device)
                            not_aug_inputs = not_aug_inputs.to(self.device)
                            outputs = self.net(inputs)
                            if all(ce == 0):
                                break

                            # Update past logits
                            if self.current_task > 0:
                                outputs = self.update_logits(outputs, outputs, labels, 0, self.current_task)

                            flags = torch.zeros(len(inputs)).bool()
                            for j in range(len(flags)):
                                if ce[labels[j] % self.n_classes_current_task] > 0:
                                    flags[j] = True
                                    ce[labels[j] % self.n_classes_current_task] -= 1

                            self.buffer.add_data(examples=not_aug_inputs[flags],
                                                 labels=labels[flags],
                                                 logits=outputs.data[flags],
                                                 task_labels=(torch.ones(len(not_aug_inputs)) * self.current_task)[flags])

                    # Update future past logits
                    buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.buffer.buffer_size,
                                                                                          transform=self.transform, return_index=True, device=self.device)

                    buf_outputs = []
                    while len(buf_inputs):
                        buf_outputs.append(self.net(buf_inputs[:self.args.batch_size]))
                        buf_inputs = buf_inputs[self.args.batch_size:]
                    buf_outputs = torch.cat(buf_outputs)

                    chosen = ((buf_labels // self.n_classes_current_task) < self.current_task).to(self.buffer.device)

                    if chosen.any():
                        to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task)
                        self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                        self.buffer.task_labels[buf_idx[chosen]] = self.current_task

        self.update_counter = torch.zeros(self.args.buffer_size)

        self.train(tng)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):
        offset_1, _ = self.dataset.get_offsets(task_start)
        offset_2, _ = self.dataset.get_offsets(task_start + n_tasks)

        transplant = new[:, offset_1:offset_2]

        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, offset_2 - offset_1)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, offset_2 - offset_1)
        transplant[mask] *= coeff[mask]
        old[:, offset_1:offset_2] = transplant

        return old

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        with bn_track_stats(self, self.args.align_bn==0 or self.current_task == 0):
            outputs = self.net(inputs)

        # Present head
        loss_stream = self.loss(outputs[:, self.n_past_classes:], labels % self.n_classes_current_task)

        loss_der, loss_derpp = torch.tensor(0.), torch.tensor(0.)
        if not self.buffer.is_empty():
            # Distillation Replay Loss (all heads)
            buf_idx1, buf_inputs1, buf_labels1, buf_logits1, buf_tl1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
            if self.args.align_bn:
                buf_inputs1 = torch.cat([buf_inputs1, inputs[:self.args.minibatch_size // self.current_task]])

            buf_outputs1 = self.net(buf_inputs1)
            
            if self.args.align_bn:
                buf_inputs1 = buf_inputs1[:self.args.minibatch_size]
                buf_outputs1 = buf_outputs1[:self.args.minibatch_size]

            buf_logits1 = buf_logits1.type(buf_outputs1.dtype)
            mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_der = self.args.alpha * mse.mean()

            # Label Replay Loss (past heads)
            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
            with bn_track_stats(self, self.args.align_bn==0):
                buf_outputs2 = self.net(buf_inputs2)

            _, offset = self.dataset.get_offsets(self.current_task + (1 if self.current_task == 0 else 0))
            buf_ce = self.loss(buf_outputs2[:, :offset], buf_labels2)
            loss_derpp = self.args.beta * buf_ce

            # Merge Batches & Remove Duplicates
            buf_idx = torch.cat([buf_idx1, buf_idx2])
            buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
            buf_labels = torch.cat([buf_labels1, buf_labels2])
            buf_logits = torch.cat([buf_logits1, buf_logits2])
            buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
            buf_tl = torch.cat([buf_tl1, buf_tl2])
            eyey = torch.eye(self.buffer.buffer_size).to(buf_idx.device)[buf_idx]
            umask = (eyey * eyey.cumsum(0)).sum(1) < 2

            buf_idx = buf_idx[umask].to(self.buffer.device)
            buf_inputs = buf_inputs[umask]
            buf_labels = buf_labels[umask]
            buf_logits = buf_logits[umask]
            buf_outputs = buf_outputs[umask]
            buf_tl = buf_tl[umask]

            # Update Future Past Logits
            with torch.no_grad():
                chosen = ((buf_labels // self.n_classes_current_task) < self.current_task).to(self.buffer.device)
                self.update_counter[buf_idx[chosen]] += 1
                c = chosen.clone()
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                if chosen.any():
                    assert self.current_task > 0
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task).to(self.buffer.device)
                    self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.current_task

        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.args.past_constraint and self.current_task > 0:
            chead = F.softmax(outputs[:, :self.n_seen_classes], 1)

            good_head = chead[:, self.n_past_classes:self.n_seen_classes]
            bad_head = chead[:, :self.n_past_classes]

            loss_constr = bad_head.max(1)[0].detach() + self.args.m - good_head.max(1)[0]

            mask = loss_constr > 0

            if (mask).any():
                loss_constr_past = self.args.eta * loss_constr[mask].mean()

        # Future Logits Constraint
        loss_constr_futu = torch.tensor(0.)
        if self.args.future_constraint and self.current_task < self.n_tasks - 1:
            bad_head = outputs[:, self.n_seen_classes:]
            good_head = outputs[:, self.n_past_classes:self.n_seen_classes]

            if not self.buffer.is_empty():
                buf_tlgt = buf_labels // self.n_classes_current_task
                bad_head = torch.cat([bad_head, buf_outputs[:, self.n_seen_classes:]])
                good_head = torch.cat([good_head, torch.stack(buf_outputs.split(self.n_classes_current_task, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

            loss_constr = bad_head.max(1)[0] + self.args.m - good_head.max(1)[0]

            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = self.args.eta * loss_constr[mask].mean()

        loss = loss_stream + loss_der + loss_derpp + loss_constr_futu + loss_constr_past

        loss.backward()
        self.opt.step()

        if self.current_task == 0:
            self.buffer.add_data(
                examples=inputs,
                labels=labels,
                logits=outputs.detach(),
                task_labels=torch.zeros_like(labels)
            )

        return loss.item()
