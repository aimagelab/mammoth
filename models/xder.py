# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils import binary_to_boolean_type
from utils.spkdloss import SPKDLoss
from datasets import get_dataset
from torch.nn import functional as F
from utils.args import *
import torch
from models.utils.continual_model import ContinualModel
from utils.augmentations import *
from utils.batch_norm import bn_track_stats
from utils.simclrloss import SupConLoss


class XDer(ContinualModel):
    """Continual learning via eXtended Dark Experience Replay."""
    NAME = 'xder'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')

        parser.add_argument('--simclr_temp', type=float, default=5, help='Temperature for SimCLR loss')
        parser.add_argument('--gamma', type=float, default=0.85, help='Weight for logit update')  # log_update_weight
        parser.add_argument('--simclr_batch_size', type=int, default=64, help='Batch size for SimCLR loss')
        parser.add_argument('--simclr_num_aug', type=int, default=4, help='Number of augmentations for SimCLR loss')
        parser.add_argument('--lambd', type=float, default=0.05, help='Weight for consistency loss')  # simclr_weight
        parser.add_argument('--constr_eta', type=float, default=0.1, help='Regularization weight for past/future constraints')  # constr_weight
        parser.add_argument('--constr_margin', type=float, default=0.3, help='Margin for past/future constraints')

        parser.add_argument('--dp_weight', type=float, default=0, help='Weight for distance preserving loss')

        parser.add_argument('--past_constraint', type=binary_to_boolean_type, default=0, help='Enable past constraint')
        parser.add_argument('--future_constraint', type=binary_to_boolean_type, default=1, help='Enable future constraint')
        parser.add_argument('--align_bn', type=binary_to_boolean_type, default=1, help='Use BatchNorm alignment')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        from utils.buffer import Buffer
        self.buffer = Buffer(self.args.buffer_size)
        self.update_counter = torch.zeros(self.args.buffer_size)

        denorm = self.dataset.get_denormalization_transform()
        self.dataset_mean, self.dataset_std = denorm.mean, denorm.std
        self.dataset_shape = self.dataset.SIZE
        self.gpu_augmentation = strong_aug(self.dataset_shape, self.dataset_mean, self.dataset_std)
        self.simclr_lss = SupConLoss(temperature=self.args.simclr_temp, base_temperature=self.args.simclr_temp, reduction='sum')

        self.spkdloss = SPKDLoss('batchmean')

    def end_task(self, dataset):

        tng = self.training
        self.train()

        # fdr reduce coreset
        if self.current_task > 0:
            examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
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

        # fdr add new task
        examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
        examples_per_class = examples_last_task // self.cpt
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

        with torch.no_grad():
            with bn_track_stats(self, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data[0], data[1], data[2]
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    outputs = self.net(inputs)
                    if all(ce == 0):
                        break

                    # update past
                    if self.current_task > 0:
                        outputs = self.update_logits(outputs, outputs, labels, 0, self.current_task)

                    flags = torch.zeros(len(inputs)).bool()
                    for j in range(len(flags)):
                        if ce[labels[j] % self.cpt] > 0:
                            flags[j] = True
                            ce[labels[j] % self.cpt] -= 1

                    self.buffer.add_data(examples=not_aug_inputs[flags],
                                         labels=labels[flags],
                                         logits=outputs.data[flags],
                                         task_labels=(torch.ones(self.args.batch_size) * self.current_task)[flags])

                # update future past
                buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.buffer.buffer_size,
                                                                                      transform=self.transform, return_index=True, device=self.device)

                buf_outputs = []
                while len(buf_inputs):
                    buf_outputs.append(self.net(buf_inputs[:self.args.batch_size]))
                    buf_inputs = buf_inputs[self.args.batch_size:]
                buf_outputs = torch.cat(buf_outputs)

                chosen = ((buf_labels // self.cpt) < self.current_task).to(self.buffer.device)

                if chosen.any():
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                    self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.current_task

        self.update_counter = torch.zeros(self.args.buffer_size)

        self.train(tng)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):

        transplant = new[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt]

        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt] = transplant

        return old

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        with bn_track_stats(self, not self.args.align_bn or self.current_task == 0):
            outputs = self.net(inputs)

        # Present head
        loss_stream = self.loss(outputs[:, self.n_past_classes:self.n_seen_classes], labels - self.n_past_classes)

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

            mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_der = self.args.alpha * mse.mean()

            # Label Replay Loss (past heads)
            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
            with bn_track_stats(self, not self.args.align_bn):
                buf_outputs2 = self.net(buf_inputs2)

            buf_ce = self.loss(buf_outputs2[:, :self.n_past_classes], buf_labels2)
            loss_derpp = self.args.beta * buf_ce

            # Merge Batches & Remove Duplicates
            buf_idx = torch.cat([buf_idx1, buf_idx2])
            buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
            buf_labels = torch.cat([buf_labels1, buf_labels2])
            buf_logits = torch.cat([buf_logits1, buf_logits2])
            buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
            buf_tl = torch.cat([buf_tl1, buf_tl2])

            # remove dupulicates
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
                chosen = ((buf_labels // self.cpt) < self.current_task).to(self.buffer.device)
                c = chosen.clone()
                self.update_counter[buf_idx[chosen]] += 1
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                if chosen.any():
                    assert self.current_task > 0
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                    self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.current_task

        # Consistency Loss (future heads)
        loss_cons, loss_dp = torch.tensor(0.), torch.tensor(0.)
        loss_constr_futu = torch.tensor(0.)
        if self.current_task < self.n_tasks - 1:

            scl_labels = labels  # [:self.args.simclr_batch_size]
            scl_na_inputs = not_aug_inputs  # [:self.args.simclr_batch_size]
            if not self.buffer.is_empty():
                buf_idxscl, buf_na_inputsscl, buf_labelsscl, buf_logitsscl, _ = self.buffer.get_data(self.args.simclr_batch_size,
                                                                                                     transform=None, return_index=True, device=self.device)
                scl_na_inputs = torch.cat([buf_na_inputsscl, scl_na_inputs])
                scl_labels = torch.cat([buf_labelsscl, scl_labels])
            with torch.no_grad():
                scl_inputs = self.gpu_augmentation(scl_na_inputs.repeat_interleave(self.args.simclr_num_aug, 0)).to(self.device)

            with bn_track_stats(self, not self.args.align_bn):
                scl_outputs = self.net(scl_inputs)

            scl_featuresFull = scl_outputs.reshape(-1, self.args.simclr_num_aug, scl_outputs.shape[-1])

            scl_features = scl_featuresFull[:, :, (self.current_task + 1) * self.cpt:]
            scl_n_heads = self.n_tasks - self.current_task - 1

            scl_features = torch.stack(scl_features.split(self.cpt, 2), 1)

            loss_cons = torch.stack([self.simclr_lss(features=F.normalize(scl_features[:, h], dim=2), labels=scl_labels) for h in range(scl_n_heads)]).sum()
            loss_cons /= scl_n_heads * scl_features.shape[0]
            loss_cons *= self.args.lambd

            # DP loss
            if self.args.dp_weight > 0 and not self.buffer.is_empty():
                dp_features = scl_featuresFull[:len(buf_logitsscl), :, (self.current_task + 1) * self.cpt:]
                dp_logits = buf_logitsscl[:, (self.current_task + 1) * self.cpt:]

                dp_features = torch.stack(dp_features.split(self.cpt, 2), 1)

                dp_logits = torch.stack(dp_logits.split(self.cpt, 1), 1)

                loss_dp = self.args.dp_weight * torch.mean(torch.stack(
                    [self.spkdloss(dp_features[:, i, k, :], dp_logits[:, i, :]) for i in range(self.n_tasks - self.current_task - 1) for k in range(self.args.simclr_num_aug)]
                ))

            # Future Logits Constraint
            if self.args.future_constraint:
                bad_head = outputs[:, (self.current_task + 1) * self.cpt:]
                good_head = outputs[:, self.current_task * self.cpt:(self.current_task + 1) * self.cpt]

                if not self.buffer.is_empty():
                    buf_tlgt = buf_labels // self.cpt
                    bad_head = torch.cat([bad_head, buf_outputs[:, (self.current_task + 1) * self.cpt:]])
                    good_head = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

                loss_constr = bad_head.max(1)[0] + self.args.constr_margin - good_head.max(1)[0]

                mask = loss_constr > 0
                if (mask).any():
                    loss_constr_futu = self.args.constr_eta * loss_constr[mask].mean()

        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.args.past_constraint and self.current_task > 0:
            chead = F.softmax(outputs[:, :(self.current_task + 1) * self.cpt], 1)

            good_head = chead[:, self.current_task * self.cpt:(self.current_task + 1) * self.cpt]
            bad_head = chead[:, :self.cpt * self.current_task]

            loss_constr = bad_head.max(1)[0].detach() + self.args.constr_margin - good_head.max(1)[0]

            mask = loss_constr > 0

            if (mask).any():
                loss_constr_past = self.args.constr_eta * loss_constr[mask].mean()

        loss = loss_stream + loss_der + loss_derpp + loss_cons + loss_dp + loss_constr_futu + loss_constr_past

        loss.backward()
        self.opt.step()

        return loss.item()
