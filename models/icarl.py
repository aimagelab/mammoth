# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    return parser


def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        feats = self.net(norm_trans(not_norm_x), returnt='features')
        outs = self.net.classifier(feats)
        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device)
            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)


class ICarl(ContinualModel):
    NAME = 'icarl'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ICarl, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.task = 0

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.task, logits)
        loss.backward()

        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, :ac]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss

    def begin_task(self, dataset):
        icarl_replay(self, dataset)

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            fill_buffer(self, self.buffer, dataset, self.task)
        self.task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)
