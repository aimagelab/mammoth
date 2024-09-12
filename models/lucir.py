# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from datasets import get_dataset
from torch import nn

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, fill_buffer, icarl_replay
from utils.conf import create_seeded_dataloader


def lucir_batch_hard_triplet_loss(labels, embeddings, k, margin, num_old_classes):
    """
    LUCIR triplet loss.
    """
    gt_index = torch.zeros(embeddings.size()).to(embeddings.device)
    gt_index = gt_index.scatter(1, labels.reshape(-1, 1).long(), 1).ge(0.5)
    gt_scores = embeddings.masked_select(gt_index)
    # get top-K scores on novel classes
    max_novel_scores = embeddings[:, num_old_classes:].topk(k, dim=1)[0]
    # the index of hard samples, i.e., samples of old classes
    hard_index = labels.lt(num_old_classes)
    hard_num = torch.nonzero(hard_index).size(0)
    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, k)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        target = torch.ones(hard_num * k, 1).to(embeddings.device)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1),
                                                   max_novel_scores.view(-1, 1), target)
    else:
        loss = torch.zeros(1).to(embeddings.device)

    return loss


class CustomClassifier(nn.Module):
    def __init__(self, in_features, cpt, n_tasks):
        super().__init__()

        self.weights = nn.ParameterList(
            [nn.parameter.Parameter(torch.Tensor(cpt, in_features))
             for _ in range(n_tasks)]
        )
        self.sigma = nn.parameter.Parameter(torch.Tensor(1))

        self.in_features = in_features
        self.cpt = cpt
        self.n_tasks = n_tasks
        self.reset_parameters()
        self.weights[0].requires_grad = True

    def reset_parameters(self):
        for i in range(self.n_tasks):
            stdv = 1. / math.sqrt(self.weights[i].size(1))
            self.weights[i].data.uniform_(-stdv, stdv)
            self.weights[i].requires_grad = False

        self.sigma.data.fill_(1)

    def forward(self, x):
        return self.noscale_forward(x) * self.sigma

    def reset_weight(self, i):
        stdv = 1. / math.sqrt(self.weights[i].size(1))
        self.weights[i].data.uniform_(-stdv, stdv)
        self.weights[i].requires_grad = True
        self.weights[i - 1].requires_grad = False

    def noscale_forward(self, x):
        out = None

        x = F.normalize(x, p=2, dim=1).reshape(len(x), -1)

        for t in range(self.n_tasks):
            o = F.linear(x, F.normalize(self.weights[t], p=2, dim=1))
            if out is None:
                out = o
            else:
                out = torch.cat((out, o), dim=1)

        return out


class Lucir(ContinualModel):
    """Continual Learning via Lucir."""
    NAME = 'lucir'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--lamda_base', type=float, required=False, default=5.,
                            help='Regularization weight for embedding cosine similarity.')
        parser.add_argument('--lamda_mr', type=float, required=False, default=1.,
                            help='Regularization weight for embedding cosine similarity.')
        parser.add_argument('--k_mr', type=int, required=False, default=2,
                            help='K for margin-ranking loss.')
        parser.add_argument('--mr_margin', type=float, default=0.5,
                            required=False, help='Margin for margin-ranking loss.')
        parser.add_argument('--fitting_epochs', type=int, required=False, default=20,
                            help='Number of epochs to finetune on coreset after each task.')
        parser.add_argument('--lr_finetune', type=float, required=False, default=0.01,
                            help='Learning Rate for finetuning.')
        parser.add_argument('--imprint_weights', type=binary_to_boolean_type, required=False, default=1,
                            help='Apply weight imprinting?')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Lucir, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.old_net = None
        self.epochs = int(args.n_epochs)
        self.lamda_cos_sim = args.lamda_base

        self.net.classifier = CustomClassifier(
            self.net.classifier.in_features, self.dataset.N_CLASSES_PER_TASK, self.dataset.N_TASKS)

        upd_weights = [p for n, p in self.net.named_parameters()
                       if 'classifier' not in n and '_fc' not in n] + [self.net.classifier.weights[0], self.net.classifier.sigma]
        fix_weights = list(self.net.classifier.weights[1:])

        self.opt = torch.optim.SGD([{'params': upd_weights, 'lr': self.args.lr, 'momentum': self.args.optim_mom, 'weight_decay': self.args.optim_wd}, {
            'params': fix_weights, 'lr': 0, 'momentum': self.args.optim_mom, 'weight_decay': 0}])

        self.ft_lr_strat = [10]

        self.c_epoch = -1

    def update_classifier(self):
        self.net.classifier.reset_weight(self.current_task)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.net(x)

        return outputs

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None, fitting=False):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()
        loss = self.get_loss(
            inputs, labels.long(), self.current_task)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index

        Returns:
            the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs, returnt='features').float()

        cos_output = self.net.classifier.noscale_forward(outputs)
        outputs = outputs.reshape(outputs.size(0), -1)

        loss = F.cross_entropy(cos_output * self.net.classifier.sigma, labels)

        if task_idx > 0:
            with torch.no_grad():
                logits = self.old_net(inputs, returnt='features')
                logits = logits.reshape(logits.size(0), -1)

            loss2 = F.cosine_embedding_loss(
                outputs, logits.detach(), torch.ones(outputs.shape[0]).to(outputs.device)) * self.lamda_cos_sim

            # Remove rescale by sigma before this loss
            loss3 = lucir_batch_hard_triplet_loss(
                labels, cos_output, self.args.k_mr, self.args.mr_margin, pc) * self.args.lamda_mr

            loss = loss + loss2 + loss3

        return loss

    def begin_task(self, dataset):

        if self.current_task > 0:
            icarl_replay(self, dataset)

            with torch.no_grad():
                # Update model classifier
                self.update_classifier()

                if self.args.imprint_weights:
                    self.imprint_weights(dataset)

                # Restore optimizer LR
                upd_weights = [p for n, p in self.net.named_parameters()
                               if 'classifier' not in n] + [self.net.classifier.weights[self.current_task], self.net.classifier.sigma]
                fix_weights = list(
                    self.net.classifier.weights[:self.current_task])

                if self.current_task < self.dataset.N_TASKS - 1:
                    fix_weights += list(
                        self.net.classifier.weights[self.current_task + 1:])

                self.opt = torch.optim.SGD([{'params': upd_weights, 'lr': self.args.lr, 'weight_decay': self.args.optim_wd}, {
                    'params': fix_weights, 'lr': 0, 'weight_decay': 0}], lr=self.args.lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())

        self.net.train()
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=True)

        if self.args.fitting_epochs is not None and self.args.fitting_epochs > 0:
            self.fit_buffer(self.args.fitting_epochs)

        # Adapt lambda
        self.lamda_cos_sim = math.sqrt(self.current_task) * float(self.args.lamda_base)

    def imprint_weights(self, dataset):
        self.net.eval()
        old_embedding_norm = torch.cat([self.net.classifier.weights[i] for i in range(self.current_task)]).norm(
            dim=1, keepdim=True)
        average_old_embedding_norm = torch.mean(
            old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
        num_features = self.net.classifier.in_features
        novel_embedding = torch.zeros(
            (self.dataset.N_CLASSES_PER_TASK, num_features))
        loader = dataset.train_loader

        cur_dataset = deepcopy(loader.dataset)

        for cls_idx in range(self.current_task * self.dataset.N_CLASSES_PER_TASK, (self.current_task + 1) * self.dataset.N_CLASSES_PER_TASK):

            cls_indices = np.asarray(
                loader.dataset.targets) == cls_idx
            cur_dataset.data = loader.dataset.data[cls_indices]
            cur_dataset.targets = np.zeros((cur_dataset.data.shape[0]))
            dt = create_seeded_dataloader(self.args,
                                          cur_dataset, batch_size=self.args.batch_size, num_workers=0)

            num_samples = cur_dataset.data.shape[0]
            cls_features = torch.empty((num_samples, num_features))
            for j, d in enumerate(dt):
                tt = self.net(d[0].to(self.device), returnt='features').cpu()
                if 'ntu' in self.args.dataset:
                    tt = F.adaptive_avg_pool3d(tt, 1)
                cls_features[j * self.args.batch_size:(
                    j + 1) * self.args.batch_size] = tt.reshape(len(tt), -1)

            norm_features = F.normalize(cls_features, p=2, dim=1)
            cls_embedding = torch.mean(norm_features, dim=0)

            novel_embedding[cls_idx - self.current_task * self.dataset.N_CLASSES_PER_TASK] = F.normalize(
                cls_embedding, p=2, dim=0) * average_old_embedding_norm

        self.net.classifier.weights[self.current_task].data = novel_embedding.to(
            self.device)
        self.net.train()

    def fit_buffer(self, opt_steps):

        old_opt = self.opt
        # Optimize only final embeddings
        self.opt = torch.optim.SGD(self.net.classifier.parameters(
        ), self.args.lr_finetune, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=self.ft_lr_strat, gamma=0.1)

        with bn_track_stats(self, False):
            for _ in range(opt_steps):
                examples, labels = self.buffer.get_all_data(self.transform, device=self.device)
                dset = torch.utils.data.TensorDataset(examples, labels)
                torch.cuda.synchronize()
                dt = create_seeded_dataloader(self.args, dset, shuffle=True, batch_size=self.args.batch_size, num_workers=0)
                for inputs, labels in dt:
                    self.observe(inputs, labels, None, fitting=True)
                    lr_scheduler.step()

        self.opt = old_opt
