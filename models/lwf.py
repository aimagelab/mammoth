# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    """Continual learning via Learning without Forgetting."""
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Penalty weight.')
        parser.add_argument('--softmax_temp', type=float, default=2,
                            help='Temperature of the softmax function.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Lwf, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.classifier.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(dataset.train_loader):
                    inputs, labels = data[0], data[1]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net(inputs, returnt='features')
                    outputs = self.net.classifier(feats)[:, self.n_past_classes: self.n_seen_classes]
                    loss = self.loss(outputs, labels - self.n_past_classes)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                    inputs = torch.stack([dataset.train_loader.dataset.__getitem__(j)[2]
                                          for j in range(i, min(i + self.args.batch_size,
                                                         len(dataset.train_loader.dataset)))])
                    log = self.net(inputs.to(self.device)).cpu()
                    logits.append(log)
            dataset.train_loader.dataset.logits = torch.cat(logits)
            dataset.train_loader.dataset.extra_return_fields += ('logits',)
        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)

        loss = self.loss(outputs[:, :self.n_seen_classes], labels)
        if logits is not None:
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[:, :self.n_past_classes]).to(self.device), self.args.softmax_temp, 1),
                                                      smooth(self.soft(outputs[:, :self.n_past_classes]), self.args.softmax_temp, 1))

        loss.backward()
        self.opt.step()

        return loss.item()
