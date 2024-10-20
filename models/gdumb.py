# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from torch.optim import SGD, lr_scheduler

from backbone import get_backbone
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.augmentations import cutmix_data
from utils.buffer import Buffer
from utils.status import progress_bar


def fit_buffer(self: ContinualModel, epochs):
    optimizer = SGD(self.get_parameters(), lr=self.args.maxlr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd, nesterov=self.args.optim_nesterov)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=self.args.minlr)

    for epoch in range(epochs):
        if epoch <= 0:  # Warm start of 1 epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.args.maxlr * 0.1
        elif epoch == 1:  # Then set to maxlr
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.args.maxlr
        else:
            scheduler.step()

        all_inputs, all_labels = self.buffer.get_data(
            len(self.buffer.examples), transform=self.transform, device=self.device)

        it = 0
        while len(all_inputs):
            if it > self.get_debug_iters() and self.args.debug_mode:
                break
            it += 1
            optimizer.zero_grad()
            buf_inputs, buf_labels = all_inputs[:self.args.batch_size], all_labels[:self.args.batch_size]
            all_inputs, all_labels = all_inputs[self.args.batch_size:], all_labels[self.args.batch_size:]

            if self.args.cutmix_alpha is not None:
                inputs, labels_a, labels_b, lam = cutmix_data(x=buf_inputs.cpu(), y=buf_labels.cpu(), alpha=self.args.cutmix_alpha)
                buf_inputs = inputs.to(self.device)
                buf_labels_a = labels_a.to(self.device)
                buf_labels_b = labels_b.to(self.device)
                buf_outputs = self.net(buf_inputs)
                loss = lam * self.loss(buf_outputs, buf_labels_a) + (1 - lam) * self.loss(buf_outputs, buf_labels_b)
            else:
                buf_outputs = self.net(buf_inputs)
                loss = self.loss(buf_outputs, buf_labels)

            loss.backward()
            optimizer.step()
        progress_bar(epoch, epochs, 1, 'G', loss.item())


class GDumb(ContinualModel):
    """Greedy sampler and Dumb Learner."""
    NAME = 'gdumb'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0, n_epochs=1)  # lr is managed elsewhere, need only 1 epoch to fill buffer
        add_rehearsal_args(parser)
        parser.add_argument('--maxlr', type=float, default=5e-2,
                            help='Max learning rate.')
        parser.add_argument('--minlr', type=float, default=5e-4,
                            help='Min learning rate.')
        parser.add_argument('--fitting_epochs', type=int, default=256,
                            help='Number of epochs to fit the buffer.')
        parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                            help='Alpha parameter for cutmix')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.n_epochs != 1:
            args.n_epochs = 1
            logging.info('GDumb needs only 1 epoch to fill the buffer.')

        super(GDumb, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)
        return 0

    def end_task(self, dataset):
        # new model
        if not (self.current_task == dataset.N_TASKS - 1):
            return
        self.net = get_backbone(self.args).to(self.device)
        fit_buffer(self, self.args.fitting_epochs)
