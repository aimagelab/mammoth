# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay, fill_buffer
from torchvision.transforms import functional as TF


@torch.no_grad()
def c100_transform(inputs):
    """
    Original augmentation from `https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/utils_cifar100.py`
    """
    orig_device = inputs.device
    inputs = inputs.cpu().numpy()
    batchsize = inputs.shape[0]

    padded = np.pad(inputs, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
    random_cropped = np.zeros(inputs.shape, dtype=np.float32)
    crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
    for r in range(batchsize):
        # Cropping and possible flipping
        if (np.random.randint(2) > 0):
            random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32), crops[r, 1]:(crops[r, 1] + 32)]
        else:
            random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32), crops[r, 1]:(crops[r, 1] + 32)][:, :, ::-1]
    inp_exc = random_cropped

    return torch.from_numpy(inp_exc).to(orig_device)


class ICarl(ContinualModel):
    """Continual Learning via iCaRL."""
    NAME = 'icarl'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--compute_theoretical_best', type=binary_to_boolean_type, default=0,
                            help='Compute NCM with the theoretical case where all the training samples are stored AND extend the buffer to equalize number of samples? (as in the original code)')
        parser.add_argument('--use_original_icarl_transform', type=binary_to_boolean_type, default=0,
                            help='Use the original iCaRL transform?')
        parser.add_argument('--opt_wd', type=float, default=1e-5,
                            help='Optimizer weight decay')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.optim_wd != 0:
            logging.warning('iCaRL uses a custom weight decay, the optimizer weight decay will be ignored.')
            args.optim_wd = 0
        if args.use_original_icarl_transform:
            assert args.dataset == 'seq-cifar100', 'The original iCaRL transform is only available for CIFAR-100.'

        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size)
        self.eye = torch.eye(self.num_classes).to(self.device)

        if self.args.compute_theoretical_best:
            logging.warning('Using the theoretical best for NCM, using all the training samples from ALL TASKS!')
            logging.warning('This will also extend the buffer to equalize the number of samples for each class.')
            self.ncm_buffer = Buffer(-1, device='cpu', sample_selection_strategy='unlimited')  # Unlimited buffer

        self.class_means = None
        self.old_net = None

        self.alpha_dr_herding = np.zeros((self.n_tasks, 500, self.cpt), np.float32)

    def forward(self, x):
        if self.class_means is None:
            print('Computing class means...')
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats / feats.norm(dim=1, keepdim=True)
        feats = feats.view(feats.size(0), -1)

        pred = (self.class_means.unsqueeze(0) - feats.unsqueeze(1)).pow(2).sum(2)
        return -pred

    def wd(self):
        loss = 0
        for p in self.net.parameters():
            loss += p.pow(2).sum()
        return loss

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        if self.args.use_original_icarl_transform:
            inputs = c100_transform(self.normalization_transform(not_aug_inputs))

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, logits)
        loss = loss + self.wd() * self.args.opt_wd
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

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index
            logits: the logits of the old network

        Returns:
            the differentiable loss value
        """

        outputs = self.net(inputs)
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)  # self.binary_cross_entropy(F.sigmoid(outputs), targets)  #
            assert loss >= 0
        else:
            targets = self.eye[labels]
            comb_targets = torch.cat((logits[:, :self.n_past_classes], targets[:, self.n_past_classes:]), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.allx = dataset.train_loader.dataset.data
            self.ally = dataset.train_loader.dataset.targets
        else:
            self.allx = np.concatenate((self.allx, dataset.train_loader.dataset.data))
            self.ally = np.concatenate((self.ally, dataset.train_loader.dataset.targets))

        icarl_replay(self, dataset)
        self.net.train()

    def end_task(self, dataset) -> None:
        self.net.eval()
        self.old_net = deepcopy(self.net)
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=True, normalize_features=True,
                        extend_equalize_buffer=self.args.compute_theoretical_best)

            if self.args.compute_theoretical_best:
                fill_buffer(self.ncm_buffer, dataset, self.current_task, net=self.net, use_herding=True, normalize_features=True,
                            extend_equalize_buffer=self.args.compute_theoretical_best)
        self.class_means = None

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        was_training = self.net.training
        self.net.eval()
        buffer = self.ncm_buffer if self.args.compute_theoretical_best else self.buffer
        transform = self.dataset.get_normalization_transform()
        class_means = []
        buf_data = buffer.get_all_data(transform, device=self.device)
        examples, labels = buf_data[0], buf_data[1]
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = []
                allt_flipped = []
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]

                    feats = self.net(batch, returnt='features')
                    flipped_feats = self.net(TF.hflip(batch), returnt='features')
                    feats = feats / feats.norm(dim=1, keepdim=True)
                    flipped_feats = flipped_feats / flipped_feats.norm(dim=1, keepdim=True)

                    allt.append(feats)
                    allt_flipped.append(flipped_feats)
                allt = torch.cat(allt).mean(0)
                allt_flipped = torch.cat(allt_flipped).mean(0)
                allt = (allt + allt_flipped) / 2
                allt = allt / allt.norm()
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)
        self.net.train(was_training)
