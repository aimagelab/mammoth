

from utils.args import *
from models.utils.continual_model import ContinualModel

import timm
import torch
from utils.conf import get_device
from models.slca_utils.slca import SLCA_Model


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--prefix', type=str, default='reproduce')
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--memory_per_class', type=int, default=0)
    parser.add_argument('--fixed_memory', type=int, choices=[0, 1], default=0)
    parser.add_argument('--shuffle', type=int, choices=[0, 1], default=1)
    parser.add_argument('--convnet_type', type=str, default='vit-b-p16')
    parser.add_argument('--ca_epochs', type=int, default=5)
    parser.add_argument('--ca_with_logit_norm', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default='40')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--virtual_batch_size', type=int, required=False)
    return parser


class SLCA(ContinualModel):
    NAME = 'slca'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        args.milestones = args.milestones.split(',')
        self.device = get_device()
        backbone = SLCA_Model(self.device, args)
        n_features = backbone._network.feature_dim
        if args.virtual_batch_size is None:
            args.virtual_batch_size = args.batch_size
        super().__init__(backbone, loss, args, transform)
        self.class_means = torch.zeros(self.num_classes, n_features).to(self.device)
        self.class_covs = torch.zeros(self.num_classes, n_features, n_features).to(self.device)

    def get_parameters(self):
        return self.net._network.parameters()

    def end_task(self, dataset):

        self.net._network.fc.backup()

        dataset.train_loader.dataset.transform = self.dataset.TEST_TRANSFORM
        class_means, class_covs = self.net.my_compute_class_means(dataset.train_loader, self.offset_1, self.offset_2)
        for k in class_means:
            self.class_means[k] = class_means[k]
            self.class_covs[k] = class_covs[k]

        if self.current_task > 0:
            self.net._stage2_compact_classifier(self.class_means, self.class_covs, self.offset_1, self.offset_2)

    def begin_task(self, dataset):
        if self.current_task > 0:
            self.net._network.fc.recall()
        self.offset_1, self.offset_2 = self.dataset.get_offsets(self.current_task)
        self.net._cur_task += 1
        self.net._network.update_fc(self.offset_2 - self.offset_1)
        self.net._network.to(self.device)
        self.opt, self.scheduler = self.net.get_optimizer()
        self.net._network.train()

        self.old_epoch = 0
        self.opt.zero_grad()
        self._it = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.old_epoch != epoch:
            self.old_epoch = epoch
            self.scheduler.step()

        labels = labels.long()
        logits = self.net._network(inputs, bcb_no_grad=self.net.fix_bcb)['logits']
        loss = self.loss(logits[:, self.offset_1:self.offset_2], labels - self.offset_1)

        if self._it == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        loss.backward()
        if self._it > 0 and (self._it % (self.args.virtual_batch_size // self.args.batch_size) == 0 or (self.args.virtual_batch_size == self.args.batch_size)):
            self.opt.step()
            self.opt.zero_grad()
        self._it += 1

        return loss.item()

    def forward(self, x):
        logits = self.net._network(x)['logits']
        return logits[:, :self.offset_2]
