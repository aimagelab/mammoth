"""
Slow Learner with Classifier Alignment.

Note:
    SLCA USES A CUSTOM BACKBONE (see `feature_extractor_type` argument)

Arguments:
    --feature_extractor_type: the type of convnet to use. `vit-b-p16` is the default: ViT-B/16 pretrained on Imagenet 21k (**NO** finetuning on ImageNet 1k)
"""

from utils import binary_to_boolean_type
from utils.args import *
from models.utils.continual_model import ContinualModel

import torch
from utils.conf import get_device
from models.slca_utils.slca import SLCA_Model


class SLCA(ContinualModel):
    """Continual Learning via Slow Learner with Classifier Alignment."""
    NAME = 'slca'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--prefix', type=str, default='reproduce')
        parser.add_argument('--memory_size', type=int, default=0)
        parser.add_argument('--memory_per_class', type=int, default=0)
        parser.add_argument('--fixed_memory', type=binary_to_boolean_type, default=0)
        parser.add_argument(
            '--feature_extractor_type',
            type=str,
            default='vit-b-p16',
            help='the type of feature extractor to use. `vit-b-p16` is the default: '
            'ViT-B/16 pretrained on Imagenet 21k (**NO** finetuning on ImageNet 1k)')
        parser.add_argument('--ca_epochs', type=int, default=5, help='number of epochs for classifier alignment')
        parser.add_argument('--ca_with_logit_norm', type=float, default=0.1)
        parser.add_argument('--milestones', type=str, default='40')
        parser.add_argument('--lr_decay', type=float, default=0.1)
        parser.add_argument('--virtual_bs_n', '--virtual_bs_iterations', dest='virtual_bs_iterations',
                            type=int, default=1, help="virtual batch size iterations")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        self.device = get_device()
        del backbone
        print("-" * 20)
        print(f"WARNING: SLCA USES A CUSTOM BACKBONE: {args.feature_extractor_type}")
        backbone = SLCA_Model(self.device, args)
        print("-" * 20)

        args.milestones = args.milestones.split(',')
        n_features = backbone._network.convnet.feature_dim
        super().__init__(backbone, loss, args, transform, dataset=dataset)
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

        self.opt.zero_grad()

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):

        labels = labels.long()
        logits = self.net._network(inputs, bcb_no_grad=self.net.fix_bcb)['logits']
        loss = self.loss(logits[:, self.offset_1:self.offset_2], labels - self.offset_1)

        if self.task_iteration == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        (loss / self.args.virtual_bs_iterations).backward()
        if self.task_iteration > 0 and self.task_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()

    def forward(self, x):
        logits = self.net._network(x)['logits']
        return logits[:, :self.offset_2]
