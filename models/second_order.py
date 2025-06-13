# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from argparse import ArgumentParser
from copy import deepcopy
from tqdm.auto import tqdm

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils import binary_to_boolean_type, none_or_float
from utils.schedulers import CosineSchedule

from datasets import get_dataset_class

from models.utils.continual_model import ContinualModel
from models.lora_prototype_utils.lora_prompt import Model
from models.lora_prototype_utils.generative_replay import FeaturesDataset
from models.lora_prototype_utils.utils import create_optimizer
from models.lora_prototype_utils.utils import get_dist
from models.lora_prototype_utils.utils import AlignmentLoss
from models.lora_prototype_utils.utils import linear_probing_epoch


def int_or_all(x):
    if x == 'all':
        return x
    return str(x)


class SecondOrder(ContinualModel):

    NAME = 'second_order'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    net: Model

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        parser.set_defaults(pretrain_type='in21k', optimizer='adamw')

        # OPTIM PARAMS
        parser.add_argument("--virtual_bs_n", type=int, default=1,
                            help="Virtual batch size iterations")
        parser.add_argument('--clip_grad', type=none_or_float, default=100,
                            help='Clip gradient norm (None means no clipping)')

        # FINE-TUNING PARAMS
        parser.add_argument('--tuning_style', type=str, default='lora',
                            choices=['lora', 'full', 'ia3'],
                            help='Strategy to use for tuning the model.\n'
                            '- "lora": LoRA\n'
                            '- "full": full fine-tuning\n'
                            '- "ia3": IA3')
        parser.add_argument('--lora_r', type=int, default=16,
                            help='LoRA rank. Used if `tuning_style` is "lora".')

        # PRE-TUNING
        parser.add_argument('--num_epochs_pretuning', type=int, default=3,
                            help='Number of epochs for pre-tuning')
        parser.add_argument("--learning_rate_pretuning", type=float, default=0.01,
                            help="Learning rate for pre-tuning.")

        parser.add_argument('--fisher_mc_classes', type=int_or_all, default='all',
                            help='Number of classes to use for EWC Fisher computation.\n'
                            '- "all": slow but accurate, uses all classes\n'
                            '- <int>: use subset of <int> classes, faster but less accurate')
        parser.add_argument("--num_samples_align_pretuning", type=int, default=256,
                            help="Num. of samples from each gaussian.")
        parser.add_argument("--batch_size_align_pretuning", type=int, default=128,
                            help="Batch size for CA.")
        parser.add_argument("--num_epochs_align_pretuning", type=int, default=10,
                            help="Num. of epochs for CA.")
        parser.add_argument("--lr_align_pretuning", type=float,
                            default=0.01, help="Learning rate for CA.")

        # REGULARIZATION PARAMS
        parser.add_argument('--use_iel', type=binary_to_boolean_type, choices=[0, 1], default=0,
                            help="Tune with ITA or IEL")
        # IEL
        parser.add_argument('--beta_iel', type=float, default=0.0, help="Beta parameter of IEL (Eq. 18/19)")
        # ITA
        parser.add_argument('--alpha_ita', type=float, default=0.0, help="Alpha parameter of ITA (Eq. 11)")
        parser.add_argument('--req_weight_cls', type=float,
                            help="Regularization weight (alpha for ITA/beta for IEL) for classifier. "
                            "If None, will use the alpha/beta of ITA/IEL.")
        parser.add_argument('--simple_reg_weight_cls', type=float, default=0.0,
                            help="Regularization weight for simple MSE-based loss for the classifier.")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.fisher_mc_classes == 'all':
            dset_cls = get_dataset_class(args)
            args.fisher_mc_classes = dset_cls.N_CLASSES
        else:
            args.fisher_mc_classes = int(args.fisher_mc_classes)

        assert args.beta_iel >= 0., "Beta parameter of IEL must be >= 0"
        assert args.alpha_ita >= 0., "Alpha parameter of ITA must be >= 0"

        args.req_weight_cls = args.req_weight_cls if args.req_weight_cls is not None else \
            (args.beta_iel if args.use_iel else args.alpha_ita)

        backbone = Model(args, dataset, backbone)
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.output_dim = backbone.output_dim

        distributions = [get_dist(self.output_dim) for _ in range(self.num_classes)]
        self.distributions = torch.nn.ModuleList(distributions).to(self.device)

        pretrain_distributions = [get_dist(self.output_dim) for _ in range(self.num_classes)]
        self.pretrain_distributions = torch.nn.ModuleList(pretrain_distributions).to(self.device)

        self.old_epoch, self.iteration = 0, 0
        self.custom_scheduler = self.get_scheduler()
        self.alignment_loss = AlignmentLoss(self.dataset, self.device)
        self.pretraining_classifier = deepcopy(self.net.vit.head)

        self.buffergrad = None
        self.buffergrad_cls = None
        self.beta_iel = self.args.beta_iel
        self.alpha_ita = self.args.alpha_ita
        self.req_weight_cls = self.args.req_weight_cls

        self.reg_loss_is_active = self.beta_iel > 0. or self.alpha_ita > 0.
        self.reg_loss_cls_is_active = self.req_weight_cls > 0.

    @torch.no_grad()
    def create_synthetic_features_dataset(self, distributions_to_sample_from=None, upto: int = None):

        labels, features = [], []
        if upto is None:
            upto = self.current_task + 1
        else:
            assert isinstance(upto, int)
        num_samples_per_class = self.args.num_samples_align_pretuning

        if distributions_to_sample_from is None:
            distributions_to_sample_from = self.distributions

        for _ti in range(upto):

            prev_t_size, cur_t_size = self.dataset.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):

                current_samples = distributions_to_sample_from[class_idx](num_samples_per_class, 1.0)
                features.append(current_samples)
                labels.append(torch.ones((num_samples_per_class,)) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return DataLoader(FeaturesDataset(features, labels),
                          batch_size=self.args.batch_size_align_pretuning, shuffle=True,
                          num_workers=0, drop_last=True)

    @torch.no_grad()
    def create_features_dataset(self, data_loader, use_lora: bool):

        labels, features = [], []

        orig_mode = self.net.training
        self.net.eval()

        for i, data in enumerate(data_loader):

            if self.args.debug_mode and i > 101:
                break

            x, y, _ = data
            x, y = x.to(self.device), y.to(self.device).long()

            z = self.net(x, train=False, return_features=True,
                         use_lora=use_lora)
            z = z[:, 0]
            features.append(z.detach().cpu())
            labels.append(y.detach().cpu())

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        self.net.train(orig_mode)

        return DataLoader(FeaturesDataset(features, labels),
                          batch_size=self.args.batch_size_align_pretuning, shuffle=True,
                          num_workers=0, drop_last=True)

    @torch.no_grad()
    def compute_statistics(self, dataset, distributions, use_lora):

        features_dict = defaultdict(list)
        orig_mode = self.net.training
        self.net.eval()

        for i, data in enumerate(dataset.train_loader):

            if self.args.debug_mode and i > 101:
                break

            x, labels, _ = data
            x, labels = x.to(self.device), labels.to(self.device).long()

            features = self.net(x, train=False, return_features=True, use_lora=use_lora)
            features = features[:, 0]

            for class_idx in labels.unique():
                features_dict[int(class_idx)].append(features[labels == class_idx])

        self.net.train(orig_mode)

        for class_idx in features_dict.keys():
            features_class_idx = torch.cat(features_dict[class_idx], dim=0).to(self.device)
            distributions[class_idx].fit(features_class_idx)

    def get_sgd_optim(self, cls, lr):
        params = cls.build_optimizer_args(lr)
        return torch.optim.SGD(lr=lr, params=params)

    def sched(self, optim, num_epochs: int):
        return CosineAnnealingLR(optimizer=optim, T_max=num_epochs)

    def align_pretuning(self, cls, distributions_to_sample_from, desc=''):
        optim = self.get_sgd_optim(cls, lr=self.args.lr_align_pretuning)
        num_epochs = self.args.num_epochs_align_pretuning + 5 * self.current_task

        lr_scheduler = self.sched(optim, num_epochs)

        for _ in tqdm(range(num_epochs), total=num_epochs, desc=desc):
            data_loader = self.create_synthetic_features_dataset(
                distributions_to_sample_from=distributions_to_sample_from)
            linear_probing_epoch(data_loader, self.alignment_loss, cls,
                                 optim, lr_scheduler, self.device)
        return cls

    def masked_loss(self, cls, x, labels):
        """
        Separate losses for current and previous tasks.
        """
        logits = cls(x)
        logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(logits, labels)
        loss_val = loss.detach().item()
        return loss, {'ce_pretuning': loss_val}

    def linear_probing(self, dataset, classifier, lr, num_epochs,
                       desc='', use_lora: bool = False):

        optim = self.get_sgd_optim(classifier, lr=lr)

        for _ in tqdm(range(num_epochs), total=num_epochs, desc=desc):
            data_loader = self.create_features_dataset(dataset.train_loader,
                                                       use_lora=use_lora)
            linear_probing_epoch(data_loader, self.masked_loss, classifier,
                                 optim, None, self.device, debug_mode=self.args.debug_mode == 1)

        return classifier

    def pretuning(self, dataset):
        self.compute_statistics(dataset, self.pretrain_distributions,
                                use_lora=False)

        lr = self.args.learning_rate_pretuning
        num_epochs = self.args.num_epochs_pretuning

        classifier = deepcopy(self.pretraining_classifier)
        classifier.enable_training()
        classifier = self.linear_probing(dataset, classifier, lr, num_epochs,
                                         desc='Pre-Tuning - Task-IL (begin)',
                                         use_lora=False)

        self.align_pretuning(classifier, self.pretrain_distributions,
                             desc='Pre-Tuning - Class-IL (begin)')

        self.pretraining_classifier.assign(classifier)

    def get_optimizer(self):
        optimizer_arg = self.net.build_optimizer_args(self.args.lr)
        return create_optimizer(self.args.optimizer, optimizer_arg, momentum=0.9)

    def get_scheduler(self):
        return CosineSchedule(self.opt, K=self.args.n_epochs)

    def update_statistics(self, dataset):
        self.net.vit.head.backup()
        self.net.vit.head.assign(self.pretraining_classifier)

        generative_dataloader = None
        if self.args.use_iel:
            if self.current_task > 0:
                generative_dataloader = self.create_synthetic_features_dataset(self.pretrain_distributions, self.current_task)
        else:
            generative_dataloader = self.create_synthetic_features_dataset(self.pretrain_distributions, self.current_task + 1)

        self.net.update_fisher(dataset, generative_dataloader, self.args.debug_mode == 1)
        self.net.vit.head.recall()

    def begin_task(self, dataset):
        num_classes = self.n_classes_current_task

        if self.current_task > 0:
            self.pretraining_classifier.update(nb_classes=num_classes)
            self.net.vit.head.update(nb_classes=num_classes)

        self.alignment_loss.set_current_task(self.current_task)
        self.net.set_current_task(self.current_task)

        self.pretuning(dataset)

        self.update_statistics(dataset)

        if hasattr(self, 'opt'):
            self.opt.zero_grad()
            del self.opt

        self.opt = self.get_optimizer()
        self.custom_scheduler = self.get_scheduler()
        self.old_epoch, self.iteration = 0, 0

        if self.buffergrad is not None:
            del self.buffergrad
        if self.buffergrad_cls is not None:
            del self.buffergrad_cls

        self.buffergrad = [torch.zeros_like(p)
                           for p in self.opt.param_groups[0]['params']]
        self.buffergrad_cls = [torch.zeros_like(p)
                               for p in self.opt.param_groups[1]['params']]

        # Train either with ITA or IEL
        self.net.ensemble(self.args.use_iel)
        torch.cuda.empty_cache()

    def end_task(self, dataset):
        # Evaluate the merged model
        self.net.ensemble(True)

    def forward(self, x, task_weights=None, returnt='out'):
        assert returnt in ['out', 'features']
        logits = self.net(x, train=False, task_weights=task_weights, return_features=returnt == 'features')
        if returnt == 'features':
            return logits
        return logits[:, :self.n_seen_classes]

    def accuracy(self, pred, labels):
        stream_preds = pred[:, :self.n_seen_classes].argmax(dim=1)
        acc = (stream_preds == labels).sum().item() / len(labels)
        return acc

    def compute_loss(self, stream_logits, stream_labels):
        """
        Compute the loss for the current task.
        """
        stream_logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(stream_logits[:, :self.n_seen_classes], stream_labels)
        return loss

    def _grad_backup(self, param_group, buffer, set_to_zero: bool):
        for idx_p, p in enumerate(param_group):
            buffer[idx_p].copy_(p.grad)
            if set_to_zero:
                torch.nn.init.zeros_(p.grad)

    def _grad_recall(self, param_group, buffer, op):
        for idx_p in range(len(param_group)):
            if op == 'add':
                param_group[idx_p].grad.add_(buffer[idx_p])
            else:
                param_group[idx_p].grad.copy_(buffer[idx_p])

    def _apply_grads(self, lr, param_group, clip_grad):
        for myparam in param_group:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(myparam, clip_grad)
            myparam.add_(myparam.grad, alpha=-lr)
            myparam.grad.zero_()

    @torch.no_grad()
    def grad_backup(self, param_group: str):
        assert param_group in ['layers', 'cls']
        id_group = {'layers': 0, 'cls': 1}[param_group]
        set_to_zero = {'layers': False, 'cls': True}[param_group]
        id_buffer = {'layers': self.buffergrad,
                     'cls': self.buffergrad_cls}[param_group]
        self._grad_backup(self.opt.param_groups[id_group]['params'],
                          id_buffer, set_to_zero=set_to_zero)

    @torch.no_grad()
    def grad_recall(self, param_group: str, op='set'):
        assert op in ['add', 'set']
        id_group = {'layers': 0, 'cls': 1}[param_group]
        id_buffer = {'layers': self.buffergrad, 'cls': self.buffergrad_cls}[param_group]
        self._grad_recall(self.opt.param_groups[id_group]['params'], id_buffer, op)

    @torch.no_grad()
    def apply_grads(self, param_group: str):
        assert param_group in ['layers', 'cls']
        id_group = {'layers': 0, 'cls': 1}[param_group]

        lr = 1.0

        if self.custom_scheduler is not None:
            base_lr = self.custom_scheduler.base_lrs[id_group]
            lr = self.custom_scheduler.get_lr()[id_group] / base_lr

        self._apply_grads(lr, self.opt.param_groups[id_group]['params'],
                          clip_grad=self.args.clip_grad)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()

        if self.custom_scheduler and self.old_epoch != epoch:
            if epoch > 0:
                self.custom_scheduler.step()
            self.old_epoch = epoch
            self.iteration = 0

        self.net.iteration = self.iteration

        log_dict = {}

        stream_inputs, stream_labels = inputs, labels
        stream_logits = self.net(stream_inputs, train=True)

        with torch.no_grad():
            log_dict['stream_class_il'] = self.accuracy(stream_logits, stream_labels)

        stream_logits[:, :self.n_past_classes] = -float('inf')

        with torch.no_grad():
            log_dict['stream_task_il'] = self.accuracy(stream_logits, stream_labels)

        loss = self.compute_loss(stream_logits, stream_labels)

        if self.iteration == 0:
            self.opt.zero_grad()

        if self.args.virtual_bs_n > 1:
            loss = loss / self.args.virtual_bs_n

        loss.backward()

        if (self.iteration > 0 or self.args.virtual_bs_n == 1) and \
                self.iteration % self.args.virtual_bs_n == 0:

            if self.reg_loss_is_active:
                self.grad_backup('layers')

            if self.reg_loss_cls_is_active:
                self.grad_backup('cls')

            with torch.set_grad_enabled(self.reg_loss_is_active):
                reg_loss, dotprod_loss = self.net.compute_reg_loss(do_backward=self.reg_loss_is_active,
                                                                   do_loss_computation=True)

            with torch.set_grad_enabled(self.reg_loss_cls_is_active):
                reg_cls = self.net.compute_classifier_reg_loss(
                    cls_ref=self.pretraining_classifier,
                    do_backward=self.reg_loss_cls_is_active)

            with torch.no_grad():
                log_dict['reg_loss'] = reg_loss.detach()
                log_dict['dotprod_loss'] = dotprod_loss.detach()
                log_dict['reg_cls'] = reg_cls.detach()

            if self.reg_loss_is_active:
                self.apply_grads('layers')
                self.grad_recall('layers')

            if self.reg_loss_cls_is_active:
                self.apply_grads('cls')
                self.grad_recall('cls')

            self.opt.step()
            self.opt.zero_grad()

        self.iteration += 1

        log_dict['loss'] = loss.item()

        return log_dict
