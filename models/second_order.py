# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
import os
import warnings
import torch
from collections import defaultdict
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from copy import deepcopy
from torch.utils.data import DataLoader

from datasets import get_dataset_class

from models.utils.continual_model import ContinualModel
from models.lora_prototype_utils.lora_prompt import Model
from models.lora_prototype_utils.utils import create_optimizer
from models.lora_prototype_utils.utils import get_dist
from models.lora_prototype_utils.utils import AlignmentLoss
from models.lora_prototype_utils.utils import linear_probing_epoch

from utils import none_or_float
from utils.schedulers import CosineSchedule


def add_slca_args(parser, defaults=None):

    if defaults is None:
        defaults = {
            'use_ca': 1,
            'distr_alignment': 'gaussian',
            'num_epochs_incremental_alignment': 0,
            'learning_rate_alignment': 0.01,
        }
    else:
        assert 'use_ca' in defaults.keys()
        assert 'distr_alignment' in defaults.keys()
        assert 'num_epochs_incremental_alignment' in defaults.keys()
        assert 'learning_rate_alignment' in defaults.keys()

    # ---- CA
    parser.add_argument("--use_ca", type=int,
                        default=defaults['use_ca'], choices=[0, 1],
                        help="use Classifier Alignment")

    parser.add_argument("--perform_recall_aligment", type=int, default=1, choices=[0, 1])

    parser.add_argument("--num_monte_carlo_alignments", type=int, default=1,
                        help="how many times to sample from the dataset for alignment")

    parser.add_argument('--distr_alignment', type=str, default=defaults['distr_alignment'],
                        choices=['gaussian', 'mog', 'full_gaussian'])

    parser.add_argument("--num_samples_alignment", type=int, default=256,
                        help="Num. of samples from each gaussian.")
    parser.add_argument("--batch_size_alignment", type=int, default=128,
                        help="Batch size for CA.")
    parser.add_argument("--num_epochs_alignment", type=int, default=10,
                        help="Num. of epochs for CA.")
    parser.add_argument("--learning_rate_alignment", type=float,
                        default=defaults['learning_rate_alignment'],
                        help="Learning rate for CA.")
    parser.add_argument("--momentum_alignment", type=float, default=0.0,
                        help="Learning rate for CA. Original paper 0.9.")
    parser.add_argument("--learning_rate_scheduler_alignment", type=str, default='none',
                        choices=['none', 'cosine'], help="Lr scheduler for CA.")
    parser.add_argument('--decay_means_alignment', type=int, choices=[0, 1], default=0,
                        help='Apply time-dependent decay on means for CA. Original paper 1.')
    parser.add_argument('--weight_decay_alignment', type=float, default=0.0,
                        help="Weight decay param for CA. Original paper 5e-4.")
    parser.add_argument("--tau_alignment", type=float, default=0.1,
                        help="Temperature for CA.")

    parser.add_argument('--ca_mog_n_components', type=int, default=5,
                        help="Number of components for CA with MOG.")
    parser.add_argument('--ca_mog_n_iters', type=int, default=500,
                        help="Number of EM iterations during fit for CA with MOG.")

    parser.add_argument("--num_epochs_incremental_alignment", type=int,
                        default=defaults['num_epochs_incremental_alignment'],
                        help="Num. of epochs for CA.")

    parser.add_argument('--norm_type_alignment', type=str, default='all', choices=['all', 'pertask'],
                        help="How to compute normalization for CA. Original paper pertask.")

    parser.add_argument('--use_slca_opt', type=int, choices=[0, 1],
                        default=0, help='use prompts for values')
    parser.add_argument('--slca_lr_scaler', type=float, default=0.01)

    return parser


class FeaturesDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        # Your code
        self.X, self.y = X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


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

        slca_default = {
            'use_ca': 0,
            'distr_alignment': 'mog',
            'num_epochs_incremental_alignment': 5,
            'learning_rate_alignment': 0.01,
        }

        add_slca_args(parser, slca_default)

        # OPTIM PARAMS
        parser.add_argument("--virtual_bs_n", type=int, default=1,
                            help="virtual batch size iterations")
        parser.add_argument('--clip_grad', type=none_or_float, default=100,
                            help='Clip gradient norm (default: None, no clipping)')
        parser.add_argument('--sched_type', type=str, default='cosine',
                            choices=['cosine', 'cosine_new', 'cosine_warmup'])

        # BACKBONE
        parser.add_argument("--backbone_type", type=str,
                            default='ViT-B/16', choices=['ViT-B/16', 'clip-ViT-B/16'],
                            help="backbone type")
        parser.add_argument('--pret', type=str, default='in21k',
                            choices=['in21k_ft_in1k', 'in21k'])
        parser.add_argument('--load_from_pret', type=str, default=None,)
        parser.add_argument('--train_classifier', type=int, choices=[0, 1], default=1)

        # EWC REG
        parser.add_argument('--ewc_ensemble_mode', type=int,
                            choices=[0, 1], required=True)
        parser.add_argument('--ewc_optim_style', type=str, default='adam',
                            choices=['adam', 'adamw'])
        parser.add_argument('--lrw_scaler', type=float, default=1.0)
        parser.add_argument('--fisher_type', type=str,
                            default="hooks", choices=["autograd", "hooks"])

        # LoRA
        parser.add_argument('--lora_style', type=str, default='task',
                            choices=['task', 'full', 'ia3'], required=True)
        parser.add_argument('--lora_momentum', type=float, default=0.9)

        parser.add_argument('--lora_r', type=int, default=16)

        # PRE-TUNING
        parser.add_argument('--do_align_pretuning', type=int, choices=[0, 1], default=1)
        parser.add_argument('--num_epochs_pretuning', type=int, default=3)
        parser.add_argument("--learning_rate_pretuning", type=float, default=0.01)
        parser.add_argument('--pretrain_heads', type=int, choices=[0, 1], default=0)

        parser.add_argument('--ewc_fisher_mc_classes', type=int_or_all, default='all')

        parser.add_argument('--ewc_lambda', type=float, default=0.0)
        parser.add_argument('--ewc_cls_lambda', type=float, default=0.0)
        parser.add_argument('--ewc_cls_prior', type=float, default=0.0)
        parser.add_argument('--ewc_cls_identity_lambda', type=float, default=0.0)
        parser.add_argument('--ewc_cls_strategy', type=str, default="generative", choices=["generative"])

        parser.add_argument('--ewc_scale_by_t', type=int, choices=[0, 1], default=0)

        parser.add_argument('--augmented_reg', type=int, choices=[0, 1], required=True)
        parser.add_argument('--ewc_alpha', type=float, default=0.0)
        parser.add_argument('--ewc_prior', type=float, default=1e-5)

        parser.add_argument('--is_training_while_ca', type=int, choices=[0, 1], default=0)

        #########################
        # UNIMPORTANT PARAMS

        parser.add_argument("--force_fp32", type=int, default=1, choices=[0, 1],
                            help="force FP32 train (overrides use_grad_scaler)")
        parser.add_argument('--adapt_clip', type=int, choices=[0, 1], default=0)

        # CLIP MODEL (USEFUL FOR THE MODEL WITH ONE LORA FOR EACH CLASS)
        parser.add_argument("--clip_backbone_type", type=str,
                            choices=[None, 'ViT-B/16', 'ViT-L/14'],
                            default=None, help="clip backbone type")

        parser.add_argument('--use_templates', type=int,
                            choices=[0, 1], default=0, help='use prompt templates')
        parser.add_argument('--keys_ckpt_path', type=str,
                            default=None, help='checkpoint path for keys')
        parser.add_argument('--auto_load_keys', type=int, default=0, choices=[0, 1])

        # TRAINING AS OPEN CLASSIFIER
        parser.add_argument('--semantic_classifier', type=int, choices=[0, 1], default=0)
        parser.add_argument('--tau', type=float, default=20.)

        parser.add_argument('--mask_group_specialize_acc', type=int, default=0, choices=[0, 1, 2])
        return parser

    def __init__(self, backbone, loss, args, transform, dataset):
        if args.ewc_fisher_mc_classes == 'all':
            dset_cls = get_dataset_class(args)
            args.ewc_fisher_mc_classes = dset_cls.N_CLASSES_PER_TASK * dset_cls.N_TASKS if isinstance(dset_cls.N_CLASSES_PER_TASK, int) else sum(dset_cls.N_CLASSES_PER_TASK)
        else:
            args.ewc_fisher_mc_classes = int(args.ewc_fisher_mc_classes)
        args.use_grad_scaler = args.use_grad_scaler if args.force_fp32 == 0 else 0

        backbone = Model(args, dataset)
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.net.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        self.output_dim = backbone.get_output_dim()

        def self_distr():
            return get_dist(self.output_dim, self.args.distr_alignment,
                            self.args.ca_mog_n_components, self.args.ca_mog_n_iters)

        distributions = [self_distr() for _ in range(self.num_classes)]
        self.distributions = torch.nn.ModuleList(distributions).to(self.device)

        pretrain_distributions = [self_distr() for _ in range(self.num_classes)]
        self.pretrain_distributions = torch.nn.ModuleList(pretrain_distributions).to(self.device)

        self.old_epoch, self.iteration = 0, 0
        self.scheduler = None

        self.use_wandb = self.args.nowand == 0

        self.alignment_loss = AlignmentLoss(args, self.dataset, self.device)
        self.pretraining_classifier = deepcopy(self.net.vit.head)

        assert self.args.ewc_lambda >= 0.

        self.buffergrad = None
        self.buffergrad_cls = None
        self.ewc_lambda = self.args.ewc_lambda
        self.ewc_alpha = self.args.ewc_alpha
        self.ewc_cls_lambda = self.args.ewc_cls_lambda

        self.ewc_loss_is_active = False

        if self.ewc_lambda > 0.:
            self.ewc_loss_is_active = True
        if self.ewc_alpha > 0.:
            assert self.args.augmented_reg == 1
            self.ewc_loss_is_active = True

        self.ewc_loss_cls_is_active = self.ewc_cls_lambda > 0.

        # todo: remove, mantained only for retrocompatibility
        # assert args.pretrain_heads == 0
        assert args.lrw_scaler == 1.0
        assert not hasattr(args, 'ewc_cls_strategy') or args.ewc_cls_strategy == 'generative'

        if self.args.load_from_pret:
            assert os.path.exists(self.args.load_from_pret)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = torch.load(self.args.load_from_pret, map_location='cpu')
                st = {k.replace('net.', 'vit.'): v for k, v in st.items()}
                missing, unexp = self.net.load_state_dict(st, strict=False)
                assert len([m for m in missing if 'head' not in m and 'lorer' not in m and 'fisher' not in m]) == 0
                assert len([u for u in unexp if 'head' not in u]) == 0

                self.pt_classifier = (st['vit.head.weight'], st['vit.head.bias'])

    @torch.no_grad()
    def create_synthetic_features_dataset(self, distributions_to_sample_from=None, upto: int = None):

        labels, features = [], []
        if upto is None:
            upto = self.current_task + 1
        else:
            assert isinstance(upto, int)
        num_samples_per_class = self.args.num_samples_alignment

        if distributions_to_sample_from is None:
            distributions_to_sample_from = self.distributions

        for _ti in range(upto):

            prev_t_size, cur_t_size = self.dataset.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):

                scale_mean = (_ti + 1) / (self.current_task + 1) * 0.1 \
                    if self.args.decay_means_alignment == 1 else 1.0

                current_samples = distributions_to_sample_from[class_idx](num_samples_per_class, scale_mean)
                features.append(current_samples)
                labels.append(torch.ones((num_samples_per_class,)) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return DataLoader(FeaturesDataset(features, labels),
                          batch_size=self.args.batch_size_alignment, shuffle=True,
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
                          batch_size=self.args.batch_size_alignment, shuffle=True,
                          num_workers=0, drop_last=True)

    @torch.no_grad()
    def compute_statistics(self, dataset, distributions, use_lora):

        features_dict = defaultdict(list)
        is_training_while_ca = True if self.args.is_training_while_ca == 1 else False

        orig_mode = self.net.training
        self.net.eval()

        for _ in tqdm(range(self.args.num_monte_carlo_alignments),
                      total=self.args.num_monte_carlo_alignments,
                      desc='CA update statistics'):

            for i, data in enumerate(dataset.train_loader):

                if self.args.debug_mode and i > 101:
                    break

                x, labels, _ = data
                x, labels = x.to(self.device), labels.to(self.device).long()

                features = self.net(x, train=is_training_while_ca,
                                    return_features=True, use_lora=use_lora)
                features = features[:, 0]

                for class_idx in labels.unique():
                    features_dict[int(class_idx)].append(features[labels == class_idx])

        self.net.train(orig_mode)

        for class_idx in features_dict.keys():
            features_class_idx = torch.cat(features_dict[class_idx], dim=0).to(self.device)
            distributions[class_idx].fit(features_class_idx)

    def logger_fn(self, loss_dict):
        # todo: remove the False
        if self.use_wandb and False:
            wandb.log(loss_dict)

    def sgd(self, cls, lr=None, momentum=None, wd=None):

        if lr is None:
            lr = self.args.learning_rate_alignment
        if momentum is None:
            momentum = self.args.momentum_alignment
        if wd is None:
            wd = self.args.weight_decay_alignment

        params = cls.build_optimizer_args(lr, wd)
        return torch.optim.SGD(lr=lr, params=params,
                               momentum=momentum, weight_decay=wd)

    def sched(self, optim, num_epochs: int, sched_type: str = None):
        if sched_type == 'cosine':
            return CosineAnnealingLR(optimizer=optim, T_max=num_epochs)
        return None

    def align(self, cls, distributions_to_sample_from, desc=''):
        optim = self.sgd(cls)
        num_epochs = self.args.num_epochs_alignment + \
            self.args.num_epochs_incremental_alignment * self.current_task

        lr_scheduler = self.sched(optim, num_epochs,
                                  self.args.learning_rate_scheduler_alignment)

        for _ in tqdm(range(num_epochs), total=num_epochs, desc=desc):
            data_loader = self.create_synthetic_features_dataset(
                distributions_to_sample_from=distributions_to_sample_from)
            linear_probing_epoch(data_loader, self.alignment_loss, cls,
                                 optim, lr_scheduler, self.device, self.logger_fn)
        return cls

    def ca(self):
        classifier = deepcopy(self.net.vit.head)
        classifier.enable_training()
        self.align(classifier, self.distributions,
                   desc='Classifier Alignment (end)')
        self.net.vit.head.assign(classifier)

    def backup(self):
        print(f"BACKUP: Task - {self.current_task} - classes from "
              f"{self.n_past_classes} - to {self.n_seen_classes}")
        self.net.vit.head.backup()

    def recall(self):
        print(f"RECALL: Task - {self.current_task} - classes from "
              f"{self.n_past_classes} - to {self.n_seen_classes}")
        if (not self.args.use_ca) or (self.current_task == 0) \
                or (not self.args.perform_recall_aligment):
            return
        self.net.vit.head.recall()

    def masked_loss(self, cls, x, labels):
        logits = cls(x)
        logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(logits, labels)
        loss_val = loss.detach().item()
        return loss, {'ce_pretuning': loss_val}

    def linear_probing(self, dataset, classifier, lr, num_epochs,
                       desc='', use_lora: bool = False):

        optim = self.sgd(classifier, lr=lr)

        for _ in tqdm(range(num_epochs), total=num_epochs, desc=desc):
            data_loader = self.create_features_dataset(dataset.train_loader,
                                                       use_lora=use_lora)
            linear_probing_epoch(data_loader, self.masked_loss, classifier,
                                 optim, None, self.device, self.logger_fn, debug_mode=self.args.debug_mode == 1)

        return classifier

    def pretuning(self, dataset):
        self.compute_statistics(dataset, self.pretrain_distributions,
                                use_lora=False)
        if self.args.load_from_pret:
            return

        lr = self.args.learning_rate_pretuning
        num_epochs = self.args.num_epochs_pretuning

        classifier = deepcopy(self.pretraining_classifier)
        classifier.enable_training()
        classifier = self.linear_probing(dataset, classifier, lr, num_epochs,
                                         desc='Pre-Tuning - Task-IL (begin)',
                                         use_lora=False)

        if self.args.do_align_pretuning == 1:
            self.align(classifier, self.pretrain_distributions,
                       desc='Pre-Tuning - Class-IL (begin)')

        self.pretraining_classifier.assign(classifier)

    def get_optimizer(self):
        lr_backbone = self.args.lr * self.args.lrw_scaler
        optimizer_arg = self.net.build_optimizer_args(lr_backbone, self.args.lr,
                                                      self.args.optim_wd, 0.)
        return create_optimizer(self.args.optimizer, optimizer_arg,
                                momentum=self.args.lora_momentum)

    def get_scheduler(self, sched_type='cosine'):
        if sched_type == 'cosine':
            return CosineSchedule(self.opt, K=self.args.n_epochs)
        if sched_type == 'cosine_new':
            return CosineAnnealingLR(optimizer=self.opt, T_max=self.args.n_epochs)
        elif sched_type == 'cosine_warmup':
            change_epoch = int(0.2 * self.args.n_epochs)
            sched1 = LinearLR(self.opt, start_factor=0.1, total_iters=change_epoch)
            sched2 = CosineAnnealingLR(optimizer=self.opt, T_max=(self.args.n_epochs - change_epoch))
            return SequentialLR(self.opt, [sched1, sched2], milestones=[change_epoch])
        return None

    def update_statistics(self, dataset):
        self.net.vit.head.backup()
        self.net.vit.head.assign(self.pretraining_classifier)

        def log_fisher_norm(fisher_dict):
            wandb.log(fisher_dict)

        logger_fn = log_fisher_norm if self.use_wandb else None

        generative_dataloader = None
        if self.args.ewc_cls_strategy == "generative":
            if self.args.ewc_ensemble_mode == 0:
                if self.current_task > 0:
                    assert self.args.do_align_pretuning == 1
                    generative_dataloader = self.create_synthetic_features_dataset(self.pretrain_distributions, self.current_task)
            else:
                assert self.args.do_align_pretuning == 1
                generative_dataloader = self.create_synthetic_features_dataset(self.pretrain_distributions, self.current_task + 1)

        self.net.update_fisher(dataset, generative_dataloader, logger_fn, self.args.debug_mode == 1)
        self.net.vit.head.recall()

    def begin_task(self, dataset):
        self.recall()

        if self.current_task > 0:
            self.pretraining_classifier.update(nb_classes=self.n_classes_current_task)
            self.net.vit.head.update(nb_classes=self.n_classes_current_task)

        self.alignment_loss.set_current_task(self.current_task)
        self.net.set_current_task(self.current_task)

        if not hasattr(self.args, 'load_only') or not self.args.load_only:
            self.pretuning(dataset)

        if self.args.load_from_pret:
            self.pretraining_classifier.heads[self.current_task].weight.data.copy_(self.pt_classifier[0][self.n_past_classes:self.n_seen_classes])
            self.pretraining_classifier.heads[self.current_task].bias.data.copy_(self.pt_classifier[1][self.n_past_classes:self.n_seen_classes])

        if self.args.pretrain_heads == 1:
            self.net.vit.head.assign(self.pretraining_classifier,
                                     which_heads=[self.current_task])

        if not hasattr(self.args, 'load_only') or not self.args.load_only:
            self.net.vit.head.requires_grad_(True)
            self.update_statistics(dataset)

        if self.args.load_from_pret:
            self.net.vit.head.requires_grad_(self.args.train_classifier == 1)

        if hasattr(self, 'opt'):
            self.opt.zero_grad()
            del self.opt

        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.args.sched_type)
        self.old_epoch, self.iteration = 0, 0

        if self.buffergrad is not None:
            del self.buffergrad
        if self.buffergrad_cls is not None:
            del self.buffergrad_cls

        self.buffergrad = [torch.zeros_like(p)
                           for p in self.opt.param_groups[0]['params']]
        self.buffergrad_cls = [torch.zeros_like(p)
                               for p in self.opt.param_groups[1]['params']]

        self.net.ensemble(self.args.ewc_ensemble_mode == 1)
        torch.cuda.empty_cache()

    def update_prevtask_statistics(self, dataset):
        assert self.args.ewc_ensemble_mode == 1
        self.net.merge_lora()
        self.net.update_prevtask_fisher(dataset, self.args.debug_mode == 1)
        self.net.unmerge_lora()

    def end_task(self, dataset):

        if self.args.ewc_ensemble_mode == 1 \
                and self.args.augmented_reg == 1:
            self.update_prevtask_statistics(dataset)

        self.net.ensemble(True)

        if self.args.use_ca:
            if not self.args.load_from_pret:
                self.compute_statistics(dataset, self.distributions, use_lora=True)

                self.backup()
                if self.current_task > 0 or not self.args.perform_recall_aligment:
                    self.ca()

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

    def _grad_clipping(self, param_group, clip_grad: float):
        for p in param_group:
            torch.nn.utils.clip_grad_norm_(p, clip_grad)

    def _apply_grads(self, lr, param_group, clip_grad):
        for myparam in param_group:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(myparam, clip_grad)
            myparam.add_(myparam.grad, alpha=-lr)
            myparam.grad.zero_()

    def _grad_flip_sign(self, param_group):
        for myparam in param_group:
            myparam.mul_(-1.)

    @torch.no_grad()
    def grad_flip_sign(self, param_group: str):
        assert param_group in ['layers']
        id_group = {'layers': 0, 'cls': 1}[param_group]
        self._grad_flip_sign(self.opt.param_groups[id_group]['params'])

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
    def gradient_clipping(self, param_group: str):
        assert param_group in ['layers', 'cls']
        if self.args.clip_grad is None:
            return
        id_group = {'layers': 0, 'cls': 1}[param_group]
        self._grad_clipping(self.opt.param_groups[id_group]['params'],
                            self.args.clip_grad)

    @torch.no_grad()
    def apply_grads(self, param_group: str):
        assert param_group in ['layers', 'cls']
        id_group = {'layers': 0, 'cls': 1}[param_group]

        lr = 1.0

        if self.scheduler is not None:
            if self.args.sched_type == 'cosine':
                base_lr = self.scheduler.base_lrs[id_group]
                lr = self.scheduler.get_lr()[id_group] / base_lr
            elif self.args.sched_type == 'cosine_warmup':
                base_lr = self.scheduler._schedulers[1].base_lrs[id_group]
                lr = self.scheduler.get_last_lr()[id_group] / base_lr
            else:
                base_lr = self.scheduler.base_lrs[id_group]
                lr = self.scheduler.get_last_lr()[id_group] / base_lr

        self._apply_grads(lr, self.opt.param_groups[id_group]['params'],
                          clip_grad=self.args.clip_grad)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()

        if self.scheduler and self.old_epoch != epoch:
            if epoch > 0:
                self.scheduler.step()
            self.old_epoch = epoch
            self.iteration = 0

        self.net.iteration = self.iteration

        log_dict = {}

        stream_inputs, stream_labels = inputs, labels
        stream_logits = self.net(stream_inputs, train=True)

        with torch.no_grad():
            log_dict['stream_class_il'] = self.accuracy(stream_logits, stream_labels)

        stream_logits[:, :self.n_past_classes] = -float('inf')

        loss = self.compute_loss(stream_logits, stream_labels)

        with torch.no_grad():
            log_dict['stream_task_il'] = self.accuracy(stream_logits, stream_labels)

        if self.iteration == 0:
            self.opt.zero_grad()

        if self.args.virtual_bs_n > 1:
            loss = loss / self.args.virtual_bs_n

        loss.backward()

        if (self.iteration > 0 or self.args.virtual_bs_n == 1) and \
                self.iteration % self.args.virtual_bs_n == 0:

            if self.ewc_loss_is_active:
                self.grad_backup('layers')

            if self.ewc_loss_cls_is_active:
                self.grad_backup('cls')

            with torch.set_grad_enabled(self.ewc_loss_is_active):
                reg_loss, dotprod_loss = \
                    self.net.compute_reg_loss(do_backward=self.ewc_loss_is_active,
                                              do_loss_computation=True)

            if self.args.train_classifier:
                with torch.set_grad_enabled(self.ewc_loss_cls_is_active):
                    reg_cls = self.net.compute_classifier_reg_loss(
                        cls_ref=self.pretraining_classifier,
                        do_backward=self.ewc_loss_cls_is_active)
            else:
                reg_cls = torch.tensor(0.)

            with torch.no_grad():
                log_dict['reg_loss'] = reg_loss.detach()
                log_dict['dotprod_loss'] = dotprod_loss.detach()
                log_dict['reg_cls'] = reg_cls.detach()

            if self.ewc_loss_is_active:
                if self.args.ewc_optim_style == 'adam':
                    self.grad_recall('layers', 'add')
                    self.gradient_clipping('layers')
                elif self.args.ewc_optim_style == 'adamw':
                    self.apply_grads('layers')
                    self.grad_recall('layers')

            if self.ewc_loss_cls_is_active:
                self.apply_grads('cls')
                self.grad_recall('cls')

            self.opt.step()
            self.opt.zero_grad()

        self.iteration += 1

        if self.use_wandb:
            wandb.log(log_dict)

        return loss.item()
