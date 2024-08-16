import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from models.slca_utils.base import BaseLearner
from models.slca_utils.inc_net import FinetuneIncrementalNet
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from datasets import get_dataset
import sys


class SLCA_Model(BaseLearner):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.device = device
        self.args = args
        self._network = FinetuneIncrementalNet(args.feature_extractor_type, pretrained=True)
        self.bcb_lrscale = 1.0 / 100
        self.fix_bcb = False
        self.save_before_ca = False

        if self.args.ca_with_logit_norm > 0:
            self.logit_norm = self.args.ca_with_logit_norm
        else:
            self.logit_norm = None
        self.topk = 5

    @property
    def training(self):
        return self._network.training

    def to(self, device):
        self._network.to(device)

    def train(self, *args):
        self._network.train(*args)

    def eval(self):
        self._network.eval()

    def get_optimizer(self):
        lrate = self.args.lr
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad == True]
        head_scale = 1
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate * self.bcb_lrscale, 'weight_decay': self.args.optim_wd}
            base_fc_params = {'params': base_fc_params, 'lr': lrate * head_scale, 'weight_decay': self.args.optim_wd}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate * head_scale, 'weight_decay': self.args.optim_wd}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=self.args.optim_wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.milestones, gamma=self.args.lr_decay)
        return optimizer, scheduler

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path + '/' + self.model_prefix + '_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def my_compute_class_means(self, loader, offset_1, offset_2):
        print('Computing class means...', file=sys.stderr)
        class_vectors = {idx: [] for idx in range(offset_1, offset_2)}
        class_means, class_covs = {}, {}
        status = self._network.training
        self._network.eval()
        for data in tqdm(loader):
            imgs, labels = data[0], data[1]
            if self.args.debug_mode and all(len(class_vectors[idx]) >= 5 for idx in range(offset_1, offset_2)):
                break
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            vectors = self._network.extract_vector(imgs)
            for c_idx in labels.unique():
                class_vectors[c_idx.item()].append(vectors[labels == c_idx].cpu().detach())
        class_vectors = {k: torch.cat(v, dim=0) for k, v in class_vectors.items()}
        for k in class_vectors.keys():
            class_means[k] = class_vectors[k].mean(dim=0)
            class_covs[k] = torch.cov(class_vectors[k].T) + torch.eye(class_means[k].shape[-1]) * 1e-4
        print('Done.', file=sys.stderr)
        self._network.train(status)
        return class_means, class_covs

    def _stage2_compact_classifier(self, class_means, class_covs, offset_1, offset_2):
        seq_dataset = get_dataset(self.args)
        cpt = seq_dataset.N_CLASSES_PER_TASK

        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = self.args.ca_epochs
        crct_num = offset_2
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.args.lr,
                           'weight_decay': self.args.optim_wd}]
        optimizer = optim.SGD(network_params, lr=self.args.lr, momentum=0.9, weight_decay=self.args.optim_wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        status = self._network.training
        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):

                if not isinstance(cpt, list):
                    cpt = [cpt] * seq_dataset.N_TASKS
                cumsum = np.cumsum(cpt)
                t_id = np.argmax(cumsum > c_id)
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = class_means[c_id].to(self._device) * (0.9 + decay)
                cls_cov = class_covs[c_id].to(self._device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in tqdm(range(crct_num)):
                if self.args.debug_mode and _iter >= 5:
                    break
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += cpt[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += cpt[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
        self._network.train(status)
