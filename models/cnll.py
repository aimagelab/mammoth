import logging
import sys
import time
import numpy as np
import torch
from torch import nn
import tqdm
from typing import TYPE_CHECKING
from copy import deepcopy
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args
from utils.autoaugment import CIFAR10Policy
from utils.buffer import Buffer
from torchvision import transforms

from utils.kornia_utils import to_kornia_transform

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset


def kl_divergence(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)


class Jensen_Shannon(nn.Module):
    def forward(self, p, q):
        m = (p + q) / 2
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return float(current)


class SemiLoss(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs_x, outputs_x2, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None, transform=None, device="cpu"):
        self.data = data.to(device)
        self.targets = targets.to(device) if targets is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        if self.targets is not None:
            return data, self.targets[idx]
        return data


def get_hard_transform(args, dataset: 'ContinualDataset'):
    assert 'seq-cifar10' in args.dataset.lower(), "Hard transform is only available for seq-cifar10 and seq-cifar100"

    return transforms.Compose([transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               CIFAR10Policy(),
                               transforms.ToTensor(),
                               dataset.get_normalization_transform()])


class Cnll(ContinualModel):
    """
    Implementation of `CNLL: A Semi-supervised Approach For Continual Noisy Label Learning <https://github.com/nazmul-karim170/CNLL>`_ from CVPRW 2022.
    """
    NAME = 'cnll'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(lr=0.0005)  # optimizer and lr are not used
        add_rehearsal_args(parser)

        parser.add_argument('--cnll_debug_mode', type=binary_to_boolean_type, default=False,
                            help='Run CNLL with just a few iterations?')
        parser.add_argument('--unlimited_buffer', type=binary_to_boolean_type, default=False,
                            help='Use unlimited buffers?')

        parser.add_argument('--delayed_buffer_size', type=int, default=500,
                            help='Size of the delayed buffer.')
        parser.add_argument('--noisy_buffer_size', type=int, default=1000,
                            help='Size of the noisy buffer.')

        parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
        parser.add_argument('--finetune_epochs', type=int, default=10, help='Finetuning epochs')
        parser.add_argument('--warmup_lr', type=float, default=0.001, help='Warmup learning rate')

        parser.add_argument('--subsample_clean', type=int, default=25,
                            help='Number of high confidence samples to subsample from the clean buffer (N_1 in the paper)')
        parser.add_argument('--subsample_noisy', type=int, default=50,
                            help='Number of high confidence samples to subsample from the noisy buffer (N_2 in the paper)')

        parser.add_argument('--sharp_temp', type=float, default=0.5, help='Temperature for label CO-Guessing')
        parser.add_argument('--mixup_alpha', type=float, default=4, help='Alpha parameter of Beta distribution for mixup')
        parser.add_argument('--lambda_u', type=float, default=30, help='Weight for unsupervised loss')
        parser.add_argument('--lambda_c', type=float, default=0.025, help='Weight for constrastive loss')

        parser.add_argument('--finetune_lr', type=float, default=0.1, help='Warmup learning rate')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone.classifier_re = deepcopy(backbone.classifier)
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # initializing buffers
        self.buffer = Buffer(self.args.buffer_size, "cpu")
        self.delayed_buffer = Buffer(self.args.delayed_buffer_size, "cpu")
        self.noisy_buffer = Buffer(self.args.noisy_buffer_size, "cpu")

        if self.args.unlimited_buffer:
            logging.warning("Using unlimited buffer!")
            self.high_fidelity_buffer = Buffer(-1, device='cpu', sample_selection_strategy='unlimited')  # unlimited buffers
            self.high_fidelity_noisy_buffer = Buffer(-1, device='cpu', sample_selection_strategy='unlimited')  # unlimited buffers
        else:
            self.high_fidelity_buffer = Buffer(self.args.buffer_size, "cpu")  # more buffers
            self.high_fidelity_noisy_buffer = Buffer(self.args.noisy_buffer_size, "cpu")  # more buffers

        self._past_it_t = time.time()
        self._task_t = time.time()
        self._avg_it_t = 0
        self.past_loss = 0
        self.conf_penalty = NegEntropy()
        self.JS_dist = Jensen_Shannon()

        self.eye = torch.eye(self.num_classes).to(self.device)

        self.hard_transform = to_kornia_transform(get_hard_transform(self.args, dataset))
        self.semi_sul_loss = SemiLoss(args)

    def warm_up_on_buffer(self, buffer: Buffer):
        opt = torch.optim.SGD(self.net.parameters(), lr=self.args.warmup_lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 240, 2e-4)

        for _ in range(self.args.warmup_epochs):
            self.net.train()
            for batch in buffer.get_dataloader(self.args, batch_size=self.args.batch_size, shuffle=True):
                opt.zero_grad()
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                x = self.transform(x)
                pred = self.net(x)
                loss = self.loss(pred, y)

                if 'asym' in self.args.noise_type:
                    penalty = self.conf_penalty(loss)
                    if not torch.isnan(penalty):
                        loss += penalty

                loss.backward()
                opt.step()

            sched.step()

    def begin_task(self, dataset):
        if self.current_task > 0:
            ct = time.time()
            remaining_time = ((ct - self._task_t) * (self.n_tasks - self.current_task)) * self.args.n_epochs
            logging.debug(f"Task {self.current_task-1} lasted {ct-self._task_t:.2f}s | remaining: {remaining_time:.2f}s")
        self.observe_it = 0
        self.tot_its = ((len(dataset.train_loader.dataset) // self.args.delayed_buffer_size) + 1) * self.args.n_epochs

        # FIRST LIE: NO TASK BOUNDARY
        self.current_classes = np.unique(dataset.train_loader.dataset.targets)
        weight = torch.zeros(self.num_classes)
        weight[self.current_classes] = 1
        weight = weight.to(self.device)
        self.loss = nn.CrossEntropyLoss(weight=weight)

        self._task_t = self._past_it_t = time.time()

    @torch.no_grad()
    def sample_selection_JSD(self, buffer: Buffer):
        selected_indexes = torch.zeros(len(buffer))

        for batch_idx, batch in enumerate(buffer.get_dataloader(self.args, batch_size=self.args.batch_size * 2, drop_last=True, shuffle=False)):
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            batch_size = inputs.size()[0]

            # Get outputs of both network
            preds = torch.softmax(self.net(inputs), dim=-1)

            out = torch.zeros(preds.size()).to(self.device)
            out[:, self.current_classes] = preds[:, self.current_classes]

            _, ind = torch.max(out, 1)
            out_final = torch.zeros(preds.size()).to(self.device)
            for kk in range(out.size()[0]):
                out_final[kk, ind[kk]] = 1

            dist = self.JS_dist(out_final, F.one_hot(targets, num_classes=self.num_classes))
            selected_indexes[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = dist

        return selected_indexes

    @torch.no_grad()
    def get_partition_buffer_indexes(self, buffer: Buffer):
        buffer_size = len(buffer)
        selected_indexes = self.sample_selection_JSD(buffer)
        threshold = torch.mean(selected_indexes)
        SR = torch.sum(selected_indexes < threshold).item() / buffer_size

        selected_indexes = selected_indexes.cpu().numpy()
        pred_idx = np.argsort(selected_indexes)[0: int(SR * buffer_size)]

        idx = np.arange(buffer_size)

        pred_idx_noisy = np.setdiff1d(idx, pred_idx)

        repl_idx = np.array(pred_idx)[:self.args.subsample_clean]
        repl_idx_noisy = np.array(pred_idx_noisy)[:self.args.subsample_noisy]

        return pred_idx, pred_idx_noisy, repl_idx, repl_idx_noisy

    def _observe(self, not_aug_x, y, true_y):
        self.delayed_buffer.add_data(examples=not_aug_x.unsqueeze(0),
                                     labels=y.unsqueeze(0),
                                     true_labels=true_y.unsqueeze(0))

        avg_expert_loss, avg_self_loss = -1, -1
        if self.delayed_buffer.is_full():
            if self.args.cnll_debug_mode and self.observe_it > 2:
                return 0, 0

            self.observe_it += 1
            ctime = time.time()
            self._avg_it_t = (self._avg_it_t + (ctime - self._past_it_t)) / (self.observe_it)
            remaing_time = (self.tot_its - self.observe_it) * self._avg_it_t
            logging.debug(f"[Task {self.current_task}] Buffer iteration: {self.observe_it}/{self.tot_its} (s/it: {self._avg_it_t:.2f}s | rem: {remaing_time:.2f}s)")
            self._past_it_t = ctime

            # Warm up on D
            pret = time.time()
            logging.debug(" - Warm up...", end='')
            avg_expert_loss = self.warm_up_on_buffer(self.delayed_buffer)
            logging.debug(f" Done (s: {time.time()-pret:.2f}s)")

            pret = time.time()
            logging.debug(" - Purifying buffer...", end='')

            # Get clean samples from D
            clean_idxs, noisy_idxs, high_fidelity_clean_idxs, high_fidelity_noisy_idxs = self.get_partition_buffer_indexes(self.delayed_buffer)

            # Add clean samples to P
            self.buffer.add_data(examples=self.delayed_buffer.examples[clean_idxs],
                                 labels=self.delayed_buffer.labels[clean_idxs],
                                 true_labels=self.delayed_buffer.true_labels[clean_idxs])
            self.noisy_buffer.add_data(examples=self.delayed_buffer.examples[noisy_idxs],
                                       labels=self.delayed_buffer.labels[noisy_idxs],
                                       true_labels=self.delayed_buffer.true_labels[noisy_idxs])

            self.high_fidelity_buffer.add_data(examples=self.delayed_buffer.examples[high_fidelity_clean_idxs],
                                               labels=self.delayed_buffer.labels[high_fidelity_clean_idxs],
                                               true_labels=self.delayed_buffer.true_labels[high_fidelity_clean_idxs])
            self.high_fidelity_noisy_buffer.add_data(examples=self.delayed_buffer.examples[high_fidelity_noisy_idxs],
                                                     labels=self.delayed_buffer.labels[high_fidelity_noisy_idxs],
                                                     true_labels=self.delayed_buffer.true_labels[high_fidelity_noisy_idxs])

            logging.debug(f" Done (s: {time.time()-pret:.2f}s)")

            self.delayed_buffer.empty()

        if self.buffer.is_full():
            pret = time.time()
            logging.debug(" - Clean buffer is full, fine-tuning model on buffers...", end='')

            self.past_loss = self.finetune_on_buffers()

            self.buffer.empty()
            self.noisy_buffer.empty()

            logging.debug(f" Done (s: {time.time()-pret:.2f}s)")

        return avg_expert_loss, avg_self_loss

    @torch.no_grad()
    def coguess_label(self, xa, xb, y):
        # Label Co-guessing of Unlabeled Samples
        outputs_u11 = self.net(xa)
        outputs_u12 = self.net(xb)

        # Pseudo-Label
        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
        ptu = pu**(1 / self.args.sharp_temp)  # Temparature Sharpening

        targets_u = ptu / ptu.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

        # Label Refinement
        outputs_x = self.net(xa)
        outputs_x2 = self.net(xb)

        px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

        px = y  # w_x*y + (1-w_x)*px # BLE
        ptx = px**(1 / self.args.sharp_temp)    # Temparature sharpening

        targets_x = ptx / ptx.sum(dim=1, keepdim=True)
        return targets_x, targets_u

    def ssl_loss(self, all_inputs, all_targets, batch_size, c_iter):
        idx = torch.randperm(all_inputs.size(0))
        l = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
        l = max(l, 1 - l)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        feats = self.net(mixed_input, 'features')
        logits1, logits = self.net.classifier(feats), self.net.classifier_re(feats)
        logits_x = logits1[:batch_size * 2]
        logits_x1 = logits[:batch_size * 2]

        logits_u = logits[batch_size * 2:]

        # Combined Loss
        Lx, Lu, lamb = self.semi_sul_loss(logits_x, logits_x1,
                                          mixed_target[:batch_size * 2], logits_u,
                                          mixed_target[batch_size * 2:], c_iter, self.args.warmup_epochs)

        # Regularization
        prior = torch.ones(self.num_classes, device=self.device) / self.num_classes
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        lamb *= self.args.lambda_u

        return Lx + 0.1 * lamb * Lu + penalty

    def finetune_on_buffers(self):
        """Fit finetuned model on purified and noisy buffer"""
        self.net.train()

        opt = torch.optim.SGD(self.net.parameters(), lr=self.args.finetune_lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

        all_clean_data = torch.cat([self.buffer.examples[:len(self.buffer)],
                                    self.high_fidelity_buffer.examples[:len(self.high_fidelity_buffer)]], dim=0)
        all_clean_labels = torch.cat([self.buffer.labels[:len(self.buffer)],
                                      self.high_fidelity_buffer.labels[:len(self.high_fidelity_buffer)]], dim=0)

        clean_dset = Dataset(all_clean_data, all_clean_labels, device=self.device)
        clean_dl = torch.utils.data.DataLoader(clean_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        all_noisy_data = torch.cat([self.noisy_buffer.examples[:len(self.noisy_buffer)],
                                    self.high_fidelity_noisy_buffer.examples[:len(self.high_fidelity_noisy_buffer)]], dim=0)
        noisy_dset = Dataset(all_noisy_data, device=self.device)
        noisy_dl = torch.utils.data.DataLoader(noisy_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        noisy_iter = iter(noisy_dl)
        for epoch in tqdm.trange(self.args.finetune_epochs, desc="Buffer fitting", leave=False, disable=True):
            avgloss = 0
            if self.args.cnll_debug_mode == 1 and epoch > 2:
                break
            for batch_idx, dat in enumerate(clean_dl):
                try:
                    noisy_dat = next(noisy_iter)
                except StopIteration:
                    noisy_iter = iter(noisy_dl)
                    noisy_dat = next(noisy_iter)

                opt.zero_grad()

                x, y = dat[0], dat[1]
                x, y = x.to(self.device), y.to(self.device)
                onehot_y = self.eye[y]
                clean_xa, clean_xb = self.transform(x), self.transform(x)

                unlabeled_x = noisy_dat.to(self.device)
                unlabeled_xa, unlabeled_xb = self.hard_transform(unlabeled_x), self.hard_transform(unlabeled_x)

                refined_y_clean, pseudo_y = self.coguess_label(unlabeled_xa, unlabeled_xb, onehot_y)

                all_inputs = torch.cat([clean_xa, clean_xb, unlabeled_xa, unlabeled_xb], dim=0)
                all_labels = torch.cat([refined_y_clean, refined_y_clean, pseudo_y, pseudo_y], dim=0)

                loss = self.ssl_loss(all_inputs, all_labels, len(clean_xa), epoch + batch_idx / len(clean_dl))

                loss.backward()

                opt.step()
                avgloss += loss.item()

        avgloss /= len(clean_dl)
        return avgloss  # retrun the average loss at the last epoch

    def observe(self, inputs, labels, not_aug_inputs, true_labels):
        for y, not_aug_x, true_y in zip(labels, not_aug_inputs, true_labels):
            avg_expert_loss, avg_self_loss = self._observe(not_aug_x, y, true_y)

        return self.past_loss if avg_self_loss < 0 else avg_self_loss
