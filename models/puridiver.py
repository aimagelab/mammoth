import time
from typing import Union
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import logging

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from argparse import ArgumentParser, Namespace
from utils.args import add_rehearsal_args
from utils.augmentations import RepeatedTransform, cutmix_data
from utils.autoaugment import CIFAR10Policy
from utils.buffer import Buffer
import torch.nn.functional as F
from torchvision import transforms

from utils.conf import create_seeded_dataloader
from utils.kornia_utils import to_kornia_transform


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None, probs=None, extra=None, device="cpu"):
        self.device = device
        self.data = data.to(self.device)
        self.targets = targets.to(device) if targets is not None else None
        self.transform = transform
        self.probs = (torch.ones(len(self.data)) / len(self.data)).to(device) if probs is None else probs.to(device)
        self.extra = extra.to(device) if extra is not None else None

    def set_probs(self, probs: Union[np.ndarray, torch.Tensor]):
        """
        Set the probability of each data point being correct (i.e., belonging to the Gaussian with the lowest mean)
        """
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.to(self.data.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the data, the target, the extra information (if any), the not augmented data, and the probability of the data point being correct

        Returns:
        - data: the augmented data
        - target: the target
        - extra: (optional) additional information
        - not_aug_data: the data without augmentation
        - prob: the probability of the data point being correct
        """
        not_aug_data = self.data[idx]
        data = not_aug_data.clone()
        if self.transform:
            data = self.transform(data)
            if len(data.shape) > 3:
                if data.shape[0] == 1:
                    data = data.squeeze(0)
                elif data.shape[1] == 1:
                    data = data.squeeze(1)

        ret = (data, self.targets[idx],)
        if self.extra is not None:
            ret += (self.extra[idx],)
        ret += (not_aug_data,)
        return ret + (self.probs[idx],)


def soft_cross_entropy_loss(input, target, reduction='mean'):
    """
    https://github.com/pytorch/pytorch/issues/11959

    Args:
        input: (batch, *)
        target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def get_hard_transform(dataset: ContinualDataset):
    return transforms.Compose([transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               CIFAR10Policy(),
                               transforms.ToTensor(),
                               dataset.get_normalization_transform()])


def get_dataloader_from_buffer(args: Namespace, buffer: Buffer, batch_size: int, shuffle=False, transform=None):
    if len(buffer) == 0:
        return None

    buf_data = buffer.get_all_data(device="cpu")
    inputs, labels = buf_data[0], buf_data[1]

    # Building train dataset
    train_dataset = CustomDataset(inputs, labels, transform=transform)

    return create_seeded_dataloader(args, train_dataset, non_verbose=True, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class PuriDivER(ContinualModel):
    """PuriDivER: Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries."""
    NAME = 'puridiver'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(n_epochs=1, optim_mom=0.9, optim_wd=1e-4, optim_nesterov=1, batch_size=16)

        add_rehearsal_args(parser)

        parser.add_argument('--use_bn_classifier', type=int, default=1, choices=[0, 1],
                            help='Use batch normalization in the classifier?')
        parser.add_argument('--freeze_buffer_after_first', type=int, default=0, choices=[0, 1],
                            help='Freeze buffer after first task (i.e., simulate online update of the buffer, useful for multi-epoch)?')
        parser.add_argument('--initial_alpha', type=float, default=0.5)
        parser.add_argument('--disable_train_aug', type=int, default=1, choices=[0, 1], help='Disable training augmentation?')
        parser.add_argument('--buffer_fitting_epochs', type=int, default=255, help='Number of epochs to fit on buffer')
        parser.add_argument('--warmup_buffer_fitting_epochs', type=int, default=10, help='Number of warmup epochs during which fit with simple CE')
        parser.add_argument('--enable_cutmix', type=int, default=1, choices=[0, 1], help='Enable cutmix augmentation?')
        parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Cutmix probability')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert args.dataset in ['seq-cifar10', 'seq-cifar100'], 'PuriDivER is only compatible with CIFAR datasets (extend `get_hard_transform` for other datasets)'

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, "cpu")

        self._past_it_t = time.time()
        self._avg_it_t = 0
        self.past_loss = 0
        self.eye = torch.eye(self.num_classes).to(self.device)

        hard_transform = get_hard_transform(self.dataset)
        try:
            self.hard_transform = to_kornia_transform(hard_transform)
        except NotImplementedError as e:
            _logging.error('Kornia not available, raising error instead of using PIL transforms (would be waaay too slow).')
            # NOTE: uncomment the following line if you want to use PIL transforms
            # self.hard_transform = hard_transform
            raise e

    def get_subset_dl_from_idxs(self, idxs, batch_size, probs=None, transform=None):
        if idxs is None:
            return None
        assert batch_size is not None

        examples, labels, true_labels = self.buffer.get_all_data()
        examples, labels, true_labels = examples[idxs], labels[idxs], true_labels[idxs]

        if probs is not None:
            probs = torch.from_numpy(probs)
        dataset = CustomDataset(examples, labels, extra=true_labels, probs=probs, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    @torch.no_grad()
    def split_data_puridiver(self, n=2):
        self.net.eval()

        losses = []
        uncertainties = []
        for batch_idx, batch in enumerate(get_dataloader_from_buffer(self.args, self.buffer, batch_size=64, shuffle=False)):
            x, y, y_true = batch[0], batch[1], batch[-1]
            x, y, y_true = x.to(self.device), y.to(self.device), y_true.to(self.device)
            x = self.normalization_transform(x)
            out = self.net(x)
            probs = F.softmax(out, dim=1)
            uncerts = 1 - torch.max(probs, 1)[0]

            losses.append(F.cross_entropy(out, y, reduction='none'))
            uncertainties.append(uncerts)

        losses = torch.cat(losses, dim=0).cpu()
        uncertainties = torch.cat(uncertainties, dim=0).cpu().reshape(-1, 1)
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        losses = losses.unsqueeze(1)

        # GMM for correct vs others samples
        gmm_loss = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_loss.fit(losses)
        gmm_loss_means = gmm_loss.means_
        if gmm_loss_means[0] <= gmm_loss_means[1]:
            small_loss_idx = 0
            large_loss_idx = 1
        else:
            small_loss_idx = 1
            large_loss_idx = 0

        loss_prob = gmm_loss.predict_proba(losses)
        pred = loss_prob.argmax(axis=1)

        corr_idxs = np.where(pred == small_loss_idx)[0]
        if len(corr_idxs) == 0:
            return None, None, None

        # 2nd GMM using large loss datasets
        high_loss_idxs = np.where(pred == large_loss_idx)[0]

        ambiguous_idxs, incorrect_idxs = None, None
        if len(high_loss_idxs) > 2:
            # GMM for uncertain vs incorrect samples
            gmm_uncert = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_uncert.fit(uncertainties[high_loss_idxs])
            prob_uncert = gmm_uncert.predict_proba(uncertainties[high_loss_idxs])
            pred_uncert = prob_uncert.argmax(axis=1)
            if gmm_uncert.means_[0] <= gmm_uncert.means_[1]:
                small_loss_idx = 0
                large_loss_idx = 1
            else:
                small_loss_idx = 1
                large_loss_idx = 0

            idx_uncert = np.where(pred_uncert == small_loss_idx)[0]
            amb_size = len(idx_uncert)
            ambiguous_batch_size = max(2, int(amb_size / len(corr_idxs) * self.args.batch_size))
            if amb_size <= 2:
                ambiguous_idxs = None
            else:
                ambiguous_idxs = high_loss_idxs[idx_uncert]

            idx_uncert = np.where(pred_uncert == large_loss_idx)[0]
            incorrect_size = len(idx_uncert)
            incorrect_batch_size = max(2, int(incorrect_size / len(corr_idxs) * self.args.batch_size))
            if incorrect_size <= 2:
                incorrect_idxs = None
            else:
                incorrect_idxs = high_loss_idxs[idx_uncert]

        correct_dl = self.get_subset_dl_from_idxs(corr_idxs, self.args.batch_size, transform=self.hard_transform)

        if ambiguous_idxs is not None:
            ambiguous_dl = self.get_subset_dl_from_idxs(ambiguous_idxs, ambiguous_batch_size, transform=RepeatedTransform([self.transform, self.hard_transform], autosqueeze=True))
        else:
            ambiguous_dl = None

        if incorrect_idxs is not None:
            incorrect_dl = self.get_subset_dl_from_idxs(incorrect_idxs, incorrect_batch_size, probs=loss_prob[incorrect_idxs], transform=RepeatedTransform([
                                                        self.transform, self.hard_transform], autosqueeze=True))
        else:
            incorrect_dl = None

        return correct_dl, ambiguous_dl, incorrect_dl

    def train_with_mixmatch(self, loader_L, loader_U, loader_R):
        criterion_U = nn.MSELoss()
        criterion_L = nn.CrossEntropyLoss()

        iter_U = iter(loader_U)
        iter_R = iter(loader_R)
        avg_loss = 0

        # R: weak, hard
        # L: hard
        # U: weak, hard
        self.net.train()
        for i, batch in enumerate(loader_L):
            if self.args.debug_mode and i > 10:
                break
            self.opt.zero_grad()
            inputs_L, labels_L = batch[0], batch[1]
            if len(inputs_L) == 1:
                continue
            try:
                inputs_U = next(iter_U)[0]
            except BaseException:
                iter_U = iter(loader_U)
                inputs_U = next(iter_U)[0]
            try:
                batch_R = next(iter_R)
                inputs_R, labels_R, probs_R = batch_R[0], batch_R[1], batch_R[-1]
            except BaseException:
                iter_R = iter(loader_R)
                batch_R = next(iter_R)
                inputs_R, labels_R, probs_R = batch_R[0], batch_R[1], batch_R[-1]

            inputs_L, labels_L = inputs_L.to(self.device), labels_L.to(self.device)
            inputs_U, inputs_R = inputs_U.to(self.device), inputs_R.to(self.device)
            labels_R, probs_R = labels_R.to(self.device), probs_R.to(self.device)
            labels_R = F.one_hot(labels_R, self.num_classes)
            corr_prob = probs_R[:, 0].unsqueeze(1).expand(-1, self.num_classes)

            inputs_U = torch.cat([inputs_U[:, 0], inputs_U[:, 1]], dim=0)
            inputs_R = torch.cat([inputs_R[:, 0], inputs_R[:, 1]], dim=0)

            do_cutmix = self.args.enable_cutmix and np.random.random(1) < self.args.cutmix_prob
            if do_cutmix:
                inputs_L, labels_L_a, labels_L_b, lam = cutmix_data(inputs_L, labels_L, force=True)

                all_inputs = torch.cat([inputs_R, inputs_U, inputs_L], dim=0)
                all_outputs = self.net(all_inputs)
                outputs_R, outputs_U, outputs_L = torch.split(all_outputs, [inputs_R.size(0), inputs_U.size(0), inputs_L.size(0)])

                loss_L = lam * self.loss(outputs_L, labels_L_a) + (1 - lam) * criterion_L(outputs_L, labels_L_b)
            else:
                all_inputs = torch.cat([inputs_R, inputs_U, inputs_L], dim=0)
                all_outputs = self.net(all_inputs)
                outputs_R, outputs_U, outputs_L = torch.split(all_outputs, [inputs_R.size(0), inputs_U.size(0), inputs_L.size(0)])
                outputs_L = self.net(inputs_L)

                loss_L = self.loss(outputs_L, labels_L)

            outputs_U_weak, outputs_U_strong = torch.split(outputs_U, outputs_U.size(0) // 2)
            outputs_R_pseudo, outputs_R = torch.split(outputs_R, outputs_R.size(0) // 2)  # weak, strong

            probs_R_pseudo = torch.softmax(outputs_R_pseudo, dim=1)
            soft_pseudo_labels = corr_prob * labels_R + (1 - corr_prob) * probs_R_pseudo.detach()

            loss_R = soft_cross_entropy_loss(outputs_R, soft_pseudo_labels)
            loss_U = criterion_U(outputs_U_weak, outputs_U_strong)

            coeff_L = (len(labels_L) / (len(labels_L) + len(labels_R) + len(outputs_U_weak)))
            coeff_R = (len(labels_R) / (len(labels_R) + len(labels_L) + len(outputs_U_weak)))
            coeff_U = (len(outputs_U_weak) / (len(labels_R) + len(labels_L) + len(outputs_U_weak)))
            loss = coeff_L * loss_L + coeff_U * loss_U + coeff_R * loss_R

            assert not torch.isnan(loss).any()
            # backward
            loss.backward()
            self.opt.step()

            avg_loss += loss.item()
        return avg_loss / len(loader_L)

    def base_fit_buffer(self, loader=None):
        self.net.train()
        avg_loss = 0
        if loader is None:
            loader = get_dataloader_from_buffer(self.args, self.buffer, batch_size=self.args.batch_size, shuffle=True, transform=self.hard_transform)

        for i, batch in enumerate(loader):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            if len(x) == 1:
                continue
            if self.args.debug_mode and i > 10:
                break

            self.opt.zero_grad()

            do_cutmix = self.args.enable_cutmix and np.random.rand(1) < self.args.cutmix_prob
            if do_cutmix:
                x, y_a, y_b, lam = cutmix_data(x, y, force=True)

                out = self.net(x)

                loss = lam * self.loss(out, y_a) + (1 - lam) * self.loss(out, y_b)
            else:
                out = self.net(x)

                loss = self.loss(out, y)

            assert not torch.isnan(loss).any()
            loss.backward()
            self.opt.step()

            avg_loss += loss.item()
        return avg_loss / len(loader)

    def fit_buffer(self):
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.args.lr

        with tqdm.trange(self.args.buffer_fitting_epochs) as pbar:
            for epoch in pbar:
                if self.args.debug_mode and epoch > self.args.warmup_buffer_fitting_epochs + 50:
                    break
                if epoch < self.args.warmup_buffer_fitting_epochs:
                    tp = 'warmup'
                    loss = self.base_fit_buffer()
                else:
                    correct_dl, ambiguous_dl, incorrect_dl = self.split_data_puridiver()

                    if ambiguous_dl is not None and incorrect_dl is not None:
                        tp = 'puridiver'
                        loss = self.train_with_mixmatch(correct_dl, ambiguous_dl, incorrect_dl)
                    else:
                        tp = 'base'
                        loss = self.base_fit_buffer()

                buf_not_aug_inputs, buf_labels, buf_true_labels = self.buffer.get_all_data()
                _, _, buf_acc, true_buf_acc = self._non_observe_data(self.normalization_transform(buf_not_aug_inputs), buf_labels, buf_true_labels)

                perc_clean = (self.buffer.labels == self.buffer.true_labels).float().mean().item()

                pbar.set_postfix(loss=loss, buf_acc=buf_acc, true_buf_acc=true_buf_acc, perc_clean=perc_clean, lr=self.opt.param_groups[0]["lr"], refresh=False)
                pbar.set_description(f'Epoch {epoch + 1}/{self.args.buffer_fitting_epochs} [{tp}]', refresh=False)

                self.scheduler.step()

    def end_task(self, dataset):
        # fit classifier on P
        if self.args.buffer_fitting_epochs > 0:
            self.fit_buffer()

    def get_classifier_weights(self):
        if isinstance(self.net.classifier, nn.Sequential):
            return self.net.classifier[0].weight.detach()
        return self.net.classifier.weight.detach()

    def get_sim_score(self, feats, targets):
        # relevant representation
        cl_weights = self.get_classifier_weights()

        relevant_idx = cl_weights[targets[0], :] > cl_weights.mean(dim=0)

        cls_features = feats[:, relevant_idx]
        sim_score = torch.cosine_similarity(cls_features, cls_features, dim=1)

        return (sim_score - sim_score.mean()) / sim_score.std()

    def get_current_alpha_sim_score(self, loss):
        return self.args.initial_alpha * min(1, 1 / loss)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=1, T_mult=2, eta_min=self.args.lr * 0.01
        )

    def begin_task(self, dataset):
        self.total_its = len(dataset.train_loader) * self.args.n_epochs

        if self.current_task == 0 and self.args.use_bn_classifier:
            self.net.classifier = nn.Sequential(nn.Linear(self.net.classifier.in_features, self.net.classifier.out_features, bias=False),
                                                nn.BatchNorm1d(self.net.classifier.out_features, affine=True, eps=1e-6).to(self.device)).to(self.device)

            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    m.eps = 1e-6

        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.args.lr

        if self.args.disable_train_aug:
            dataset.train_loader.dataset.transform = self.dataset.TEST_TRANSFORM

    @torch.no_grad()
    def _non_observe_data(self, inputs: torch.Tensor, labels: torch.Tensor, true_labels: torch.Tensor = None):
        was_training = self.net.training
        self.net.eval()

        dset = CustomDataset(inputs, labels, extra=true_labels, device=self.device)
        dl = DataLoader(dset, batch_size=min(len(dset), 256), shuffle=False, num_workers=0)

        feats = []
        losses = []
        true_accs, accs = [], []
        for batch in dl:
            inputs, labels, true_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            out, feat = self.net(inputs, returnt='both')
            acc = (out.argmax(dim=1) == labels).float().mean().item()
            tacc = (out.argmax(dim=1) == true_labels).float().mean().item()
            feats.append(feat)
            losses.append(F.cross_entropy(out, labels, reduction='none'))
            accs.append(acc)
            true_accs.append(tacc)

        feats = torch.cat(feats, dim=0)
        losses = torch.cat(losses, dim=0)
        acc = np.mean(accs)
        true_acc = np.mean(true_accs)

        self.net.train(was_training)

        return feats, losses, acc, true_acc

    def puridiver_update_buffer(self, stream_not_aug_inputs: torch.Tensor, stream_labels: torch.Tensor, stream_true_labels: torch.Tensor):
        if len(self.buffer) < self.args.buffer_size:
            self.buffer.add_data(examples=stream_not_aug_inputs, labels=stream_labels, true_labels=stream_true_labels)
            return -1, -1
        buf_not_aug_inputs, buf_labels, buf_true_labels = self.buffer.get_all_data()
        buf_not_aug_inputs, buf_labels, buf_true_labels = buf_not_aug_inputs.to(self.device), buf_labels.to(self.device), buf_true_labels.to(self.device)
        not_aug_inputs = torch.cat([buf_not_aug_inputs, stream_not_aug_inputs], dim=0)
        labels = torch.cat([buf_labels, stream_labels], dim=0)
        true_labels = torch.cat([buf_true_labels, stream_true_labels], dim=0)

        cur_idxs = torch.arange(len(not_aug_inputs)).to(self.device)
        feats, losses, buf_acc, true_buf_acc = self._non_observe_data(self.normalization_transform(not_aug_inputs), labels, true_labels=true_labels)
        alpha_sim_score = self.get_current_alpha_sim_score(losses.mean())
        lbs = labels[cur_idxs]
        while len(lbs) > self.args.buffer_size:
            fts = feats[cur_idxs]
            lss = losses[cur_idxs]

            clss, cls_cnt = lbs.unique(return_counts=True)
            # argmax w/ random tie-breaking
            cls_to_drop = clss[cls_cnt == cls_cnt.max()]
            cls_to_drop = cls_to_drop[torch.randperm(len(cls_to_drop))][0]
            mask = lbs == cls_to_drop

            sim_score = self.get_sim_score(fts[mask], lbs[mask])
            div_score = (1 - alpha_sim_score) * lss[mask] + alpha_sim_score * sim_score

            drop_cls_idx = div_score.argmax()
            drop_idx = cur_idxs[mask][drop_cls_idx]
            cur_idxs = cur_idxs[cur_idxs != drop_idx]

            lbs = labels[cur_idxs]

        self.buffer.empty()
        self.buffer.add_data(examples=not_aug_inputs[cur_idxs], labels=labels[cur_idxs], true_labels=true_labels[cur_idxs])
        return buf_acc, true_buf_acc

    def observe(self, inputs, labels, not_aug_inputs, true_labels, epoch):
        self.net.train()

        B = len(inputs)

        self.opt.zero_grad()

        if self.current_task > 0:  # starting from second task
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.hard_transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        do_cutmix = self.args.enable_cutmix and np.random.rand(1) < self.args.cutmix_prob
        if do_cutmix:
            inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, force=True)

            outputs = self.net(inputs)

            loss = lam * self.loss(outputs, labels_a) + (1 - lam) * self.loss(outputs, labels_b)
        else:
            outputs = self.net(inputs)

            loss = self.loss(outputs, labels)

        assert not torch.isnan(loss).any()
        loss.backward()
        self.opt.step()

        if self.args.freeze_buffer_after_first == 0 or epoch == 0:
            self.puridiver_update_buffer(not_aug_inputs[:B], labels[:B], true_labels[:B])

        return loss.item()
