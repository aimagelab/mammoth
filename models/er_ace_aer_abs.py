
import copy
from typing import Union
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.beta import Beta
from argparse import ArgumentParser
import tqdm
import numpy as np

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args
from utils.buffer import Buffer
from utils.augmentations import apply_transform, cutmix_data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None, extra=None, device="cpu"):
        self.device = device
        self.data = data.to(self.device)
        self.targets = targets.to(device) if targets is not None else None
        self.transform = transform
        self.probs = (torch.ones(len(self.data)) / len(self.data)).to(device)
        self.extra = extra

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
            data = apply_transform(data, self.transform, autosqueeze=True)
        ret = (data, self.targets[idx],)
        if self.extra is not None:
            ret += (self.extra[idx],)
        ret += (not_aug_data,)
        return ret + (self.probs[idx],)


class ErAceAerAbs(ContinualModel):
    """Er-ACE with AER and ABS, from `May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels`."""
    NAME = 'er_ace_aer_abs'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--sample_selection_strategy', default='abs', type=str, choices=['reservoir', 'lars', 'abs'],
                            help='Sample selection strategy to use: `reservoir`, `lars` (known as LASS in the paper), or `abs`')
        parser.add_argument('--use_aer', default=1, type=int, choices=[0, 1], help='Use Alternate Replay?')
        parser.add_argument('--alpha_sample_insertion', default=0.75, type=float, help='percentage of high loss samples to remove from buffer')

        group = parser.add_argument_group('Buffer fitting')
        group.add_argument('--buffer_fitting_epochs', type=int, default=0, help='Number of epochs to fit on buffer')
        group.add_argument('--enable_cutmix_buffer_fitting', type=int, default=0, choices=[0, 1], help='Enable cutmix augmentation during buffer fitting?')
        group.add_argument('--buffer_fitting_lr', type=float, default=0.05, help='Buffer fitting learning rate')
        group.add_argument('--buffer_fitting_type', type=str, default='mixmatch', choices=['simple', 'mixmatch'], help='Buffer fitting strategy:'
                           ' - `simple`: simple buffer fitting based on CE'
                           ' - `mixmatch`: try separate clean/noisy and fit with mixmatch')
        group.add_argument('--mixmatch_naug_buffer_fitting', type=int, default=3, help='Number of augmentations for mixmatch during buffer fitting')
        group.add_argument('--mixmatch_alpha_buffer_fitting', type=float, default=0.5, help='Alpha parameter for mixmatch for the Beta distribution')
        group.add_argument('--mixmatch_lambda_buffer_fitting', type=float, default=0.01, help='Lambda parameter for mixmatch')
        group.add_argument('--buffer_fitting_batch_size', type=int, default=32)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.alpha_sample_insertion > 1:
            assert args.alpha_sample_insertion <= 100, 'alpha_sample_insertion should be a percentage'
            args.alpha_sample_insertion = args.alpha_sample_insertion / 100

        super(ErAceAerAbs, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, self.device, dataset=self.dataset,
                             sample_selection_strategy=self.args.sample_selection_strategy)

        self.seen_so_far = torch.tensor([]).long().to(self.device)

    def reload_model_checkpoint(self):
        self.net.load_state_dict(self.past_model_ckpt)

    def save_model_checkpoint(self):
        self.past_model_ckpt = copy.deepcopy(self.net.state_dict())

    def begin_task(self, dataset: ContinualDataset) -> None:
        self.opt.zero_grad()

    def end_task(self, dataset):
        if self.args.buffer_fitting_epochs > 0:
            self.past_model_ckpt = None
            if self.args.buffer_fitting_type == 'simple':
                self.fit_buffer_ce()
            elif self.args.buffer_fitting_type == 'mixmatch':
                self.fit_buffer_mixmatch()
            else:
                raise ValueError(f'Unknown buffer fitting type: {self.args.buffer_fitting_type}')

    def is_aer_fitting_epoch(self, epoch):
        # fit the buffer only during odd epochs or the last epoch (i.e., start with no replay and end with replay)
        return epoch % 2 == 1 or epoch == self.args.n_epochs - 1

    def begin_epoch(self, epoch: int, dataset: ContinualDataset):
        torch.cuda.empty_cache()

        if self.is_aer_fitting_epoch(epoch) and hasattr(self, 'past_model_ckpt') and self.past_model_ckpt is not None:
            # this is a learning epoch, restore the checkpoint to when the buffer was fitted
            self.reload_model_checkpoint()

    def end_epoch(self, epoch: int, dataset: ContinualDataset):
        if self.is_aer_fitting_epoch(epoch):
            # the epoch was a buffer fitting epoch, save the model checkpoint
            self.save_model_checkpoint()

    def observe(self, inputs, labels, not_aug_inputs, epoch, true_labels):

        present = labels.unique()

        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)

        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        with torch.no_grad():
            not_aug_logits = self.net(apply_transform(not_aug_inputs, self.normalization_transform))
            not_aug_logits = not_aug_logits.masked_fill(mask == 0, torch.finfo(not_aug_logits.dtype).min)
            loss_not_aug_ext = self.loss(not_aug_logits, labels, reduction='none')

        loss = self.loss(logits, labels)

        loss_re = torch.tensor(0.)

        if not self.buffer.is_empty():
            # always sample from buffer (needed to update scores)
            buf_indexes, not_aug_buf_inputs, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, return_not_aug=True,
                not_aug_transform=self.normalization_transform)

            if self.args.sample_selection_strategy != 'reservoir':
                with torch.no_grad():
                    # update scores for existing samples with not augmented data for best logits
                    not_aug_buf_logits = self.net(not_aug_buf_inputs)
                    loss_not_aug_re_ext = self.loss(not_aug_buf_logits, buf_labels, reduction='none')
                    self.buffer.sample_selection_fn.update(buf_indexes, loss_not_aug_re_ext)

            # replay if AER is disabled or if epoch is odd (or last)
            # if not self.args.use_aer or self.is_aer_fitting_epoch(epoch):
            buf_logits = self.net(buf_inputs)
            loss_re = self.loss(buf_logits, buf_labels)

            if self.args.use_aer and epoch % 2 == 0:
                loss_re = torch.tensor(0.)

        loss += loss_re
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.opt.step()

        with torch.no_grad():
            # update buffer if not using AER or only during buffer forgetting epochs (i.e., when the buffer is NOT optimized)
            if not self.args.use_aer or (self.args.use_aer and not self.is_aer_fitting_epoch(epoch)):
                # not_aug_logits = self.net(apply_transform(not_aug_inputs, self.normalization_transform))
                # not_aug_logits = not_aug_logits.masked_fill(mask == 0, torch.finfo(not_aug_logits.dtype).min)
                # loss_not_aug_ext = self.loss(not_aug_logits, labels, reduction='none')

                # sample insertion
                _, clean_mask = torch.topk(loss_not_aug_ext, round((1 - self.args.alpha_sample_insertion) * inputs.shape[0]), largest=False)

                self.buffer.add_data(examples=not_aug_inputs[clean_mask],
                                     labels=labels[clean_mask],
                                     sample_selection_scores=loss_not_aug_ext[clean_mask] if self.args.sample_selection_strategy != 'reservoir' else None)

        return loss.item()

    # ------------------ OPTIONAL STUFF FOR BUFFER FITTING ---------------------

    @torch.no_grad()
    def pseudo_label(self, net: nn.Module, not_aug_inputs: torch.Tensor, orig_labels: torch.Tensor, corr_probs: torch.Tensor, T=0.5):
        """
        Pseudo-label generation for MixMatch:
        1. Augment the unlabeled data
        2. Get the output of the model
        3. Compute the average of the output
        4. Sharpen the output

        Additional step:
        - Compute the pseudo label as the weighted sum of the original label and the sharpened output, with the weight being the probability of the original label being correct
        """
        N = self.args.mixmatch_naug_buffer_fitting
        # ------------------ PSEUDO LABEL ---------------------
        was_training = net.training
        net.eval()

        unsup_aug_inputs = apply_transform(not_aug_inputs.repeat_interleave(N, 0), self.transform)
        orig_labels = F.one_hot(orig_labels.repeat_interleave(N, 0), self.num_classes).float()
        corr_probs = corr_probs.repeat_interleave(N, 0)

        unsup_aug_outputs = self.net(unsup_aug_inputs).reshape(N, -1, self.num_classes).mean(0)
        unsup_sharp_outputs = unsup_aug_outputs ** (1 / T)
        unsup_norm_outputs = unsup_sharp_outputs / unsup_sharp_outputs.sum(1).unsqueeze(1)
        unsup_norm_outputs = unsup_norm_outputs.repeat(N, 1)

        pseudo_labels_u = corr_probs * orig_labels + (1 - corr_probs) * unsup_norm_outputs

        net.train(was_training)
        return pseudo_labels_u.float(), unsup_aug_inputs

    def mixmatch_epoch(self, net: nn.Module, opt: torch.optim.Optimizer, loader: DataLoader):
        net.train()
        with tqdm.tqdm(loader, desc=' - MixMatch epoch', leave=False) as pbar:
            for i, data in enumerate(pbar):
                if self.args.debug_mode and i > 10:
                    break
                inputs, labels, is_ambiguous, not_aug_inputs, corr_probs = data
                inputs, labels, is_ambiguous, not_aug_inputs, corr_probs = inputs.to(
                    self.device), labels.to(
                    self.device), is_ambiguous.to(
                    self.device), not_aug_inputs.to(
                    self.device), corr_probs.to(self.device)
                is_ambiguous = is_ambiguous.bool()
                corr_probs = corr_probs[:, 0].expand(-1, self.num_classes)

                N_SUP = len(labels[is_ambiguous])
                inputs_s = inputs[is_ambiguous]
                labels_s = F.one_hot(labels[is_ambiguous], self.num_classes).float()
                not_aug_inputs_u = not_aug_inputs[~is_ambiguous]

                # mixmatch
                if len(not_aug_inputs_u) > 0:
                    pseudo_labels_u, inputs_u = self.pseudo_label(net, not_aug_inputs_u, labels[~is_ambiguous], corr_probs[~is_ambiguous])
                    all_inputs = torch.cat([inputs_s, inputs_u], dim=0)
                    all_targets = torch.cat([labels_s, pseudo_labels_u], dim=0)
                else:
                    all_inputs = inputs_s
                    all_targets = labels_s

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                lamda = Beta(self.args.mixmatch_alpha_buffer_fitting, self.args.mixmatch_alpha_buffer_fitting).rsample((len(all_inputs),)).to(self.device)
                lamda = torch.max(lamda, 1 - lamda)
                lamda_inputs = lamda.reshape([lamda.shape[0]] + [1] * (len(input_a.shape) - 1))
                lamda_targets = lamda.reshape([lamda.shape[0]] + [1] * (len(target_a.shape) - 1))

                mixed_input = lamda_inputs * input_a + (1 - lamda_inputs) * input_b
                mixed_target = lamda_targets * target_a + (1 - lamda_targets) * target_b

                logits = net(mixed_input)
                mixed_target = mixed_target.to(logits.dtype)
                logits_x = logits[:N_SUP]
                logits_u = logits[N_SUP:]

                loss_sup = torch.tensor(0.)
                loss_unsup = torch.tensor(0.)
                if len(logits_x) > 0:
                    loss_sup = F.cross_entropy(logits_x, mixed_target[:N_SUP], reduction='mean')
                if len(logits_u) > 0:
                    loss_unsup = F.mse_loss(logits_u, mixed_target[N_SUP:], reduction='mean')
                loss = loss_sup + self.args.mixmatch_lambda_buffer_fitting * loss_unsup

                # compute gradient and do SGD step
                opt.zero_grad()
                loss.backward()
                opt.step()

                assert not torch.isnan(loss).any().item(), 'Loss is NaN'

                pbar.set_postfix({'loss': loss.item(), 'lr': opt.param_groups[0]['lr']}, refresh=False)

    @ torch.no_grad()
    def split_data(self, test_loader: DataLoader, model: nn.Module):
        CE = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        model = model.to(self.device)

        losses = torch.tensor([])
        # Compute the loss for each data point without augmentation
        for data in test_loader:
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = CE(outputs, targets)
            losses = torch.cat([losses, loss.detach().cpu()])
        losses = (losses - losses.min()) / ((losses.max() - losses.min()) + torch.finfo(torch.float32).eps)
        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)

        # get the probabilities of each data point belonging to each component
        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        pred = prob.argmax(axis=1)

        # the component with the lowest mean are probably clean
        correct_idx = np.where(pred == 0)[0]

        # the rest are ambiguous (may be noisy)
        amb_idx = np.where(pred == 1)[0]

        return correct_idx, amb_idx, prob

    def fit_buffer_mixmatch(self):
        # get all data from the buffer
        buf_data = self.buffer.get_all_data(device="cpu")
        inputs, labels = buf_data[0], buf_data[1]

        # Building train dataset
        train_dataset = CustomDataset(inputs, labels, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.args.buffer_fitting_batch_size, shuffle=True)

        # Building test dataset
        not_aug_dataset = CustomDataset(inputs, labels, transform=self.normalization_transform)
        not_aug_loader = DataLoader(not_aug_dataset, batch_size=self.args.buffer_fitting_batch_size, shuffle=False)

        # build optimizer
        opt = self.get_optimizer([p for p in self.net.parameters() if p.requires_grad], lr=self.args.buffer_fitting_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
        )

        for e in tqdm.trange(self.args.buffer_fitting_epochs, desc='Fitting on buffer with MixMatch'):
            _, amb_idxs, probs = self.split_data(test_loader=not_aug_loader, model=self.net)
            amb_idxs, probs = torch.from_numpy(amb_idxs), torch.from_numpy(probs)
            corr_lab = torch.ones(len(train_dataset))
            corr_lab[amb_idxs] = 0
            train_loader.dataset.set_probs(probs.to(train_loader.dataset.device))
            train_loader.dataset.extra = corr_lab.to(train_loader.dataset.device)
            self.mixmatch_epoch(self.net, opt, train_loader)

            scheduler.step()

    def fit_buffer_ce(self):

        inputs, labels = self.buffer.get_all_data()

        train_dataset = CustomDataset(inputs, targets=labels, transform=self.transform, device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)

        opt = torch.optim.SGD(self.net.parameters(), lr=self.args.buffer_fitting_lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
        )

        for e in tqdm.trange(self.args.buffer_fitting_epochs, desc='Fitting on buffer with CE'):
            for i, data in enumerate(train_loader):
                if self.args.debug_mode and i > 10:
                    break

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                opt.zero_grad()
                if self.args.enable_cutmix_buffer_fitting and np.random.rand(0, 1) < 0.5:
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, force=True)

                    logits = self.net(inputs)

                    loss = lam * self.loss(logits, labels_a) + (1 - lam) * self.loss(logits, labels_b)
                else:
                    logits = self.net(inputs)

                    loss_ext = self.loss(logits, labels, reduction='none')

                loss = loss_ext.mean()
                loss.backward()
                opt.step()

            scheduler.step()
