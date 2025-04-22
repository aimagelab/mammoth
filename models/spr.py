import tqdm
import copy
import logging
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import networkx as nx

from backbone import get_backbone
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args
from utils.bmm import BetaMixture1D
from utils.buffer import Buffer
from utils.conf import get_device
from utils.distributed import make_dp, CustomDP
from utils.kornia_utils import to_kornia_transform


def _get_projector_prenet(net, device=None, bn=True):
    device = net.device if hasattr(net, 'device') else device if device is not None else "cpu"
    assert "resnet" in type(net).__name__.lower() or "mnistmlp" in type(net).__name__.lower(), "Only resnet and simple MLP are supported for now"

    if "resnet" in type(net).__name__.lower():
        sizes = [net.nf * 8, net.nf * 8, 256]
    else:
        sizes = [net.classifier.in_features, net.classifier.in_features, 256]

    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True).to(device))
        if bn:
            layers.append(nn.BatchNorm1d(sizes[i + 1]).to(device))
        layers.append(nn.ReLU(inplace=True).to(device))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True).to(device))
    return nn.Sequential(*layers).to(device)


def init_simclr_net(model, device=None):
    model.projector = _get_projector_prenet(model, device=device, bn=False)
    model.predictor = copy.deepcopy(model.projector)
    return model


class SimCLR:
    def __init__(self, transform, temp=0.5, eps=1e-6, filter_bs_len=None, correlation_mask=None):
        self.temp = temp
        self.eps = eps
        self.filter_bs_len = filter_bs_len
        self.correlation_mask = correlation_mask
        self.transform = transform

    def __call__(self, model, x):
        with torch.no_grad():
            xa = self.transform(x)
            xb = self.transform(x)

        outa = model.projector(model(xa))
        outb = model.projector(model(xb))

        outa = F.normalize(outa, dim=1)
        outb = F.normalize(outb, dim=1)

        out = torch.cat([outb, outa], dim=0)

        cov = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)

        # filter out the scores from the positive samples
        l_pos = torch.diag(cov, self.filter_bs_len)
        r_pos = torch.diag(cov, -self.filter_bs_len)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.filter_bs_len, 1)
        negatives = cov[self.correlation_mask].view(2 * self.filter_bs_len, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp

        labels = torch.zeros(2 * self.filter_bs_len).to(cov.device).long()
        loss = F.cross_entropy(logits, labels, reduction='sum') / (2 * self.filter_bs_len)

        return loss


def disable_linear(backbone):
    # disable linear base net
    in_features = backbone.classifier.in_features
    out_features = backbone.classifier.out_features
    backbone.classifier = nn.Identity()
    backbone.classifier.in_features = in_features
    backbone.classifier.out_features = out_features

    return backbone


class Spr(ContinualModel):
    """
    Implementation of `Continual Learning on Noisy Data Streams via Self-Purified Replay <https://github.com/ecrireme/SPR>`_ from ICCV 2021.
    """
    NAME = 'spr'
    COMPATIBILITY = ['class-il', 'task-il']
    OVERRIDE_SUPPORT_DISTRIBUTED = True

    @staticmethod
    def get_parser(parser):
        parser.set_defaults(optimizer='adam', lr=0.0002, num_workers=0)
        add_rehearsal_args(parser)

        parser.add_argument('--spr_debug_mode', type=binary_to_boolean_type, default=False, help='Run SPR with just a few iterations?')
        parser.add_argument('--spr_custom_dp', type=binary_to_boolean_type, default=False, help='Use DataParallel?')
        parser.add_argument('--delayed_buffer_size', type=int,
                            help='Size of the delayed buffer. If `None`, it will be set to the buffer size.')
        parser.add_argument('--fitting_lr', type=float, default=0.002,
                            help='LR used during finetuining (classifier buffer fitting on P)')
        parser.add_argument('--fitting_epochs', type=int, default=50,
                            help='Number of epochs used during finetuining (classifier buffer fitting on P)')
        parser.add_argument('--inner_train_epochs', type=int, default=3000,
                            help='Inner train epochs for SSL (base net)')
        parser.add_argument('--expert_train_epochs', type=int, default=4000,
                            help='Innert train epochs for SSL (expert)')
        parser.add_argument('--simclr_temp', type=float, default=0.5,
                            help='Temperature for simclr SSL loss')
        parser.add_argument('--fitting_sched_lr_stepsize', type=int, default=300,
                            help='Step size for the LR scheduler during finetuining (classifier buffer fitting on P)')
        parser.add_argument('--fitting_sched_lr_gamma', type=float, default=0.1,
                            help='Gamma for the LR scheduler during finetuining (classifier buffer fitting on P)')
        parser.add_argument('--fitting_batch_size', type=int, default=16,
                            help='Batch size for finetuining (classifier buffer fitting on P)')
        parser.add_argument('--fitting_clip_value', type=float, default=0.5,
                            help='Gradient clipping for finetuning')
        parser.add_argument('--E_max', type=int, default=5,
                            help='Number of stochastic ensemble for expert')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert args.distributed == 'no', "To use SPR with distributed use `--spr_custom_dp=1`."

        if args.spr_custom_dp:
            get_device.device = 'cuda:0'

        args.delayed_buffer_size = args.buffer_size if args.delayed_buffer_size is None else args.delayed_buffer_size
        cl_in_features = backbone.classifier.in_features
        # disable linear base net
        backbone = disable_linear(backbone)
        init_simclr_net(backbone)

        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.cl_in_features = cl_in_features
        self.buffer = Buffer(self.args.buffer_size, "cpu")
        self.delayed_buffer = Buffer(self.args.delayed_buffer_size, "cpu")

        self.past_loss = 0

        self.get_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.inner_train_epochs, eta_min=0, last_epoch=-1)

        self.finetuned_model = get_backbone(args)
        missing, ignored = self.finetuned_model.load_state_dict(copy.deepcopy(self.net.state_dict()), strict=False)
        assert len([m for m in missing if 'classifier' not in m and 'fc' not in m]) == 0, missing
        self.finetuned_model.classifier = nn.Linear(self.cl_in_features, self.num_classes).to(self.device)
        self.finetuned_model.to(self.device)

        self.expert_model = get_backbone(args)
        self.expert_model = disable_linear(self.expert_model)
        init_simclr_net(self.expert_model)
        missing, ignored = self.expert_model.load_state_dict(copy.deepcopy(self.net.state_dict()), strict=False)
        assert len([m for m in missing if 'classifier' not in m and 'fc' not in m]) == 0, missing
        self.expert_model.to(self.device)
        self.expert_opt = self.get_optimizer(self.expert_model.parameters())
        self.expert_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.expert_opt, T_max=self.args.expert_train_epochs, eta_min=0, last_epoch=-1)

        self.expert_model.to("cpu")
        self.net.to("cpu")
        self.finetuned_model.to("cpu")

    @torch.no_grad()
    def cluster_and_sample(self):
        """filter samples in delay buffer"""
        self.expert_model.eval()
        self.expert_model.to(self.device)

        self.delayed_buffer.to(self.device)
        xs = self.delayed_buffer.examples
        ys = self.delayed_buffer.labels
        tls = self.delayed_buffer.true_labels
        corrs = tls == ys

        features = self.expert_model.projector(self.expert_model(xs))
        features = F.normalize(features, dim=1)

        clean_idx, clean_p = [], []

        noisy_samples_selected, corr_samples_selected = 0, 0

        for u_y in torch.unique(ys):
            y_mask = ys == u_y
            corr = corrs[y_mask]
            feature = features[y_mask]

            # ignore negative similairties
            _similarity_matrix = torch.relu(F.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=-1))

            # stochastic ensemble
            _clean_ps = torch.zeros((self.args.E_max, len(feature)), dtype=torch.double)
            for _i in range(self.args.E_max):
                # sample binary adjacency matrix from bernoulli distribution (of similarity matrix)
                similarity_matrix = (_similarity_matrix > torch.rand_like(_similarity_matrix)).type(torch.float32)
                similarity_matrix[similarity_matrix == 0] = 1e-5  # add small num for ensuring positive matrix

                # get centrality
                g = nx.from_numpy_array(similarity_matrix.cpu().numpy())
                info = nx.eigenvector_centrality(g, max_iter=6000, weight='weight')  # index: value
                centrality = np.asarray(list(info.values()))

                # fit BMM
                bmm_model = BetaMixture1D(max_iters=10)
                # fit beta mixture model
                c = np.copy(centrality)
                c, c_min, c_max = bmm_model.outlier_remove(c)
                c = bmm_model.normalize(c, c_min, c_max)
                bmm_model.fit(c)
                bmm_model.create_lookup(1)  # 0: noisy, 1: clean

                # get posterior
                c = np.copy(centrality)
                c = bmm_model.normalize(c, c_min, c_max)
                p = bmm_model.look_lookup(c)
                _clean_ps[_i] = torch.from_numpy(p)

            _clean_ps = torch.mean(_clean_ps, dim=0)
            m = _clean_ps > torch.rand_like(_clean_ps)

            clean_idx.extend(torch.nonzero(y_mask)[:, -1][m].tolist())
            clean_p.extend(_clean_ps[m].tolist())

            corr_samples_selected += corr[m].sum().item()
            noisy_samples_selected += (~corr)[~m].sum().item()

        return clean_idx, torch.Tensor(clean_p), corr_samples_selected, noisy_samples_selected

    def get_strong_transform(self):
        """Get strong transform for the base and expert network"""
        if self.transform is None:
            return lambda x: x
        if isinstance(self.original_transform.transforms, transforms.Compose):
            tr = self.original_transform.transforms[-1].transforms[:-2]
        else:
            tr = self.original_transform.transforms
        return to_kornia_transform(tr)

    def train_self_expert(self):
        """Train expert model with samples from delay buffer only"""
        self.finetuned_model.to("cpu")
        self.net.to("cpu")

        # reset expert model
        nt = get_backbone(self.args)
        nt = disable_linear(nt)
        nt = init_simclr_net(nt)

        missing, ignored = self.expert_model.load_state_dict(nt.state_dict())
        assert len([m for m in missing if 'classifier' not in m and 'fc' not in m]) == 0, missing

        self.expert_model.to(self.device)

        if self.args.spr_custom_dp and not isinstance(self.expert_model, CustomDP):  # initialize DP only once
            self.expert_model.to('cuda:0')
            self.expert_model = make_dp(self.expert_model)

        torch.cuda.empty_cache()

        def _get_correlated_mask(bs):
            diag = np.eye(2 * bs)
            l1 = np.eye((2 * bs), 2 * bs, k=-bs)
            l2 = np.eye((2 * bs), 2 * bs, k=bs)
            mask = torch.from_numpy((diag + l1 + l2))
            mask = (1 - mask).type(torch.bool)
            return mask

        # total batch size = buffer size (delay only)
        bs = min(self.args.delayed_buffer_size, len(self.delayed_buffer))

        self.expert_model.train()
        correlation_mask = _get_correlated_mask(bs).to(self.device)
        tr = self.get_strong_transform()  # SPR does not use normalization for the expert network
        loss_fn = SimCLR(tr, temp=self.args.simclr_temp, filter_bs_len=bs, correlation_mask=correlation_mask)

        self.delayed_buffer.to(self.device)
        dset = torch.utils.data.TensorDataset(self.delayed_buffer.examples, self.delayed_buffer.labels)
        delayed_sampler = torch.utils.data.RandomSampler(dset, replacement=True)
        delayed_dl = torch.utils.data.DataLoader(dset, batch_size=bs, drop_last=False, sampler=delayed_sampler)
        totloss, cit = 0, 0
        for epoch_i in tqdm.trange(self.args.expert_train_epochs, desc="Expert network training", leave=False):
            if self.args.spr_debug_mode == 1 and epoch_i > 10:
                break
            for data in delayed_dl:
                inputs = data[0].to(self.device)
                self.expert_opt.zero_grad()

                loss = loss_fn(self.expert_model, inputs)
                loss.backward()
                self.expert_opt.step()

                totloss += loss.item()
                cit += 1

            # warmup for the first 10 epochs
            if epoch_i >= 10:
                self.expert_lr_scheduler.step()

        return totloss / cit

    def train_self_base(self):
        """Self Replay. train base model with samples from delay and purified buffer"""
        self.expert_model.to("cpu")
        self.finetuned_model.to("cpu")
        self.net.to(self.device)

        def _get_correlated_mask(bs):
            diag = np.eye(2 * bs)
            l1 = np.eye((2 * bs), 2 * bs, k=-bs)
            l2 = np.eye((2 * bs), 2 * bs, k=bs)
            mask = torch.from_numpy((diag + l1 + l2))
            mask = (1 - mask).type(torch.bool)
            return mask

        if self.args.spr_custom_dp and not isinstance(self.net, CustomDP):  # initialize DP only once
            self.net = make_dp(self.net)

        # total batch size = buffer size (splitted btw delay and purified)
        bs = self.args.buffer_size
        # If purified buffer is full, train using it also
        db_bs = (bs // 2) if self.buffer.is_full() else bs
        db_bs = min(db_bs, len(self.delayed_buffer))
        pb_bs = min(bs - db_bs, len(self.buffer))

        self.net.train()
        correlation_mask = _get_correlated_mask(db_bs + pb_bs).to(self.device)
        tr = self.get_strong_transform()  # SPR does not use normalization for the base network
        loss_fn = SimCLR(tr, temp=self.args.simclr_temp, filter_bs_len=db_bs + pb_bs, correlation_mask=correlation_mask)

        totloss, cit = 0, 0

        dset = torch.utils.data.TensorDataset(self.delayed_buffer.examples, self.delayed_buffer.labels)
        delayed_dl = torch.utils.data.DataLoader(dset, batch_size=db_bs, drop_last=False, shuffle=True)
        for epoch_i in tqdm.trange(self.args.inner_train_epochs, desc="Base network training", leave=False):
            if self.args.spr_debug_mode == 1 and epoch_i > 10:
                break
            for data in delayed_dl:
                x = data[0].to(self.device)

                if pb_bs > 0:
                    xp = self.buffer.get_data(pb_bs)[0].to(self.device)
                    x = torch.cat([x, xp], dim=0)

                self.opt.zero_grad()

                loss = loss_fn(self.net, x)
                loss.backward()
                self.opt.step()

                totloss += loss.item()
                cit += 1

            # warmup for the first 10 epochs
            if epoch_i >= 10:
                self.lr_scheduler.step()

        return totloss / cit

    def begin_task(self, dataset):
        self.observe_it = 0

    def _observe(self, not_aug_x, y, true_y):
        self.delayed_buffer.add_data(examples=not_aug_x.unsqueeze(0),
                                     labels=y.unsqueeze(0),
                                     true_labels=true_y.unsqueeze(0))

        avg_expert_loss, avg_self_loss = -1, -1
        if self.delayed_buffer.is_full():

            self.observe_it += 1

            # Train expert net with SSL on D only
            avg_expert_loss = self.train_self_expert()

            # Train base net with SSL on D and P
            avg_self_loss = self.train_self_base()

            pret = time.time()
            # Get clean samples from D
            clean_idx, _, corr_sel, noisy_sel = self.cluster_and_sample()

            # Add clean samples to P
            self.buffer.add_data(examples=self.delayed_buffer.examples[clean_idx],
                                 labels=self.delayed_buffer.labels[clean_idx],
                                 true_labels=self.delayed_buffer.true_labels[clean_idx])

            logging.debug("Purifying buffer took", time.time() - pret)

            self.delayed_buffer.empty()

        return avg_expert_loss, avg_self_loss

    def fit_buffer(self):
        """Fit finetuned model on purified buffer, before eval"""
        logging.debug("Fitting finetuned model on purified buffer")
        self.expert_model.to("cpu")
        self.net.to("cpu")

        self.finetuned_model = get_backbone(self.args)
        missing, ignored = self.finetuned_model.load_state_dict(copy.deepcopy(self.net.state_dict()), strict=False)
        assert len([m for m in missing if 'classifier' not in m and 'fc' not in m]) == 0
        assert len([m for m in ignored if 'projector' not in m and 'predictor' not in m]) == 0
        self.finetuned_model.classifier = nn.Linear(self.cl_in_features, self.n_seen_classes).to(self.device)
        self.finetuned_model.to(self.device)

        opt = self.get_optimizer(self.finetuned_model.parameters(), lr=self.args.fitting_lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.args.fitting_sched_lr_stepsize, gamma=self.args.fitting_sched_lr_gamma)

        sampler = torch.utils.data.RandomSampler(self.buffer)
        buffer_dl = self.buffer.get_dataloader(self.args, batch_size=self.args.fitting_batch_size, drop_last=True, sampler=sampler)  # NO TRANSFORM

        self.finetuned_model.train()
        ce_loss = nn.NLLLoss()
        for epoch in tqdm.trange(self.args.fitting_epochs, desc="Buffer fitting", leave=False, disable=True):
            if self.args.spr_debug_mode == 1 and epoch > 10:
                break
            for dat in buffer_dl:
                x, y = dat[0], dat[1]
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = self.finetuned_model(x)
                loss = ce_loss(F.log_softmax(out, dim=1), y)
                loss.backward()
                if self.args.fitting_clip_value is not None:
                    nn.utils.clip_grad_value_(self.parameters(), self.args.fitting_clip_value)
                opt.step()
                sched.step()
        self.finetuned_model.eval()

    def forward(self, inputs):
        if self.finetuned_model.device != inputs.device:
            self.finetuned_model.to(inputs.device)
        return self.finetuned_model(inputs)

    def end_task(self, dataset):
        # fit classifier on P
        self.fit_buffer()
        self.buffer.to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, true_labels):
        for y, not_aug_x, true_y in zip(labels, not_aug_inputs, true_labels):  # un-batch the data
            avg_expert_loss, avg_self_loss = self._observe(not_aug_x, y, true_y)

        return self.past_loss if avg_self_loss < 0 else avg_self_loss
