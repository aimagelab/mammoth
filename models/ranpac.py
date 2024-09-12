"""
Slow Learner with Classifier Alignment.

Note:
    SLCA USES A CUSTOM BACKBONE (see `feature_extractor_type` argument)

Arguments:
    --feature_extractor_type: the type of convnet to use. `vit-b-p16` is the default: ViT-B/16 pretrained on Imagenet 21k (**NO** finetuning on ImageNet 1k)
"""

import copy

import numpy as np
from models.ranpac_utils.toolkit import target2onehot
from utils import binary_to_boolean_type
from utils.args import *
from models.utils.continual_model import ContinualModel

import torch
import torch.nn.functional as F
from utils.conf import get_device
from models.ranpac_utils.ranpac import RanPAC_Model


class RanPAC(ContinualModel):
    """RanPAC: Random Projections and Pre-trained Models for Continual Learning."""
    NAME = 'ranpac'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']
    net: RanPAC_Model

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(pretrain_type='in21k')
        parser.set_defaults(optim_mom=0.9, optim_wd=0.0005, batch_size=48)
        parser.add_argument('--rp_size', type=int, default=10000, help='size of the random projection layer (L in the paper)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        self.device = get_device()
        print("-" * 20)
        print(f"WARNING: RanPAC USES `in21k` AS DEFAULT PRETRAIN. CHANGE IT WITH `--pretrain_type` IF NEEDED.")
        backbone = RanPAC_Model(backbone, args)
        print("-" * 20)

        super().__init__(backbone, loss, args, transform, dataset=dataset)

    def get_parameters(self):
        return self.net._network.parameters()

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.n_epochs, eta_min=0)

    def end_task(self, dataset):
        if self.current_task == 0:
            self.freeze_backbone()
            self.setup_RP()
        dataset.train_loader.dataset.transform = self.dataset.TEST_TRANSFORM
        self.replace_fc(dataset.train_loader)

    def setup_RP(self):
        self.net._network.fc.use_RP = True

        # RP with M > 0
        M = self.args.rp_size
        self.net._network.fc.weight = torch.nn.Parameter(torch.Tensor(self.net._network.fc.out_features, M).to(self.net._network.device))  # num classes in task x M
        self.net._network.fc.reset_parameters()
        self.net._network.fc.W_rand = torch.randn(self.net._network.fc.in_features, M).to(self.net._network.device)
        self.W_rand = copy.deepcopy(self.net._network.fc.W_rand)  # make a copy that gets passed each time the head is replaced

        self.Q = torch.zeros(M, self.dataset.N_CLASSES)
        self.G = torch.zeros(M, M)

    def replace_fc(self, trainloader):
        self.net._network.eval()

        # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
        self.net._network.fc.use_RP = True
        self.net._network.fc.W_rand = self.W_rand

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                data, label = data[0].to(self.device), data[1].to(self.device)
                embedding = self.net._network.convnet(data)

                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.dataset.N_CLASSES)
        # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
        Features_h = torch.nn.functional.relu(Features_f @ self.net._network.fc.W_rand.cpu())

        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T  # better nmerical stability than .inv
        self.net._network.fc.weight.data = Wo[0:self.net._network.fc.weight.shape[0], :].to(self.net._network.device)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0**np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: " + str(ridge))
        return ridge

    def begin_task(self, dataset):
        # temporarily remove RP weights
        del self.net._network.fc
        self.net._network.fc = None

        self.net._network.update_fc(self.n_seen_classes)  # creates a new head with a new number of classes (if CIL)

        if self.current_task == 0:
            self.opt = self.get_optimizer()
            self.scheduler = self.get_scheduler()
            self.opt.zero_grad()

    def freeze_backbone(self, is_first_session=False):
        # Freeze the parameters for ViT.
        if isinstance(self.net._network.convnet, torch.nn.Module):
            for name, param in self.net._network.convnet.named_parameters():
                if is_first_session:
                    if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.current_task == 0:  # simple train on first task
            logits = self.net._network(inputs)["logits"]
            loss = self.loss(logits, labels)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss.item()

        return 0

    def forward(self, x):
        return self.net._network(x)['logits']
