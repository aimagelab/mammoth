import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from models.ranpac_utils.inc_net import RanPACNet
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from datasets import get_dataset
import sys

from models.ranpac_utils.toolkit import target2onehot


class RanPAC_Model(object):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        self._network = RanPACNet(backbone)

    @property
    def training(self):
        return self._network.training

    def to(self, device):
        self._network.to(device)

    def train(self, *args):
        self._network.train(*args)

    def eval(self):
        self._network.eval()

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP = True
            if self.args['M'] > 0:
                self._network.fc.W_rand = self.W_rand
            else:
                self._network.fc.W_rand = None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._network.device)
                label = label.to(self._network.device)
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)
        if self.args['use_RP']:
            # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M'] > 0:
                Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:
                Features_h = Features_f
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h
            ridge = self.optimise_ridge_parameter(Features_h, Y)
            Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T  # better nmerical stability than .inv
            self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :].to(self._network.device)
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index] += class_prototype.to(self._network.device)  # for dil, we update all classes in all tasks
                else:
                    # original cosine similarity approach of Zhou et al (2023)
                    class_prototype = Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index] = class_prototype  # for cil, only new classes get updated

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
