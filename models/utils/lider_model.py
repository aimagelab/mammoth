"""
Base class for all models that use the Lipschitz regularization in LiDER (https://arxiv.org/pdf/2210.06443.pdf).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
from models.utils.continual_model import ContinualModel


def add_lipschitz_args(parser):
    # BUFFER LIP LOSS
    parser.add_argument('--alpha_lip_lambda', type=float, required=False, default=0,
                        help='Lambda parameter for lipschitz minimization loss on buffer samples')

    # BUDGET LIP LOSS
    parser.add_argument('--beta_lip_lambda', type=float, required=False, default=0,
                        help='Lambda parameter for lipschitz budget distribution loss')

    # Extra
    parser.add_argument('--headless_init_act', type=str, choices=["relu", "lrelu"], default="relu")
    parser.add_argument('--grad_iter_step', type=int, required=False, default=-2,
                        help='Step from which to enable gradient computation.')


class LiderOptimizer(ContinualModel):
    """
    Superclass for all models that use the Lipschitz regularization in LiDER (https://arxiv.org/pdf/2210.06443.pdf).
    """

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        if self.args.alpha_lip_lambda == 0 and self.args.beta_lip_lambda == 0:
            print("WARNING: LiDER is enabled but both `alpha_lip_lambda` and `beta_lip_lambda` are 0. LiDER will not be used.")

    def transmitting_matrix(self, fm1: torch.Tensor, fm2: torch.Tensor):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(-2), fm2.size(-1)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def compute_transition_matrix(self, front: torch.Tensor, latter: torch.Tensor):
        return torch.bmm(self.transmitting_matrix(front, latter), self.transmitting_matrix(front, latter).transpose(2, 1))

    def top_eigenvalue(self, K: torch.Tensor, n_power_iterations=10):
        """
        Compute the top eigenvalue of a matrix K using the power iteration method.
        Stop gradient propagation after `n_power_iterations`.

        Args:
            K (torch.Tensor): The matrix to compute the top eigenvalue of.
            n_power_iterations (int): The number of power iterations to run. If positive, compute gradient only for the first `n_power_iterations` iterations. If negative, compute gradient only for the last `n_power_iterations` iterations.

        Returns:
            torch.Tensor: The top eigenvalue of K.
        """
        if self.args.grad_iter_step < 0:
            start_grad_it = n_power_iterations + self.args.grad_iter_step + 1
        else:
            start_grad_it = self.args.grad_iter_step
        assert start_grad_it >= 0 and start_grad_it <= n_power_iterations

        v = torch.rand(K.shape[0], K.shape[1], 1).to(K.device, dtype=K.dtype)
        for itt in range(n_power_iterations):
            with torch.set_grad_enabled(itt >= start_grad_it):
                m = torch.bmm(K, v)
                n = (torch.norm(m, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps)
                v = m / n

        top_eigenvalue = torch.sqrt(n / (torch.norm(v, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps))
        return top_eigenvalue

    def get_layer_lip_coeffs(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lipschitz coefficient of a layer given its batches of input and output features.
        Estimates the Lipschitz coefficient with https://arxiv.org/pdf/2108.12905.pdf.

        Args:
            features_a (torch.Tensor): The batch of input features.
            features_b (torch.Tensor): The batch of output features.

        Returns:
            torch.Tensor: The Lipschitz coefficient of the layer.
        """
        features_a, features_b = features_a.double(), features_b.double()
        features_a, features_b = features_a / self.get_norm(features_a), features_b / self.get_norm(features_b)

        TM_s = self.compute_transition_matrix(features_a, features_b)
        L = self.top_eigenvalue(K=TM_s)
        return L

    def get_feature_lip_coeffs(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute the Lipschitz coefficient for all the layers of a network given a list of batches of features.
        The features are assumed to be ordered from the input to the output of the network.

        Args:
            features (List[torch.Tensor]): The list features of each layer.

        Returns:
            List[torch.Tensor]: The list of Lipschitz coefficients for each layer.
        """
        N = len(features) - 1
        B = len(features[0])

        lip_values = [torch.zeros(B, device=self.device, dtype=features[0].dtype)] * N

        for i in range(N):
            fma, fmb = features[i], features[i + 1]
            fmb = F.adaptive_avg_pool1d(fmb.reshape(*fmb.shape[:2], -1).permute(0, 2, 1), fma.shape[1]).permute(0, 2, 1).reshape(fmb.shape[0], -1, *fmb.shape[2:])
            L = self.get_layer_lip_coeffs(fma, fmb)

            L = L.reshape(B)

            lip_values[i] = L
        return lip_values

    @torch.no_grad()
    def init_net(self, dataset):
        """
        Compute the target Lipschitz coefficients for the network and initialize the network's Lipschitz coefficients to match them.

        Args:
            dataset (ContinualDataset): The dataset to use for the computation.
        """
        was_training = self.net.training
        self.net.eval()

        all_lips = []
        for i, (inputs, labels, _) in enumerate(tqdm(dataset.train_loader, desc="Computing target L budget")):
            if self.args.debug_mode and i > self.get_debug_iters():
                continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if len(inputs.shape) == 5:
                B, n, C, H, W = inputs.shape
                inputs = inputs.view(B * n, C, H, W)

            _, partial_features = self.net(inputs, returnt='full')

            lip_inputs = [inputs] + partial_features

            lip_values = self.get_feature_lip_coeffs(lip_inputs)
            # (B, F)
            lip_values = torch.stack(lip_values, dim=1)

            all_lips.append(lip_values)

        budget_lip = torch.cat(all_lips, dim=0).mean(0).detach().clone()

        inp = next(iter(dataset.train_loader))[0]
        _, teacher_feats = self.net(inp.to(self.device), returnt='full')

        self.net.lip_coeffs = torch.autograd.Variable(torch.randn(len(teacher_feats), dtype=torch.float), requires_grad=True).to(self.device)
        self.net.lip_coeffs.data = budget_lip

        self.net.train(was_training)

    def get_norm(self, t: torch.Tensor):
        """
        Compute the norm of a tensor.

        Args:
            t (torch.Tensor): The tensor.

        Returns:
            torch.Tensor: The norm of the tensor.
        """
        return torch.norm(t, dim=1, keepdim=True) + torch.finfo(torch.float32).eps

    def minimization_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Lipschitz minimization loss for a batch of features (eq. 8).

        Args:
            features (List[torch.Tensor]): The list features of each layer. The features are assumed to be ordered from the input to the output of the network.

        Returns:
            torch.Tensor: The Lipschitz minimization loss.
        """
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        return lip_values.mean()

    def dynamic_budget_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the dynamic budget Lipschitz loss for a batch of features (eq. 7).

        Args:
            features (List[torch.Tensor]): The list features of each layer. The features are assumed to be ordered from the input to the output of the network.

        Returns:
            torch.Tensor: The dynamic budget Lipschitz loss.
        """
        loss = 0
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        if self.args.headless_init_act == "relu":
            tgt = F.relu(self.net.lip_coeffs[:len(lip_values[0])])
        elif self.args.headless_init_act == "lrelu":
            tgt = F.leaky_relu(self.net.lip_coeffs[:len(lip_values[0])])
        else:
            raise NotImplementedError

        tgt = tgt.unsqueeze(0).expand(lip_values.shape)

        loss += F.l1_loss(lip_values, tgt)

        return loss
