# this perturber model was based on the code by AWP (https://github.com/csdongxian/AWP) and of course, modified to our needs

import torch
import torch.nn.functional as F
from collections import OrderedDict
from backbone import get_backbone


def add_perturb_args(parser):
    parser.add_argument('--p-steps', type=int, default=1)
    parser.add_argument('--p-lam', type=float, default=0.01)
    parser.add_argument('--p-gamma', type=float, default=0.05, help='how far we can go from original weights')


EPS = 1E-20


def diff_in_weights(model, proxy):
    with torch.no_grad():
        diff_dict = OrderedDict()
        model_state_dict = model.state_dict()
        proxy_state_dict = proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_w.size()) <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff_dict[old_k] = diff_w  # old_w.norm() / (diff_w.norm() + EPS) *
        return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def normalize(perturbations, weights):
    perturbations.mul_(weights.norm() / (perturbations.norm() + EPS))


def normalize_grad(weights, ref_weights):
    with torch.no_grad():
        for w, ref_w in zip(weights, ref_weights):
            if w.dim() <= 1:
                w.grad.data.fill_(0)  # ignore perturbations with 1 dimension (e.g. BN, bias)
            else:
                normalize(w.grad.data, ref_w)


class Perturber():
    EPS = 1E-20

    def __init__(self, continual_model):
        self.continual_model = continual_model
        self.device = continual_model.device
        self.args = continual_model.args
        self.net = continual_model.net
        self.proxy = get_backbone(self.args).to(self.device)
        self.steps = self.args.p_steps
        self.lam = self.args.p_lam
        self.gamma = self.args.p_gamma
        self.diff = None

    def init_rand(self, model):
        with torch.no_grad():
            for w in model.parameters():
                if w.dim() <= 1:
                    continue
                else:
                    # z = torch.randn_like(w) # uncomment for random perturbations
                    # z = z/torch.linalg.norm(z) #
                    w.add_(torch.randn_like(w) * torch.norm(w) * EPS)  # z *torch.norm(w) * self.gamma) # This is changed for ablation

    def perturb_model(self, X, y):
        out_o = F.softmax(self.net(X), dim=-1).detach()
        self.proxy.load_state_dict(self.net.state_dict())

        # initialize small random noise (delta = 0 is global minimizer)
        self.init_rand(self.proxy)

        self.proxy.train()

        pertopt = torch.optim.SGD(self.proxy.parameters(), lr=self.gamma / self.steps)
        # perturb the model
        mask = torch.where(out_o.max(1)[1] == y, 1., 0.).detach()  # This is changed for ablation

        if mask.sum() < 2:
            return None, mask
        for idx in range(self.steps):  # to have multiple steps (set to 1 step by default)
            pertopt.zero_grad()
            loss = -(F.kl_div(F.log_softmax(self.proxy(X), dim=1), out_o, reduction='none').sum(dim=1) * mask).sum() / mask.sum()

            loss.backward()
            normalize_grad(self.proxy.parameters(), self.net.parameters())

            pertopt.step()

        # calculate the weight perturbation and add onto original network
        self.diff = diff_in_weights(self.net, self.proxy)
        add_into_weights(self.net, self.diff, coeff=1.0)
        return out_o, mask

    def get_loss(self, X, y):
        outs, mask = self.perturb_model(X, y)
        out_n = self.net(X)
        if outs is not None:
            loss_kl = self.lam * (F.kl_div(F.log_softmax(out_n, dim=1), outs, reduction='none').sum(dim=1) * mask).sum() / mask.sum()
            return loss_kl
        else:
            return None

    def restore_model(self):
        add_into_weights(self.net, self.diff, coeff=-1.0)

    def __call__(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        loss_kl = self.get_loss(X, y)
        if loss_kl is not None:
            loss_kl.backward()
            self.restore_model()
