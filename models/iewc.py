"""
Implementation of Improved Elastic Weight Consolidation (IEWC) from the paper
"Improved Elastic Weight Consolidation as an Optimization Constraint for Continual Learning"
Paper: https://arxiv.org/abs/2611.16039

IEWC keeps the online EWC penalty of `ewc_on` and only changes the diagonal importance.
Each old sample contributes its squared parameter gradient divided by its own squared
output-gradient norm, so well-fitted samples are not down-weighted by their small loss gradient:

    F = sum_n h_n * g_n^2 / (s_n + tau) / sum_n h_n

where `g_n` is the per-sample gradient of the loss w.r.t. the parameters, `s_n = ||dl_n / df_n||^2`
(see `IEWC.output_scale`), `tau = kappa * median{s_n : s_n > 0}` and `h_n` is either 1 (`uniform`),
the residual-diversity weight (`rdw`), or given by a custom function (see `IEWC.get_sample_weighting`).

As in `ewc_on`, the penalty is `e_lambda * sum_i F_i (theta_i - theta*_i)^2`, so `e_lambda`
corresponds to `lambda / 2` in the paper. The importance has a different scale than in `ewc_on`,
so `e_lambda` has to be tuned separately. Unlike `ewc_on`, samples are not weighted by p(y|x)
and the importance is computed with the backbone in eval mode.
"""

import torch

from models.ewc_on import EwcOn
from utils.args import ArgumentParser


def squared_norm(output_grads: torch.Tensor) -> torch.Tensor:
    """Default output geometry (identity metric): `s_n = ||r_n||^2` for each row `r_n`."""
    return output_grads.flatten(1).pow(2).sum(1)


class ResidualBasis:
    """
    Streaming residual-diversity weights (RDW).

    Keeps an orthonormal basis of at most `max_size` earlier residual directions (on CPU)
    and returns, for each new gradient, the squared norm of its unit direction outside that span.
    """

    def __init__(self, dim: int, max_size: int = 64, eps: float = 1e-12):
        self.basis = torch.zeros(max_size, dim)
        self.size = 0
        self.eps = eps

    def weight(self, grad: torch.Tensor) -> float:
        norm = grad.norm()
        if norm == 0:
            return 0.
        residual = (grad / norm).detach().float().cpu()
        if self.size > 0:
            basis = self.basis[:self.size]
            residual = residual - basis.t() @ (basis @ residual)
        h = residual.pow(2).sum().item()
        if h > self.eps and self.size < len(self.basis):
            self.basis[self.size] = residual / h ** 0.5
            self.size += 1
        return h


def iewc_importance(net, loss_fn, batches, kappa=1e-2, sample_weighting='uniform', rdw_max_basis=64,
                    output_scale=squared_norm):
    """
    Computes the diagonal IEWC importance.

    Args:
        net: the backbone (a `MammothBackbone`)
        loss_fn: the per-sample loss, called as `loss_fn(outputs, labels)` on single-sample batches
        batches: a list of `(inputs, labels)` batches; both passes use exactly these inputs
        kappa: relative damping, `tau = kappa * median{s_n : s_n > 0}`
        sample_weighting: `uniform`, `rdw`, or a function `fn(x, y, flat_grad) -> float` called once per
            sample in order, returning a non-negative (unnormalized) weight
        rdw_max_basis: maximum number of directions kept by RDW
        output_scale: maps the output gradients `(B, ...)` to the scales `s_n` `(B,)`

    Returns:
        the importance (same shape as `net.get_params()`) and the mean-normalized sample weights
    """
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()

    # pass 1: output gradients only, to get the scales s_n and the damping tau
    scales = []
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        outputs.requires_grad_(True)
        loss = sum(loss_fn(outputs[i:i + 1], labels[i:i + 1]) for i in range(len(outputs)))
        output_grads = torch.autograd.grad(loss, outputs)[0]
        scales.append(output_scale(output_grads).detach().float().cpu())
    scales = torch.cat(scales)
    positive = scales[scales > 0]
    tau = kappa * positive.quantile(0.5).item() if len(positive) > 0 else 0.

    # pass 2: per-sample parameter gradients
    fish = torch.zeros_like(net.get_params())
    if not callable(sample_weighting) and sample_weighting not in ('uniform', 'rdw'):
        raise ValueError(f'Unknown sample weighting: {sample_weighting}')
    basis = ResidualBasis(fish.numel(), rdw_max_basis) if sample_weighting == 'rdw' else None
    weights = torch.ones(len(scales))
    n = 0
    for inputs, labels in batches:
        for ex, lab in zip(inputs.to(device), labels.to(device)):
            net.zero_grad()
            loss = loss_fn(net(ex.unsqueeze(0)), lab.unsqueeze(0))
            loss.backward()
            grads = net.get_grads()
            if basis is not None:
                weights[n] = basis.weight(grads)
            elif callable(sample_weighting):
                weights[n] = float(sample_weighting(ex, lab, grads))
                if weights[n] < 0:
                    raise ValueError('Sample weights must be non-negative')
            denom = scales[n].item() + tau
            if denom > 0:
                fish += weights[n].item() * grads ** 2 / denom
            n += 1

    if weights.sum() > 0:
        fish /= weights.sum().item()
        weights *= n / weights.sum()
    else:  # all weights are zero: the importance is zero
        weights.fill_(1.)
    net.zero_grad(set_to_none=True)
    net.train(was_training)
    return fish, weights


class IEWC(EwcOn):
    """Continual learning via Improved Elastic Weight Consolidation (IEWC)."""
    NAME = 'iewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        EwcOn.get_parser(parser)
        parser.add_argument('--kappa', type=float, default=1e-2,
                            help='Relative damping: tau = kappa * median of the nonzero output-gradient scales of the task.')
        parser.add_argument('--sample_weighting', type=str, default='uniform', choices=['uniform', 'rdw'],
                            help='Per-sample aggregation weights: `uniform` or residual-diversity weighting (`rdw`).')
        parser.add_argument('--rdw_max_basis', type=int, default=64,
                            help='Maximum number of earlier gradient directions kept by `rdw`.')
        return parser

    def output_scale(self, output_grads):
        """
        Maps the output gradients `r_n` (one per row) to the scales `s_n = r_n^T G_n^{-1} r_n`.
        The default is the identity metric. Override this method to use a different output geometry.
        """
        return squared_norm(output_grads)

    def get_sample_weighting(self):
        """
        Returns the aggregation weights used by `iewc_importance`: `uniform` or `rdw` (from `--sample_weighting`).
        Override this method to return a function `fn(x, y, flat_grad) -> float` with custom non-negative weights.
        """
        return self.args.sample_weighting

    def end_task(self, dataset):
        # cache the (augmented) inputs so that both passes see the same samples
        batches = []
        for i, data in enumerate(dataset.train_loader):
            if self.args.debug_mode and i > self.get_debug_iters():
                break
            batches.append((data[0], data[1]))

        fish, _ = iewc_importance(self.net, self.loss, batches, kappa=self.args.kappa,
                                  sample_weighting=self.get_sample_weighting(),
                                  rdw_max_basis=self.args.rdw_max_basis,
                                  output_scale=self.output_scale)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()
