import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest
import torch
import torch.nn.functional as F

from backbone.MNISTMLP import BaseMNISTMLP
from models.iewc import iewc_importance, squared_norm


def _setup(seed=0, n_batches=3, batch_size=4):
    torch.manual_seed(seed)
    net = BaseMNISTMLP(8, 5, hidden_size=6)
    batches = [(torch.randn(batch_size, 8), torch.randint(0, 5, (batch_size,))) for _ in range(n_batches)]
    return net, batches


def _per_sample(net, loss_fn, batches):
    """Reference per-sample parameter gradients and output-gradient scales."""
    grads, scales = [], []
    for inputs, labels in batches:
        for x, y in zip(inputs, labels):
            out = net(x.unsqueeze(0))
            loss = loss_fn(out, y.unsqueeze(0))
            r, = torch.autograd.grad(loss, out, retain_graph=True)
            g = torch.autograd.grad(loss, list(net.parameters()))
            grads.append(torch.cat([gg.flatten() for gg in g]))
            scales.append(r.pow(2).sum())
    return torch.stack(grads), torch.stack(scales)


def test_reduces_to_empirical_fisher():
    # with s_n + tau replaced by 1, IEWC is the empirical Fisher mean_n g_n^2
    net, batches = _setup()
    fish, _ = iewc_importance(net, F.cross_entropy, batches, kappa=0.,
                              output_scale=lambda r: torch.ones(len(r)))
    grads, _ = _per_sample(net, F.cross_entropy, batches)
    torch.testing.assert_close(fish, grads.pow(2).mean(0))


def test_reduces_to_ewc_on_fisher():
    # with s_n + tau replaced by 1 / p(y_n|x_n), IEWC is the Fisher of `ewc_on`
    # (for cross-entropy, r_n = softmax - onehot, so p(y_n|x_n) = 1 + min(r_n))
    net, batches = _setup()
    fish, _ = iewc_importance(net, F.cross_entropy, batches, kappa=0.,
                              output_scale=lambda r: 1 / (1 + r.min(1).values))
    expected = torch.zeros_like(net.get_params())
    for inputs, labels in batches:
        for ex, lab in zip(inputs, labels):
            net.zero_grad()
            loss = - F.nll_loss(F.log_softmax(net(ex.unsqueeze(0)), dim=1), lab.unsqueeze(0), reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            expected += exp_cond_prob * net.get_grads() ** 2
    expected /= sum(len(b[0]) for b in batches)
    torch.testing.assert_close(fish, expected)


@pytest.mark.parametrize('kappa', [0., 1e-2, 1.])
def test_relative_damping(kappa):
    net, batches = _setup(batch_size=5)
    fish, weights = iewc_importance(net, F.cross_entropy, batches, kappa=kappa)
    grads, scales = _per_sample(net, F.cross_entropy, batches)
    tau = kappa * scales.median()  # odd number of samples: the median is unique
    torch.testing.assert_close(fish, (grads.pow(2) / (scales + tau)[:, None]).mean(0))
    assert torch.all(weights == 1)


@pytest.mark.parametrize('sample_weighting', ['uniform', 'rdw'])
def test_loss_scale_invariance(sample_weighting):
    net, batches = _setup()
    fish, _ = iewc_importance(net, F.cross_entropy, batches, sample_weighting=sample_weighting)
    fish_scaled, _ = iewc_importance(net, lambda o, y: 1000. * F.cross_entropy(o, y), batches,
                                     sample_weighting=sample_weighting)
    torch.testing.assert_close(fish_scaled, fish, rtol=1e-4, atol=0.)


def test_all_zero():
    net, batches = _setup()
    fish, _ = iewc_importance(net, lambda o, y: 0. * o.sum(), batches)
    assert torch.all(fish == 0)


def test_rdw_weights():
    net, batches = _setup()
    _, weights = iewc_importance(net, F.cross_entropy, batches, sample_weighting='rdw', rdw_max_basis=4)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))
    assert torch.all(weights >= 0)

    # repeated samples carry no new direction
    x, y = batches[0][0][:1], batches[0][1][:1]
    fish, weights = iewc_importance(net, F.cross_entropy, [(x.repeat(4, 1), y.repeat(4))], sample_weighting='rdw')
    torch.testing.assert_close(weights, torch.tensor([4., 0., 0., 0.]))
    fish_single, _ = iewc_importance(net, F.cross_entropy, [(x, y)])
    torch.testing.assert_close(fish, fish_single)


def test_custom_sample_weighting():
    net, batches = _setup()
    fish, weights = iewc_importance(net, F.cross_entropy, batches)
    fish_const, weights_const = iewc_importance(net, F.cross_entropy, batches, sample_weighting=lambda x, y, g: 3.)
    torch.testing.assert_close(fish_const, fish)
    torch.testing.assert_close(weights_const, weights)

    # keep only the samples with label < 3
    fish, weights = iewc_importance(net, F.cross_entropy, batches, sample_weighting=lambda x, y, g: float(y < 3))
    grads, scales = _per_sample(net, F.cross_entropy, batches)
    h = torch.cat([b[1] for b in batches]).lt(3).float()
    tau = 1e-2 * scales.quantile(0.5)
    expected = (h[:, None] * grads.pow(2) / (scales + tau)[:, None]).sum(0) / h.sum()
    torch.testing.assert_close(fish, expected)
    torch.testing.assert_close(weights, h / h.mean())


def test_squared_norm():
    r = torch.randn(3, 4, 2)
    torch.testing.assert_close(squared_norm(r), r.pow(2).sum((1, 2)))


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('sample_weighting', ['uniform', 'rdw'])
def test_iewc(dataset, sample_weighting):
    sys.argv = ['mammoth',
                '--model',
                'iewc',
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
                '--e_lambda',
                '0.5',
                '--gamma',
                '1',
                '--sample_weighting',
                sample_weighting,
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()
