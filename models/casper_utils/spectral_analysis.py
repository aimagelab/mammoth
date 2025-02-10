import math

import torch
from xitorch import LinearOperator
from xitorch.linalg import symeig

from .knn import NeuralNearestNeighbors


def calc_ADL_from_dist(dist_matrix: torch.Tensor, sigma=1.):
    # compute affinity matrix, heat_kernel
    A = torch.exp(-dist_matrix / (sigma ** 2))
    # compute degree matrix
    D = torch.diag(A.sum(1))
    # compute laplacian
    L = D - A
    return A, D, L


def calc_euclid_dist(data: torch.Tensor):
    return ((data.unsqueeze(0) - data.unsqueeze(1)) ** 2).sum(-1)


def calc_cos_dist(data):
    return -torch.cosine_similarity(data.unsqueeze(0), data.unsqueeze(1), dim=-1)


def calc_dist_weiss(nu: torch.Tensor, logvar: torch.Tensor):
    var = logvar.exp()
    edist = calc_euclid_dist(nu)
    wdiff = (var.unsqueeze(0) + var.unsqueeze(1) - 2 * (torch.sqrt(var.unsqueeze(0) * var.unsqueeze(1)))).sum(-1)
    return edist + wdiff


def calc_ADL_heat(dist_matrix: torch.Tensor, sigma=1.):
    # compute affinity matrix, heat_kernel
    A = torch.exp(-dist_matrix / (dist_matrix.mean().detach()))
    # compute degree matrix
    d_values = A.sum(1)
    assert not (d_values == 0).any(), f'D contains zeros in diag: \n{d_values}'  # \n{A.tolist()}\n{distances.tolist()}'
    D = torch.diag(d_values)
    # compute laplacian
    L = D - A
    return A, D, L


def calc_ADL_knn(distances: torch.Tensor, k: int, symmetric: bool = True):
    new_A = torch.clone(distances)
    new_A[torch.eye(len(new_A)).bool()] = +math.inf

    knn = NeuralNearestNeighbors(k)

    # knn.log_temp = torch.nn.Parameter(torch.tensor(-10.))
    # final_A = knn(-new_A.unsqueeze(0)).squeeze().sum(-1)
    # final_A += final_A.clone().T
    # final_A[final_A != 0] /= final_A.clone()[final_A != 0]

    final_A = torch.zeros_like(new_A)
    idxes = new_A.topk(k, largest=False)[1]
    final_A[torch.arange(len(idxes)).unsqueeze(1), idxes] = 1
    # backpropagation trick
    w = knn(-new_A.unsqueeze(0)).squeeze().sum(-1)
    if symmetric:
        # final_A += final_A.T
        final_A = ((final_A + final_A.T) > 0).float()
        w = w + w.T

    # Ahk, _, _ = calc_ADL_from_dist(distances, sigma=1)
    A = final_A.detach() + (w - w.detach())
    # A = final_A

    # compute degree matrix
    d_values = A.sum(1)
    assert not (d_values == 0).any(), f'D contains zeros in diag: \n{d_values}'  # \n{A.tolist()}\n{distances.tolist()}'
    D = torch.diag(d_values)
    # compute laplacian
    L = D - A
    return A, D, L


def calc_ADL(data: torch.Tensor, sigma=1.):
    return calc_ADL_from_dist(calc_euclid_dist(data), sigma)


def find_eigs(laplacian: torch.Tensor, n_pairs: int = 0, largest=False):
    # n_pairs = 0
    if n_pairs > 0:
        # eigenvalues, eigenvectors = torch.lobpcg(laplacian, n_pairs, largest=torch.tensor([largest]))
        # eigenvalues, eigenvectors = LOBPCG2.apply(laplacian, n_pairs)
        eigenvalues, eigenvectors = symeig(LinearOperator.m(laplacian, True), n_pairs)
    else:
        # eigenvalues = eigenvalues.to(float)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # eigenvectors = eigenvectors.to(float)
        sorted_indices = torch.argsort(eigenvalues, descending=largest)
        eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def calc_energy_from_values(values: torch.Tensor, norm=False):
    nsamples = len(values)
    max_value = nsamples - 1 if norm else nsamples * (nsamples - 1)
    dir_energy = values.sum()
    energy_p = dir_energy / max_value
    return energy_p.cpu().item()


def normalize_A(A, D):
    inv_d = torch.diag(D[torch.eye(len(D)).bool()].pow(-0.5))
    assert not torch.isinf(inv_d).any(), 'D^-0.5 contains inf'
    # inv_d[torch.isinf(inv_d)] = 0
    # return torch.sqrt(torch.linalg.inv(D)) @ A @ torch.sqrt(torch.linalg.inv(D))
    return inv_d @ A @ inv_d


def dir_energy_normal(data: torch.Tensor, sigma=1.):
    A, D, L = calc_ADL(data, sigma)
    L_norm = torch.eye(A.shape[0]).to(data.device) - normalize_A(A, D)
    eigenvalues, eigenvectors = find_eigs(L_norm)
    energy = calc_energy_from_values(eigenvalues, norm=True)
    return energy, eigenvalues, eigenvectors


def dir_energy(data: torch.Tensor, sigma=1):
    A, D, L = calc_ADL(data, sigma=sigma)
    eigenvalues, eigenvectors = find_eigs(L)
    energy = calc_energy_from_values(eigenvalues)
    return energy


def laplacian_analysis(data: torch.Tensor, sigma=1., knn=0, logvars: torch.Tensor = None,
                       norm_lap=False, norm_eigs=False, n_pairs=0):
    if logvars is None:
        distances = calc_euclid_dist(data)
    else:
        distances = calc_dist_weiss(data, logvars)
    if knn > 0:
        A, D, L = calc_ADL_knn(distances, knn, symmetric=True)
    else:
        A, D, L = calc_ADL_from_dist(distances, sigma)
    if norm_lap:
        L = torch.eye(A.shape[0]).to(data.device) - normalize_A(A, D)
    eigenvalues, eigenvectors = find_eigs(L, n_pairs=n_pairs)
    energy = calc_energy_from_values(eigenvalues, norm=norm_lap)
    if norm_eigs and not norm_lap:
        eigenvalues = eigenvalues / (len(eigenvalues))
    return energy, eigenvalues, eigenvectors, L, (A, D, distances)


class LOBPCG2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, k: int):
        e, v = torch.lobpcg(A, k=k, largest=False)
        res = (A @ v) - (v @ torch.diag(e))
        assert (res.abs() < 1e-3).all(), 'A v != e v => incorrect eigenpairs'
        ctx.save_for_backward(e, v, A)
        return e, v

    @staticmethod
    def backward(ctx, de, dv):
        """
        solve `dA v + A dv = dv diag(e) + v diag(de)` for `dA`
        """
        e, v, A = ctx.saved_tensors

        vt = v.transpose(-2, -1)
        rhs = ((dv @ torch.diag(e)) + (v @ torch.diag(de)) - (A @ dv)).transpose(-2, -1)

        n, k = v.shape
        K = vt[:, :vt.shape[0]]
        # print('K.det=', K.det())  # should be > 0
        iK = K.inverse()

        dAt = torch.zeros((n, n), device=rhs.device)
        dAt[:k] = (iK @ rhs)[:k]
        dA = dAt.transpose(-2, -1)

        # res = T.mm(dA, v) + T.mm(A, dv) - T.mm(dv, T.diag(e)) - T.mm(v, T.diag(de))
        # print('res=', res)
        return dA, None
