import torch
from models.utils.continual_model import ContinualModel

from utils.buffer import Buffer

from .spectral_analysis import calc_ADL_knn, calc_euclid_dist, find_eigs, normalize_A


class CasperModel(ContinualModel):

    @staticmethod
    def add_casper_args(parser):
        parser.add_argument('--casper_batch', type=int, default=None,
                            help='Size of minibatch for casper. Equal to batch_size by default, if negative equal to buffer_size.')

        parser.add_argument('--rho', type=float, default=0.01, help='Weight for casper loss.')
        parser.add_argument('--knn_laplace', type=int, default=10, help='K of knn to build the graph for laplacian.')
        parser.add_argument('--p', default=None, type=int, help='Number of classes to be drawn from the buffer. Default is N_CLASSES_PER_TASK.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert 'buffer_size' in args, 'The model requires a buffer'
        if args.casper_batch is None:
            args.casper_batch = args.batch_size
        if args.casper_batch < 0:
            args.casper_batch = args.buffer_size
        super().__init__(backbone, loss, args, transform, dataset)

        self.buffer = Buffer(self.args.buffer_size, device=self.device, sample_selection_strategy='balancoir')

        self.nc = self.args.p if self.args.p is not None else self.cpt

    def get_casper_loss(self):
        if self.args.rho == 0:
            return torch.tensor(0., dtype=torch.float, device=self.device)
        if self.args.casper_batch == self.args.buffer_size:
            buffer_data = self.buffer.get_all_data(transform=self.transform)
        else:
            buffer_data = self.buffer.get_balanced_data(self.args.casper_batch, transform=self.transform, n_classes=self.nc)
        inputs, labels = buffer_data[0], buffer_data[1]
        features = self.net.features(inputs.to(self.device))

        dists = calc_euclid_dist(features)
        A, D, L = calc_ADL_knn(dists, k=self.args.knn_laplace, symmetric=True)

        L = torch.eye(A.shape[0], device=A.device) - normalize_A(A, D)

        n = self.nc
        # evals = torch.linalg.eigvalsh(L)
        evals, _ = find_eigs(L, n_pairs=min(2 * n, len(L)))

        # gaps = evals[1:] - evals[:-1]
        return evals[:n + 1].sum() - evals[n + 1]
