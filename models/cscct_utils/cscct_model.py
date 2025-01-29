import torch
from torch.functional import F

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from copy import deepcopy


def sim_matrix(a, b, eps=1e-8):
    """
    Batch cosine similarity taken from https://stackoverflow.com/a/58144658/10425618
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class CscCtModel(ContinualModel):

    @staticmethod
    def add_cscct_args(parser):
        parser.add_argument('--csc_weight', type=float, default=3, help='Weight of CSC loss.')
        parser.add_argument('--ct_weight', type=float, default=1.5, help='Weight of CT loss.')
        parser.add_argument('--ct_temperature', type=float, default=2, help='Temperature of CT loss.')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(CscCtModel, self).__init__(backbone, loss, args, transform, dataset)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.old_net = None

    def get_csc_loss(self, targets, cur_features, ref_features):
        targets_unsqueezed = targets.unsqueeze(1)
        indexes = (targets_unsqueezed == targets_unsqueezed.T).to(torch.int)
        indexes[indexes == 0] = -1
        computed_similarity = sim_matrix(cur_features, ref_features).flatten()
        csc_loss = 1 - computed_similarity
        csc_loss *= indexes.flatten()
        csc_loss = csc_loss.mean()
        return csc_loss * self.args.csc_weight

    def get_ct_loss(self, targets, cur_features, ref_features):
        tasks = targets // self.cpt

        ref_features_curtask = ref_features[tasks == self.current_task]
        ref_features_prevtask = ref_features[tasks < self.current_task]
        cur_features_curtask = cur_features[tasks == self.current_task]
        cur_features_prevtask = cur_features[tasks < self.current_task]
        previous_model_similarities = sim_matrix(ref_features_curtask, ref_features_prevtask)
        current_model_similarities = sim_matrix(cur_features_curtask, cur_features_prevtask)
        ct_loss = torch.nn.KLDivLoss()(
            F.log_softmax(current_model_similarities / self.args.ct_temperature, dim=1),
            F.softmax(previous_model_similarities / self.args.ct_temperature, dim=1)
        ) * (self.args.ct_temperature ** 2)
        return ct_loss * self.args.ct_weight

    def get_cscct_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        # get the current features
        cur_features = self.net.features(inputs)
        with torch.no_grad():
            ref_features = self.old_net.features(inputs)

        csc_loss = self.get_csc_loss(targets, cur_features, ref_features)
        ct_loss = self.get_ct_loss(targets, cur_features, ref_features)

        return csc_loss + ct_loss

    def end_task(self, dataset):
        super().end_task(dataset)
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
