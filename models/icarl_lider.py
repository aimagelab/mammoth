import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple

from backbone import get_backbone
from utils.buffer import Buffer, fill_buffer, icarl_replay
from utils.args import *
from utils.distributed import make_dp
from models.utils.lider_model import LiderOptimizer, add_lipschitz_args
from utils.batch_norm import bn_track_stats


class ICarlLider(LiderOptimizer):
    """Continual Learning via iCaRL. Treated with LiDER!"""
    NAME = 'icarl_lider'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        add_lipschitz_args(parser)

        parser.add_argument('--wd_reg', type=float, default=0.00001,
                            help='L2 regularization applied to the parameters.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.optim_wd != 0:
            logging.warning('iCaRL uses a custom weight decay, the optimizer weight decay will be ignored.')
            args.optim_wd = 0
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size)
        self.eye = torch.eye(self.num_classes).to(self.device)

        self.class_means = None
        self.old_net = None

    def to(self, device):
        self.eye = self.eye.to(device)
        return super().to(device)

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss, output_features = self.get_loss(inputs, labels, self.current_task, logits)

        # Lipschitz losses
        if not self.buffer.is_empty():
            future_mask = labels <= self.n_past_classes
            if future_mask.sum() > 0:
                inputs, output_features = inputs[future_mask], [f[future_mask] for f in output_features]

                lip_inputs = [inputs] + output_features

                if self.args.alpha_lip_lambda > 0:
                    loss_lip_minimize = self.args.alpha_lip_lambda * self.minimization_lip_loss(lip_inputs)
                    loss += loss_lip_minimize

                if self.args.beta_lip_lambda > 0:
                    loss_lip_budget = self.args.beta_lip_lambda * self.dynamic_budget_lip_loss(lip_inputs)
                    loss += loss_lip_budget

        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index
            logits: the logits of the old network

        Returns:
            torch.Tensor: the loss tensor
            List[torch.Tensor]: the output features
        """

        outputs, output_features = self.net(inputs, returnt='full')
        outputs = outputs[:, :self.n_seen_classes]

        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :self.n_seen_classes]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, self.n_past_classes:self.n_seen_classes]
            comb_targets = torch.cat((logits[:, :self.n_past_classes], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        if self.args.wd_reg:
            loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        return loss, output_features

    def begin_task(self, dataset):
        icarl_replay(self, dataset)
        if self.current_task == 0:
            self.net.set_return_prerelu(True)

            self.init_net(dataset)

    def end_task(self, dataset) -> None:
        self.old_net = get_backbone(self.args).to(self.device)
        if self.args.distributed == 'dp':
            self.old_net = make_dp(self.old_net)
        _, unexpected = self.old_net.load_state_dict(deepcopy(self.net.state_dict()), strict=False)
        assert len([k for k in unexpected if 'lip_coeffs' not in k]) == 0, f"Unexpected keys in pretrained model: {unexpected}"
        self.old_net.eval()
        self.old_net.set_return_prerelu(True)

        self.net.train()
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=True)
        self.class_means = None

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        buf_data = self.buffer.get_all_data(transform, device=self.device)
        examples, labels = buf_data[0], buf_data[1]
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                all_features = []
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features')
                    all_features.append(feats)
                all_features = torch.cat(all_features).mean(0)
                class_means.append(all_features.flatten())
        self.class_means = torch.stack(class_means)
