from copy import deepcopy

import torch
import torch.nn.functional as F

from models.casper_utils.casper_model import CasperModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, fill_buffer, icarl_replay


class ICarlCasper(CasperModel):
    """Continual Learning via iCaRL. Treated with CaSpeR!"""
    NAME = 'icarl_casper'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        CasperModel.add_casper_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.eye = torch.eye(self.num_classes).to(self.device)

        self.class_means = None
        self.old_net = None

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

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

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
        loss = self.get_loss(inputs, labels, self.current_task, logits)

        if self.current_task > 0 and self.args.casper_batch > 0 and self.args.rho > 0:
            casper_loss = self.get_casper_loss()
            loss += casper_loss * self.args.rho

        loss.backward()

        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index
            logits: the logits of the old network

        Returns:
            the differentiable loss value
        """

        outputs = self.net(inputs)[:, :self.n_seen_classes]
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

        return loss

    def begin_task(self, dataset):
        icarl_replay(self, dataset)

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
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
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)
