import torch

from utils.args import add_rehearsal_args, ArgumentParser
from models.casper_utils.casper_model import CasperModel


class ErACECasper(CasperModel):
    """ER-ACE with future not fixed (as made by authors). Treated with CaSpeR!"""
    NAME = 'er_ace_casper'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        CasperModel.add_casper_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)
        self.seen_so_far = torch.tensor([], dtype=torch.long, device=self.device)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        self.opt.zero_grad()

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.N_CLASSES - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss_class = self.loss(logits, labels)
        loss = loss_class
        if self.current_task > 0 and self.args.buffer_size > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            loss_erace = self.loss(self.net(buf_inputs), buf_labels)
            loss += loss_erace

            if self.args.casper_batch > 0 and self.args.rho > 0:
                loss_casper = self.get_casper_loss()
                loss += loss_casper * self.args.rho

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        loss.backward()
        self.opt.step()

        return loss.item()
