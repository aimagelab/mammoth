import torch

from utils.args import add_rehearsal_args, ArgumentParser
from models.cscct_utils.cscct_model import CscCtModel


class ErACECscCt(CscCtModel):
    """ER-ACE with future not fixed (as made by authors). Treated with CSCCT!"""
    NAME = 'er_ace_cscct'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        CscCtModel.add_cscct_args(parser)
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

            if self.args.csc_weight > 0 and self.args.ct_weight > 0:
                # concatenate stream with buf
                full_inputs = torch.cat([inputs, buf_inputs], dim=0)
                full_targets = torch.cat([labels, buf_labels], dim=0)
                loss_cscct = self.get_cscct_loss(full_inputs, full_targets)
                loss += loss_cscct

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        loss.backward()
        self.opt.step()

        return loss.item()
