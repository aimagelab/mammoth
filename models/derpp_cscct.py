import torch
from torch.nn import functional as F

from utils.args import add_rehearsal_args, ArgumentParser
from models.cscct_utils.cscct_model import CscCtModel


class DerppCscCt(CscCtModel):
    """Continual learning via Dark Experience Replay++. Treated with CSCCT!"""
    NAME = 'derpp_cscct'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        CscCtModel.add_cscct_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            loss += derpp_loss

            if self.current_task > 0 and self.args.csc_weight > 0 and self.args.ct_weight > 0:
                # concatenate stream with buf
                full_inputs = torch.cat([inputs, buf_inputs], dim=0)
                full_targets = torch.cat([labels, buf_labels], dim=0)
                cscct_loss = self.get_cscct_loss(full_inputs, full_targets)
                loss += cscct_loss

        loss.backward()
        self.opt.step()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)

        return loss.item()
