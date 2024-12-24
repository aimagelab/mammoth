from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
import torch
from models.utils.lider_model import LiderOptimizer, add_lipschitz_args


class DerppLider(LiderOptimizer):
    """Continual learning via Dark Experience Replay++. Treated with LiDER!"""
    NAME = 'derpp_lider'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        add_lipschitz_args(parser)

        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.net.set_return_prerelu(True)

            self.init_net(dataset)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs, buf_output_features = self.net(buf_inputs, returnt='full')
            loss_mse = F.mse_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * loss_mse

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs).float()
            loss_ce = self.loss(buf_outputs, buf_labels)
            loss += self.args.beta * loss_ce

            if self.args.alpha_lip_lambda > 0:
                buf_inputs, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features

                loss_lip_minimize = self.minimization_lip_loss(lip_inputs)
                loss += self.args.alpha_lip_lambda * loss_lip_minimize

            if self.args.beta_lip_lambda > 0:
                buf_inputs, _, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features

                loss_lip_dyn_budget = self.dynamic_budget_lip_loss(lip_inputs)
                loss += self.args.beta_lip_lambda * loss_lip_dyn_budget

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()
