import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.lider_model import LiderOptimizer, add_lipschitz_args


class ErACELider(LiderOptimizer):
    NAME = 'er_ace_lider'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors)'
                                'Treated with LiDER!')
        add_rehearsal_args(parser)
        add_lipschitz_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.seen_so_far = torch.tensor([]).long().to(self.device)

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.net.set_return_prerelu(True)

            self.init_net(dataset)

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        tot_loss = loss.item()
        loss.backward()

        if self.current_task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)
            tot_loss += loss_re.item()
            loss_re.backward()

        if not self.buffer.is_empty():
            if self.args.alpha_lip_lambda > 0:
                buf_inputs, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features

                loss_lip_minimize = self.args.alpha_lip_lambda * self.minimization_lip_loss(lip_inputs)
                tot_loss += loss_lip_minimize.item()
                loss_lip_minimize.backward()

            if self.args.beta_lip_lambda > 0:
                buf_inputs, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                lip_inputs = [buf_inputs] + buf_output_features

                loss_lip_dyn_budget = self.args.beta_lip_lambda * self.dynamic_budget_lip_loss(lip_inputs)
                tot_loss += loss_lip_dyn_budget.item()
                loss_lip_dyn_budget.backward()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return tot_loss
