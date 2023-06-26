import torch

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from models.utils import ContinualModel
from datasets import get_dataset

from models.dualcoop_utils import build_model


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)

    return parser


class DualCoop(ContinualModel):
    NAME = 'dualcoop'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.classnames = self.dataset.get_classnames()

        # TODO: correct classnames by task (now we are cheating)
        backbone = build_model(args, self.classnames.tolist())

        super().__init__(backbone, loss, args, transform)
        self.current_task = 0


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]
        logits = self.net(inputs)

        # TODO: train regime (change loss etc...)
        loss = self.loss(logits, labels)

        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.model.parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.savecheck_martin()
        self.current_task += 1

    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task - 1)
        return self.net(x)[:, :offset_2]
