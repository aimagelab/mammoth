import torch
import torch.nn.functional as F
from utils import none_or_float
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
import timm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors)')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # add_aux_dataset_args(parser)
    parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1, help='Should use pretrained weights?')

    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def get_backbone(self, args):
        pretrained = args.pretrained == 1
        return timm.create_model(args.network, pretrained=pretrained, num_classes=self.n_classes)

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.pretrained = args.pretrained == 1
        backbone = self.get_backbone(args)
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.buf_transform = self.dataset.TRANSFORM

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.opt = self.get_optimizer()
        self.old_epoch = 0

    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task-1)
        return self.net(x)[:, :offset_2]
    
    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.current_task += 1
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        output = self.net(inputs)
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        output = output[:, :offset_2]
        idx = labels.sum(0).nonzero().squeeze(1)
        filtered_output = output[:, idx]
        filtered_target = labels[:, idx]
        loss = self.loss(filtered_output, filtered_target.float())
        loss_re = torch.tensor(0.).to(self.device)

        if self.current_task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.batch_size, transform=self.buf_transform)
            
            loss_re = self.loss(self.net(buf_inputs), buf_labels.float())
            loss += loss_re
        
        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()