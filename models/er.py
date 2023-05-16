import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils import none_or_float
import timm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    #add_aux_dataset_args(parser)
    parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1, help='Should use pretrained weights?')
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.pretrained = args.pretrained == 1
        backbone = timm.create_model(args.network, pretrained=self.pretrained, num_classes=self.n_classes)

        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.buf_transform = self.dataset.TRANSFORM
        self.opt = self.get_optimizer()

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        if not self.buffer.is_empty():
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.batch_size, transform=self.buf_transform)
            
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        output = self.net(inputs)
        output = output[:, :offset_2]
        target = labels[:, :offset_2]
        loss = self.loss(output, target.float())
        
        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task-1)
        return self.net(x)[:, :offset_2]
