import torch
from models.dualcoop_utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3
from models.dualcoop_utils.dualcoop import load_clip_to_cpu

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from models.utils import ContinualModel
from datasets import get_dataset

from torch.nn import functional as F
from utils.metrics import mAP
import types
import timm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)

    parser.add_argument('--network', type=str, default='vit_base_patch16_224', help='Network to use')

    return parser

def load_vit_with_ckpt(args, dset):
    backbone = timm.create_model(args.network, pretrained=True, num_classes=sum(dset.N_CLASSES_PER_TASK))
    backbone.requires_grad_(False)
    backbone.head.requires_grad_(True)
    if hasattr(backbone, 'fc_head'):
        backbone.fc_head.requires_grad_(True)

    return backbone

class VitFinetuneBaseline(ContinualModel):
    NAME = 'vit_finetune_baseline'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        dset = get_dataset(args)
        backbone = load_vit_with_ckpt(args, dset)

        super().__init__(backbone, loss, args, transform)

        self.dataset = dset
        self.current_task = 0

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.cpts = self.dataset.N_CLASSES_PER_TASK
        self.old_cpts = 0
        self.inference_task_mask = None

    def begin_task(self, dataset):
        # masking current task 
        self.task_mask = dataset.train_loader.dataset.task_mask
        if self.inference_task_mask is None:
            self.inference_task_mask = self.task_mask.clone()
        else:
            self.inference_task_mask += self.task_mask

        self.eval()

    def forward(self, inputs):
        return self.net(inputs)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        logits = self.net(inputs)
        logits = logits[:, self.task_mask]
        labels = labels[:, self.task_mask]

        loss = self.loss(logits, labels)

        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.model.parameters(), self.args.clip_grad)
        self.opt.step()

        predictions = torch.sigmoid(logits)
        mAP_value = mAP(predictions.detach().cpu().numpy(), labels.cpu().numpy())

        self.autolog_wandb(locals(), {'train_mAP': mAP_value, 'lr': self.opt.param_groups[0]['lr']})

        return loss.item()

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.savecheck_martin()
        self.old_cpts += self.cpts[self.current_task]
        self.current_task += 1
