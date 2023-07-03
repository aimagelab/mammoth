import torch
from models.dualcoop_utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3
from models.dualcoop_utils.dualcoop import load_clip_to_cpu

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from models.utils import ContinualModel
from datasets import get_dataset

from torch.nn import functional as F
from utils.metrics import mAP
import types
from torch.cuda.amp import autocast

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)

    parser.add_argument('--visual_encoder_type', type=str, default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'])

    return parser

@autocast()
def custom_forward(self, x):
    x = self.visual(x)
    if isinstance(x, (list, tuple)):
        x = x[0]
    x = x.mean(-1) # avg pool
    x = self.classifier(x)
    return x

class ClipFinetuneBaseline(ContinualModel):
    NAME = 'clip_finetune_baseline'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        assert 'vit' not in args.visual_encoder_type.lower(), "ViT not supported yet"
        backbone = load_clip_to_cpu(args)
        dset = get_dataset(args)
        backbone.requires_grad_(False)
        backbone.classifier = torch.nn.Linear(backbone.text_projection.data.shape[0], sum(dset.N_CLASSES_PER_TASK))
        backbone.forward = types.MethodType(custom_forward, backbone)

        super().__init__(backbone, loss, args, transform)

        self.dataset = dset
        self.current_task = 0

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.cpts = self.dataset.N_CLASSES_PER_TASK
        self.old_cpts = 0

    def begin_task(self, dataset):
        self.task_mask = dataset.train_loader.dataset.task_mask
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
