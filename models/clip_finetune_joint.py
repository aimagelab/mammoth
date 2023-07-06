import types
import torch
from tqdm import tqdm
from datasets.utils.validation import ValidationDataset
from models.dualcoop_utils.dualcoop import load_clip_to_cpu

from utils.args import add_management_args, add_experiment_args, ArgumentParser
from models.utils import ContinualModel
from datasets import get_dataset

from utils.metrics import mAP
from torchvision import transforms
from torch.cuda.amp import autocast

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)

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

class ClipFinetuneJoint(ContinualModel):
    NAME = 'clip_finetune_joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        assert 'vit' not in args.visual_encoder_type.lower(), "ViT not supported yet"
        self.dataset = get_dataset(args)
        backbone = load_clip_to_cpu(args)
        dset = get_dataset(args)
        backbone.requires_grad_(False)
        backbone.classifier = torch.nn.Linear(backbone.text_projection.data.shape[0], sum(dset.N_CLASSES_PER_TASK))
        backbone.forward = types.MethodType(custom_forward, backbone)

        super().__init__(backbone, loss, args, transform)
        self.current_task = 0
        
        self.dataset = dset
        self.current_task = 0

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.cpts = self.dataset.N_CLASSES_PER_TASK
        self.old_cpts = 0

        self.inference_task_mask = None
        self.old_data = []
        self.old_labels = []

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

    def _observe(self, inputs, labels, not_aug_inputs, epoch=None):
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

    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0

    def end_task(self, dataset):
        self.old_data.append(dataset.train_loader.dataset.imgs)
        self.old_labels.append(dataset.train_loader.dataset.multihot_labels)

        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.old_cpts += self.cpts[self.current_task]
        self.current_task += 1

        if self.current_task < self.dataset.N_TASKS:
            return

        all_data, all_labels = None, None
        for i in range(len(self.old_data)):
            if all_data is None:
                all_data = self.old_data[i]
                all_labels = self.old_labels[i]
            else:
                all_data = torch.cat([all_data, self.old_data[i]])
                all_labels = torch.cat([all_labels, self.old_labels[i]])
                
        train_dataset = ValidationDataset(all_data.permute(0,2,3,1), all_labels, transform=transforms.Compose(self.dataset.TRANSFORM.transforms[1:]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, 
                                                   num_workers=dataset.train_loader.num_workers, pin_memory=True)

        sched = self.dataset.get_scheduler(self, self.args)
        self.eval()
        for epoch in range(self.args.n_epochs):
            with tqdm(train_loader, desc=f'Epoch {epoch}/{self.args.n_epochs}') as pbar:
                for i, batch in enumerate(pbar):
                    if self.args.debug_mode and i > 10:
                        break
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    loss = self._observe(inputs, labels, None, epoch=epoch)
                    pbar.set_postfix({'loss': loss})
            sched.step()