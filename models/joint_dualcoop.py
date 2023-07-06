import torch
from tqdm import tqdm
from datasets.utils.validation import ValidationDataset
from models.dualcoop_utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from models.utils import ContinualModel
from datasets import get_dataset

from models.dualcoop_utils import build_model
from torch.nn import functional as F
from utils.metrics import mAP
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)

    parser.add_argument('--gamma_neg', type=float, default=2.0)
    parser.add_argument('--gamma_pos', type=float, default=1.0)
    parser.add_argument('--loss_w', type=float, default=1)

    parser.add_argument('--n_ctx_pos', type=int, default=16)
    parser.add_argument('--n_ctx_neg', type=int, default=16)

    parser.add_argument('--visual_encoder_type', type=str, default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--ctx_init_pos', type=str, default='')
    parser.add_argument('--ctx_init_neg', type=str, default='')

    parser.add_argument('--use_class_specific_context', type=int, default=1, choices=[0, 1])
    parser.add_argument('--finetune_backbone', type=int, default=0, choices=[0, 1])
    parser.add_argument('--finetune_attn', type=int, default=0, choices=[0, 1])

    parser.add_argument('--single_prompt', type=str, default='pos', choices=['pos', 'neg'])

    parser.add_argument('--use_ce', type=int, default=0, choices=[0, 1])

    return parser

class JointDualcoop(ContinualModel):
    NAME = 'joint_dualcoop'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        assert 'vit' not in args.visual_encoder_type.lower(), "ViT not supported yet"
        self.dataset = get_dataset(args)
        self.classnames = self.dataset.get_classnames()

        # TODO: correct classnames by task (now we are cheating)
        backbone = build_model(args, self.classnames.tolist())

        super().__init__(backbone, loss, args, transform)
        self.current_task = 0
        
        self.criterion = AsymmetricLoss(args.gamma_neg, args.gamma_pos)
        self.criterion2 = AsymmetricLoss2(args.gamma_neg, args.gamma_pos)
        self.criterion3 = AsymmetricLoss3(args.gamma_neg, args.gamma_pos)

        self.ce = torch.nn.CrossEntropyLoss()

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
        return self.net(inputs, task_mask=self.inference_task_mask)

    def _observe(self, inputs, labels, not_aug_inputs, epoch=None):
        logits = self.net(inputs)

        # # https://github.com/sunxm2357/DualCoOp/blob/main/train.py
        # loss = self.loss(logits, labels)
        if self.args.use_ce:
            loss = F.cross_entropy(logits.permute(0,2,1).cpu().reshape(-1,2), \
                                   F.one_hot(labels.long(), 2).cpu().float().reshape(-1,2), reduction='none').reshape(logits.shape[0],-1).mean()
            loss *= self.args.loss_w 
        elif logits.dim() == 3:
            loss = self.args.loss_w * self.criterion(logits, labels)
        elif self.args.single_prompt == 'pos':
            loss = self.args.loss_w * self.criterion2(logits, labels)
        elif self.largs.single_prompt == 'neg':
            loss = self.args.loss_w * self.criterion3(logits, labels)
        else:
            raise ValueError

        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.model.parameters(), self.args.clip_grad)
        self.opt.step()

        predictions = torch.softmax(logits,1)[:,1]
        mAP_value = mAP(predictions.detach().cpu().numpy(), labels.cpu().numpy())

        self.autolog_wandb(locals(), {'train_mAP': mAP_value, 'lr': self.opt.param_groups[0]['lr']})

        return loss.item()
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0

    def end_task(self, dataset):
        self.old_data.append(dataset.train_loader.dataset.imgs)
        self.old_labels.append(torch.tensor(dataset.train_loader.dataset.multihot_labels))

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

        self.opt = self.get_optimizer()
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

        
                
            

