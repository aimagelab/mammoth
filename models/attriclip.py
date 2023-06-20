import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from datasets import get_dataset
from models.attriclip_utils.model import CoOp
from models.attriclip_utils.utils import *
import torch.nn.functional as F
import wandb

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_aux_dataset_args(parser)
    parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
    parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
    return parser


class AttriClip(ContinualModel):
    NAME = 'attriclip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.num_tasks = self.dataset.N_TASKS
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.cpt = self.dataset.N_CLASSES_PER_TASK
        backbone = CoOp(False, False, args)
        super().__init__(backbone, loss, args, transform)
        self.current_task = 0
    
    def begin_task(self, dataset):
        if 'cifar100' in dataset.NAME:
            class_names = cifar100_classes[self.current_task*self.cpt:(self.current_task+1)*self.cpt]
        else:
            raise NotImplementedError(f'Dataset "{dataset.NAME}" not supported.')

        self.per_epoch_steps = len(dataset.train_loader)

        self.net.init_model(class_names=class_names, text_key=self.net.text_key, text_prompt=self.net.text_prompt)
        self.opt, self.scheduler = self.net.get_optimizer(self.per_epoch_steps)
        self.net.model.eval()
        self.old_epoch = 0
        self.idx = 0

        if self.current_task == 0:
            self.original_prompts = self.net.model.prompt_learner.text_prompt.cpu().detach()
            self.distance_table = wandb.Table(columns=['task', 'prompt', 'distance'])

    def log_distances(self):
        prompts = self.net.model.prompt_learner.text_prompt.cpu().detach().view(self.args.num_prompt, -1)
        original_dists = torch.sum((prompts - self.original_prompts.view(self.args.num_prompt, -1))**2, dim=-1)
        for i, dist in enumerate(original_dists):
            self.distance_table.add_data(self.current_task, i, dist.item())

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()

        self.log_distances()
        if self.current_task == self.num_tasks - 1:
            wandb.log({'distance_table': self.distance_table})
            wandb.log({'distance_table': wandb.plot.line(self.distance_table, 'distance', 'task', title='Distance Table')}

        self.current_task += 1


    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.scheduler and self.old_epoch != epoch:
            self.old_epoch = epoch
            self.idx = 0

        labels = labels.long()
        labels = labels - self.cpt * self.current_task
        lab_idx = labels.cpu().numpy().tolist()

        cur_iter_idx = epoch*self.per_epoch_steps+self.idx
        self.cur_iter_idx = cur_iter_idx
        self.scheduler.step(cur_iter_idx)

        output, ima_feat, key_choose, loss_m = self.net.model(inputs)
        
        loss_main = F.cross_entropy(output, labels.cuda())
        loss_k = cosine_loss(ima_feat,key_choose)
        loss = loss_main + 0.5*loss_k + 0.1*loss_m

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.idx += 1

        return loss.item()
    
    def forward(self, x):
        test_class = cifar100_classes[:max(self.current_task, 1)*self.cpt]
        logits = self.net.model(x, test_class, test=True)
        #scores = logits.float().softmax(dim=-1)
        return logits[:, :max(self.current_task, 1)*self.cpt]