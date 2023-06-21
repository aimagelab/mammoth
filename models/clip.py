import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from datasets import get_dataset
import clip
import torchvision
from utils.conf import get_device

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_aux_dataset_args(parser)
    return parser


class FinalModel(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.clip_model, self.clip_preprocess = clip.load('ViT-L/14')
        self.classes = self.get_classes()
        self.text_inputs = torch.cat([clip.tokenize(f"something that looks like {c}") for c in self.classes]).to(get_device())
    
    def get_classes(self):
        if 'cifar100' in self.args.dataset:
            return [x.replace('_', ' ') for x in torchvision.datasets.CIFAR100(root='./data', train=True, download=True).classes]
        else:
            raise NotImplementedError
    
    def forward(self, x):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            text_features = self.clip_model.encode_text(self.text_inputs)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        return similarity


class Clip(ContinualModel):
    NAME = 'clip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.cpt = self.dataset.N_CLASSES_PER_TASK if hasattr(self.dataset, 'N_CLASSES_PER_TASK') else self.dataset.N_CLASSES // self.dataset.N_TASKS
        backbone = FinalModel(args)
        super().__init__(backbone, loss, args, transform)
        self.current_task = 0

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.savecheck_martin()
        self.current_task += 1


    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        return 0.0
    
    def forward(self, x):
        outputs = self.net(x)
        outputs = outputs[:, :max(self.current_task, 1)*self.cpt]
        return outputs