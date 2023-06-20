import os
import sys
sys.path.append(os.getcwd())
import torch
import timm
import matplotlib.pyplot as plt
from datasets import get_dataset
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import pickle
import types
from torchvision.utils import make_grid
import re
from datasets import get_dataset
from torchvision.datasets import CIFAR100
from argparse import Namespace
import backbone
from backbone.ResNet18 import resnet18 as myresnet18
#from backbone.Resnet18 import ResNet as myResNet
from onedrivedownloader import download as dn
import seaborn as sns
from torch_cka import CKA
import shlex

def _pret_forward(self, x):
    ret = []
    x = x.to(self.device)
    x = self.bn1(self.conv1(x))
    
    ret.append(x.clone().detach())
    x = F.relu(x)
    if hasattr(self, 'maxpool'):
        x = self.maxpool(x)
    x = self.layer1(x)
    ret.append(self.layer1[-1].prerelu.clone().detach())
    x = self.layer2(x)
    ret.append(self.layer2[-1].prerelu.clone().detach())
    x = self.layer3(x)
    ret.append(self.layer3[-1].prerelu.clone().detach())

    x = self.layer4(x)
    ret.append(self.layer4[-1].prerelu.clone().detach())
    x = F.avg_pool2d(x, x.shape[2])
    x = x.view(x.size(0), -1)
    if isinstance(self, backbone.ResNet18.ResNet):
        x = self.classifier(x)
    elif isinstance(self, backbone.resnet_new.ResNet):
        x = self.fc(x)
    #x = self.fc(x)
    ret.append(x.clone().detach())

    return x, ret

def load_cp(net, cp_path, device, new_classes=None, moco=False, ignore_classifier=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    s = torch.load(cp_path, map_location=device)
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
                i in s['state_dict'].items() if 'encoder_q' in k}

    if not ignore_classifier:
        cl_weights = [s[k] for k in list(s.keys()) if 'classifier' in k]
        if len(cl_weights) > 0:
            cl_size = cl_weights[-1].shape[0]
            net.classifier = torch.nn.Linear(
                net.classifier.in_features, cl_size).to(device)
    else:
        for k in list(s):
            if 'classifier' in k:
                s.pop(k)
                
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
        pass
    except:
        _, unm = net.load_state_dict(s, strict=False)

        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                        ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"

def get_ckpt_remote_url(args):
    if args.pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"
        
    elif args.pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"
            
    elif args.pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

def load_initial_checkpoint(net, device, args, ignore_classifier=False):
    
    url, ckpt_name = get_ckpt_remote_url(args)
    if args.pre_dataset == 'imagenet':
        if not os.path.exists(args.load_cp):
            print("Downloading checkpoint file...")
            dn(url, args.load_cp)
            print(f"Downloaded in: {args.load_cp}")
        load_cp(net, args.load_cp, device, moco=True, ignore_classifier=ignore_classifier)
    else:                  
        if args.load_cp is None or not os.path.isfile(args.load_cp):
            args.load_cp = args.load_cp if args.load_cp is not  None else './checkpoints/'

            print("Downloading checkpoint file...")
            dn(url, args.load_cp)
            print(f"Downloaded in: {args.load_cp}")
        load_cp(net, args.load_cp, device, moco=True, ignore_classifier=ignore_classifier)
        print("Loaded!")


def cast_value(v):
    try:
        new_v = float(v)
        if '.' not in v:
            return int(v)
        else:
            return new_v
    except:
        pass
    try:
        return int(v)
    except:
        pass
    return v

def load_checkpoint_args(checkpoint_args_path, job_id, return_args_file=False):
    #args_file = os.popen(f'find {shlex.quote(checkpoint_args_path)} -name "*args*{shlex.quote(job_id)}*"').read()
    #args_file = args_file.strip()
    args_file = [x for x in os.listdir(checkpoint_args_path) if job_id in x and 'args' in x][0]
    args_file = os.path.join(checkpoint_args_path, args_file)

    try:
        with open(args_file, 'rb') as f:
            args = pickle.load(f)
        exp_args = vars(args)
    except:
        with open(args_file, 'r') as f:
            l = f.readline()
        l = l[10:-2].split(', ')
        keys = [x.split('=')[0] for x in l]
        values = [cast_value(x.split('=')[1].replace("'", '')) for x in l]
        exp_args = {k: v for k, v in zip(keys, values)}
    if return_args_file:
        return exp_args, args_file
    else:
        return exp_args

def get_twf_vit_outputs(net, prenet, x, y, args: Namespace):
    attention_maps = []

    with torch.no_grad():
        res_s = net(x, returnt='full')
        feats_s = res_s[args.distillation_layers]
        res_t = prenet(x, returnt='full')
        feats_t = res_t[args.distillation_layers]
        

    dist_indices = [int(x) for x in args.adapter_layers.split(',')]
    partial_feats_s = [feats_s[i] for i in dist_indices]
    partial_feats_t = [feats_t[i] for i in dist_indices]

    seq_dataset = get_dataset(args)
    task_ids = torch.div(y, seq_dataset.N_CLASSES_PER_TASK, rounding_mode='floor')

    for i, (idx, net_feat, pret_feat) in enumerate(zip(dist_indices, partial_feats_s, partial_feats_t)):
        adapter = getattr(
                net, f"adapter_{idx+1}")

        output_rho, logits = adapter.attn_fn(pret_feat, torch.tensor(y), task_ids)
        attention_maps.append(output_rho)
    
    return res_s, res_t, attention_maps

def get_twf_resnet_outputs(net, prenet, x, y, args: Namespace):
    attention_maps = []
    with torch.no_grad():
        all_logits, all_partial_features = net(x, returnt='full')
        all_pret_logits, all_pret_partial_features = prenet(x)

    
    all_partial_features = all_partial_features[:-1]
    all_pret_partial_features = all_pret_partial_features[:-1]

    seq_dataset = get_dataset(args)
    task_ids = torch.div(y, seq_dataset.N_CLASSES_PER_TASK, rounding_mode='floor')

    for i, (net_feat, pret_feat) in enumerate(zip(all_partial_features, all_pret_partial_features)):
        adapter = getattr(
                net, f"adapter_{i+1}")
        
        output_rho, logits = adapter.attn_fn(pret_feat, task_ids)
        attention_maps.append(output_rho)
    return all_logits, all_pret_logits, attention_maps

def get_attention_map_grid_twf_vit(attention_maps, distillation_layers, layer, batch_idx, num_imgs=20):
    if distillation_layers == 'attention_masks':
        # Abbiamo matrici 197x197 e dobbiamo prendere le righe
        return make_grid(attention_maps[layer][batch_idx][:num_imgs, 1:].view(-1, 14, 14).unsqueeze(1), padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)
    else:
        # Abbiamo matrici 197x768 e dobbiamo prendere le colonne
        return make_grid(attention_maps[layer][batch_idx][1:].permute(1, 0).view(-1, 14, 14).unsqueeze(1)[:num_imgs], padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)

def get_attention_masks_grid_twf_vit(attention_masks, layer, batch_idx, head, num_imgs=20):
    '''Get a grid of attention masks (obtained with Q @ K) for a given layer, batch index and head'''
    # Q @ K sono matrici 197x197. Bisogna prendere le righe
    x = attention_masks[layer][batch_idx][head][:num_imgs, 1:].view(-1, 14, 14).unsqueeze(1)
    x = (x - x.min()) / (x.max() - x.min())
    img = make_grid(x, padding=5, pad_value=1.0).permute(1,2,0).cpu().detach().numpy()
    #img = img * 255
    return img

def get_outputs_grid_twf_vit(outputs, layer, batch_idx, num_imgs=20):
    '''Get a grid of outputs (obtained from the blocks of ViT) for a given layer and batch index'''
    x = outputs[layer][batch_idx][1:, :num_imgs].view(14, 14, -1).permute(2, 0, 1).unsqueeze(1)
    #x = (x - x.min()) / (x.max() - x.min())
    #x = x * 255
    img =  make_grid(x, padding=5, pad_value=1.0).permute(1,2,0).cpu().detach().numpy()
    #import pdb; pdb.set_trace()
    return img

def get_attention_map_grid_twf_resnet(attention_maps, layer, batch_idx):
    '''Get a grid of attention maps (obtained from the adapters of twf) for a given layer and batch index'''
    return make_grid(attention_maps[layer][batch_idx].cpu().detach().unsqueeze(1)).permute(1,2,0).numpy()

def calc_cka(m1,m2, dataset, layers1, layers2, device, token_index='all'):
    cka=CKA(m1, m2, model1_name="M1", model2_name="M2",model1_layers=layers1,model2_layers=layers2, device=device, token_index=token_index)
    cka.compare(dataset) 
    return cka.export()['CKA']

cka_dataset = None

@torch.no_grad()
def compute_cka_vits(args, device, model1, model2, num_samples=1000, flush_dataset=False, token_index='all'):
    global cka_dataset
    seq_dataset = get_dataset(args)

    normalize_t = seq_dataset.get_normalization_transform()
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_t
    ])

    if flush_dataset or cka_dataset is None:
        if 'cifar100' in seq_dataset.NAME:
            dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=t)
        else:
            print(f'Error! dataset {seq_dataset.NAME} not supported')
            return
        cka_dataset = dataset
    dataset = cka_dataset
    
    dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), num_samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    layers1 = [f'blocks.{i}' for i in range(12)] if token_index != 'cls' else [f'blocks.{i}' for i in range(1, 12)]
    layers2 = [f'blocks.{i}' for i in range(12)] if token_index != 'cls' else [f'blocks.{i}' for i in range(1, 12)]
    cka = calc_cka(model1, model2, dataloader, layers1, layers2, device, token_index)
    return cka

@torch.no_grad()
def compute_cka_general(args, device, model1, model2, num_samples=1000, flush_dataset=False, dataloader=None):
    if dataloader is None:
        global cka_dataset
        seq_dataset = get_dataset(args)

        normalize_t = seq_dataset.get_normalization_transform()
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_t
        ])

        if flush_dataset or cka_dataset is None:
            if 'cifar100' in seq_dataset.NAME:
                dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=t)
            else:
                print(f'Error! dataset {seq_dataset.NAME} not supported')
                return
            cka_dataset = dataset
        dataset = cka_dataset
        
        dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), num_samples, replace=False))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)


    network = args.network
    #import pdb; pdb.set_trace()

    if network.startswith('resnet'):
        layers1 = ['conv1'] + [f'layer{i}' for i in range(1, 5)]
        layers2 = layers1
    elif network.startswith('vit'):
        layers1 = [f'blocks.{i}' for i in range(12)]
        layers2 = layers1
    elif network.startswith('swin'):
        layers1 = [k for k, v in model1.named_modules() if re.search(r'^layers.\d+.blocks.\d+$', k)]
        layers2 = layers1

    cka = calc_cka(model1, model2, dataloader, layers1, layers2, device)
    return cka

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreterr

@torch.no_grad()
def mini_eval(model, device, dataset):
    tg = model.training
    model.eval()
    correct = 0
    total = 0
    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    for data in tqdm(dataset):
        data, target = data[0], data[1]
        data, target = data.to(device), target.to(device)
        output = model(data)[:, :10]
        correct += (torch.argmax(output, dim=1) == target).sum().item()
        total += len(data)
    model.train(tg)
    return correct / total

def load_exp(job_id, task):
    exp_args = load_checkpoint_args(f'/home/aba/martin_cineca/checkpoints', job_id)
    #exp_args = load_checkpoint_args(f'/home/aba/martin_cineca/checkpoints/{method}_{network}_args_{job_id}.txt')
    args = Namespace(**exp_args)
    continual_dataset = get_dataset(args)
    method = args.model
    network = args.network
    num_classes = continual_dataset.N_CLASSES_PER_TASK * continual_dataset.N_TASKS
    model = timm.create_model(args.network, num_classes=num_classes)
    print(model.load_state_dict(torch.load(f'/home/aba/martin_cineca/checkpoints/{method}_{network}_{task}_{job_id}.pt')))
    model.eval()
    return model, args