

from copy import deepcopy
import types
import torch
from torch.optim import SGD, Adam
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

from torchvision import transforms
import torch.nn.functional as F
import random
import math

import timm
from models.twf_utils.vision_transformer import vit_base_patch16_224_twf
from models.twf_utils.vit_afd import TokenizedDistillation
import numpy as np
import os
import pickle
import sys
from timm.optim import create_optimizer
import re
import wandb
import torch.nn.functional as F
from utils.hooks_handlers import HooksHandlerViT
from PIL import Image

def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Double-branch distillation + inter-branch skip attention')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    #add_aux_dataset_args(parser)

    # Griddable parameters
    parser.add_argument('--der_alpha', type=float, required=True,
                        help='Distillation alpha hyperparameter for student stream.')
    parser.add_argument('--der_beta', type=float, required=True,
                        help='Distillation beta hyperparameter.')
    parser.add_argument('--lambda_fp', type=float, required=True,
                        help='weight of feature propagation loss replay') 
    parser.add_argument('--lambda_diverse_loss', type=float, required=False, default=0,
                        help='Diverse loss hyperparameter.')
    parser.add_argument('--lambda_fp_replay', type=float, required=False, default=0,
                        help='weight of feature propagation loss replay')
    parser.add_argument('--use_prompt', type=int, choices=[0, 1], default=0, help='Use prompt token')
    parser.add_argument('--use_conditioning', type=int, choices=[0, 1], default=1, help='Use conditioning by current task')
    parser.add_argument('--adapter_layers', type=str, required=True, help='Indices of layers to add adapters. Example: 0,4,8,12')
    parser.add_argument('--distillation_layers', default='block_outputs', choices=['MHSA_outputs', 'block_outputs', 'attention_masks'], type=str, help='output layer to use for distillation')
    parser.add_argument('--adapter_lr', type=float, default=0.1, help='Learning rate of adapters')
    parser.add_argument('--adapter_type', default='mixer', choices=['mixer', 'channel', 'double_mixer', 'double_convit', 'mock', 'mimicking', 'twf_original', 'transformer',
                                                                 'clip_cross_attention', 'clip_cross_attention_v2', 'clip_cross_attention_v3',
                                                                 'transformer_pretrained', 'clip_cross_attention_v4', 'clip_cross_attention_v5', 'clip_cross_attention_v6',
                                                                 'clip_cross_attention_v7', 'transformer_pretrained_clip', 'transformer_pretrained_proj',
                                                                 'tat', 'tat_v2', 'tat_norm', 'transformer_pretrained_layer_norm', 'attention_probe_cls',
                                                                 'attention_probe_cls_norm', 'attention_probe_cls_no_gumbel', 'attention_probe_cls_norm_no_gumbel'], type=str, help='Type of adapter')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Student checkpoint path to load')
    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    parser.add_argument('--adapter_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer for adapters')
    parser.add_argument('--lambda_ignition_loss', type=float, required=False, default=0.0, help='Ignition loss hyperparameter.')
    parser.add_argument('--ignition_loss_temp', type=float, required=False, default=0.2, help='Temperature for ignition loss.')
    parser.add_argument('--exclude_class_token', type=int, choices=[0, 1], default=0, help='Exclude class token from distillation')
    parser.add_argument('--diverse_loss_mode', default='original', choices=['original', 'fix_normalized', 'fix_diagonal', 'fix_normalized_diagonal'], type=str, help='Type of diverse loss')
    return parser

def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))
    

class TwFVIT(ContinualModel):
    NAME = 'twf_vit'
    COMPATIBILITY = ['class-il', 'task-il']


    def __init__(self, backbone, loss, args, transform):
        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES if hasattr(self.dataset, 'N_CLASSES') else self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self.num_tasks = self.dataset.N_TASKS
        self.cpt = self.dataset.N_CLASSES // self.dataset.N_TASKS
        backbone = timm.create_model(f'vit_base_patch16_224_twf', pretrained=True, num_classes=self.n_classes, global_pool=args.global_pool)
        super().__init__(
            backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.buf_transformations = self.dataset.TRANSFORM
        self.not_aug_transform = self.dataset.TEST_TRANSFORM
        self.seen_y_so_far = torch.zeros(self.n_classes).bool().to(self.device)
        self.current_task = 0
    
    def get_twf_vit_outputs(self, x, y, task_labels):
        attention_maps = []
        logits_maps = []

        with torch.no_grad():
            res_s = self.net(x, returnt='full')
            feats_s = res_s[self.args.distillation_layers]
            res_t = self.prenet(x, returnt='full')
            feats_t = res_t[self.args.distillation_layers]
            

        dist_indices = [int(x) for x in self.args.adapter_layers.split(',')]
        if self.args.adapter_type == 'twf_original':
            # we must exclude the class token
            partial_feats_s = [feats_s[i][:, 1:, :] for i in dist_indices]
            partial_feats_t = [feats_t[i][:, 1:, :] for i in dist_indices]
        else:
            partial_feats_s = [feats_s[i] for i in dist_indices]
            partial_feats_t = [feats_t[i] for i in dist_indices]

        for i, (idx, net_feat, pret_feat) in enumerate(zip(dist_indices, partial_feats_s, partial_feats_t)):
            adapter = getattr(
                    self.net, f"adapter_{idx+1}")

            output_rho, logits = adapter.attn_fn(pret_feat, y, task_labels)
            attention_maps.append(output_rho)
            logits_maps.append(logits)

        return res_s, res_t, attention_maps, logits_maps

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()

        self.net.eval()
        with torch.no_grad():
            # loop over buffer
            for buf_idxs in batch_iterate(len(self.buffer), self.args.batch_size):

                buf_idxs = buf_idxs.to(self.device)
                buf_labels = self.buffer.labels[buf_idxs].to(self.device)
                buf_task_labels = self.buffer.task_labels[buf_idxs].to(self.device)

                buf_mask = buf_task_labels == self.current_task

                if not buf_mask.any():
                    continue

                buf_inputs = self.buffer.examples[buf_idxs.cpu().detach().numpy()][buf_mask.cpu().detach().numpy()]
                buf_labels = buf_labels[buf_mask]
                buf_task_labels = buf_task_labels[buf_mask]

                if type(buf_inputs[0]) == Image.Image:
                    buf_inputs = torch.stack([self.not_aug_transform(
                        ee) for ee in buf_inputs]).to(self.device)
                else:
                    buf_inputs = torch.stack([self.not_aug_transform(
                        ee.cpu()) for ee in buf_inputs]).to(self.device)

                res_s = self.net(
                    buf_inputs, returnt='full')
                buf_partial_features = res_s[self.args.distillation_layers]
                prenet_input = buf_inputs
                res_t = self.prenet(prenet_input, returnt='full')
                pret_buf_partial_features = res_t[self.args.distillation_layers]


                # buf_partial_features = buf_partial_features[:-1]
                # pret_buf_partial_features = pret_buf_partial_features[:-1]

                buf_partial_features = [buf_partial_features[i] for i in self.dist_indices]
                pret_buf_partial_features = [pret_buf_partial_features[i] for i in self.dist_indices]

                _, attention_masks, _ = self.partial_distill_loss(buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels, task_labels=buf_task_labels)

                for i_of_idx, idx in enumerate(buf_idxs[buf_mask]):
                    self.buffer.attention_maps[idx] = [
                        at[i_of_idx] for at in attention_masks]

        self.train()

        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.prenet = timm.create_model(f'vit_base_patch16_224_twf', pretrained=True, num_classes=self.n_classes)
            self.prenet = self.prenet.to(self.device)
            self.prenet.eval()

            self.hh_s = HooksHandlerViT(self.net)
            self.hh_t = HooksHandlerViT(self.prenet)

            # Retrieve features
            # Ci serve sapere la shape delle features per costruire gli adapter
            with torch.no_grad():
                x = torch.randn((1, 3, 224, 224)).to(self.device)
                res = self.net(x, returnt='full')
                feats_t = res[self.args.distillation_layers]
                prenet_input = x
                res = self.prenet(prenet_input, returnt='full')
                pret_feats_t = res[self.args.distillation_layers]

            self.dist_indices = [int(x) for x in self.args.adapter_layers.split(',')]
            feats_t = [feats_t[i] for i in self.dist_indices]
            pret_feats_t = [pret_feats_t[i] for i in self.dist_indices]
            
            for (i, x, pret_x) in zip(self.dist_indices, feats_t, pret_feats_t):
                # clear_grad=self.args.detach_skip_grad == 1
                adapt_shape = x.shape[1:]
                pret_shape = pret_x.shape[1:]
                if len(adapt_shape) == 1:
                    adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
                    pret_shape = (pret_shape[0], 1, 1)

                setattr(self.net, f"adapter_{i+1}", TokenizedDistillation(
                    adapt_shape, self.num_tasks, self.n_classes, adatype=self.args.adapter_type,
                    teacher_forcing_or=False,
                    lambda_forcing_loss=self.args.lambda_fp_replay,
                    lambda_diverse_loss=self.args.lambda_diverse_loss,
                    use_prompt=self.args.use_prompt == 1,
                    use_conditioning=self.args.use_conditioning == 1,
                    lambda_ignition_loss=self.args.lambda_ignition_loss,
                    ignition_loss_temp=self.args.ignition_loss_temp,
                    exclude_class_token=self.args.exclude_class_token == 1,
                    diverse_loss_mode=self.args.diverse_loss_mode,
                ).to(self.device))

                if self.args.adapter_type in ['transformer_pretrained', 'transformer_pretrained_proj', 'transformer_pretrained_layer_norm']:
                    adapter = getattr(self.net, f"adapter_{i+1}")
                    adapter.attn_fn.self_attn.load_state_dict(self.prenet.blocks[i+1].state_dict())
            
            # if self.args.load_checkpoint is not None:
            #     self.net.load_state_dict(torch.load(self.args.load_checkpoint))

            myparams = dict(self.net.named_parameters())
            net_params = [myparams[s] for s in myparams.keys() if 'adapter' not in s]
            adapter_params = [myparams[s] for s in myparams.keys() if 'adapter' in s]

            if self.args.optimizer == 'sgd':
                self.opt = torch.optim.SGD(net_params, lr=self.args.lr,
                            weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            elif self.args.optimizer == 'adam':
                self.opt = torch.optim.Adam(net_params, lr=self.args.lr, weight_decay=self.args.optim_wd)
            
            try:
                if self.args.adapter_optimizer == 'sgd':
                    self.opt_adapters = torch.optim.SGD(adapter_params, lr=self.args.adapter_lr,
                                    weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                elif self.args.adapter_optimizer == 'adam':
                    self.opt_adapters = torch.optim.Adam(adapter_params, lr=self.args.adapter_lr, weight_decay=self.args.optim_wd)
            except:
                self.opt_adapters = None

        self.net.train()
        for p in self.prenet.parameters():
            p.requires_grad = False
        pass


    def partial_distill_loss(self, net_partial_features: list, pret_partial_features: list,
                             targets, teacher_forcing: list = None, extern_attention_maps: list = None, task_labels=None):

        assert len(net_partial_features) == len(
            pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

        if teacher_forcing is None or extern_attention_maps is None:
            assert teacher_forcing is None
            assert extern_attention_maps is None

        loss = 0
        losses = {}
        attention_maps = []

        for i, (idx, net_feat, pret_feat) in enumerate(zip(self.dist_indices, net_partial_features, pret_partial_features)):
            assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

            adapter = getattr(
                self.net, f"adapter_{idx+1}")

            pret_feat = pret_feat.detach()

            if teacher_forcing is None:
                curr_teacher_forcing = torch.zeros(
                    len(net_feat,)).bool().to(self.device)
                curr_ext_attention_map = torch.ones(
                    (len(net_feat), adapter.embed_dim)).to(self.device)
            else:
                curr_teacher_forcing = teacher_forcing
                curr_ext_attention_map = torch.stack(
                    [b[i] for b in extern_attention_maps], dim=0).float()

            adapt_loss, adapt_attention, inlosses = adapter(net_feat, pret_feat, targets,
                                                  teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map,
                                                  task_labels=task_labels)
            losses = {**losses, **{f'adapter_{idx+1}_{k}': v for k, v in inlosses.items()}}

            loss += adapt_loss
            attention_maps.append(adapt_attention.detach().cpu().clone().data)

        # TODO: Vedere se questo i deve essere idx
        return loss / (i + 1), attention_maps, losses

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        B = len(inputs)

        stream_task_labels = torch.ones(B)*self.current_task
        stream_task_labels = stream_task_labels.long().to(self.device)
        with torch.no_grad():
            not_aug_inputs_tmp = torch.stack([self.not_aug_transform(i) for i in not_aug_inputs]).to(self.device)
            _, _, stream_not_aug_attention_maps, _ = self.get_twf_vit_outputs(not_aug_inputs_tmp,
                                                                      labels, stream_task_labels)
            stream_not_aug_attention_maps = [i.detach().cpu().clone().data for i in stream_not_aug_attention_maps]

        loss_attention_maps_replay = torch.tensor(0.0).to(self.device)
        if len(self.buffer) > 0:
            # sample from buffer
            buf_choices, buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_logits_mask = self.buffer.get_data(
                self.args.batch_size, transform=None, return_index=True)
            buf_attention_maps = [self.buffer.attention_maps[c]
                                  for c in buf_choices]

            buf_attention_maps = [[i.to(self.device) for i in v] for v in buf_attention_maps]
            
            buf_not_aug_inputs_tmp = torch.stack([self.not_aug_transform(i) for i in buf_inputs]).to(self.device)
            _, _, _, buf_not_aug_adapter_logits = self.get_twf_vit_outputs(buf_not_aug_inputs_tmp, buf_labels, 
                                                                   buf_task_labels)
            losses_attn_maps = []
            buf_attention_maps_t = []
            for i in range(len(buf_attention_maps[0])):
                buf_attention_maps_t.append([b[i] for b in buf_attention_maps])
            for gt, pred in zip(buf_attention_maps_t, buf_not_aug_adapter_logits):
                for g, p in zip(gt, pred):
                    losses_attn_maps.append(F.binary_cross_entropy_with_logits(p[:, 0, :], g.float()))
            loss_attention_maps_replay = torch.mean(torch.stack(losses_attn_maps))
            
            aug_inputs = torch.stack([self.buf_transformations(buf_input) for buf_input in buf_inputs]).to(self.device)

            # if self.args.use_patch_level_aug:
            #     aug_inputs = patch_level_aug(transforms.Resize(224)(buf_inputs))

            inputs = torch.cat([inputs, aug_inputs])
            all_labels = torch.cat([labels, buf_labels])
            all_task_labels = torch.cat([stream_task_labels, buf_task_labels])

        prenet_input =  inputs
        with torch.no_grad():
            res_t = self.prenet(prenet_input, returnt='full')
            all_pret_logits, all_pret_partial_features = res_t['output'], res_t[self.args.distillation_layers]

        res_s = self.net(inputs, returnt='full')
        all_logits, all_partial_features = res_s['output'], res_s[self.args.distillation_layers]

        all_partial_features = [all_partial_features[i] for i in self.dist_indices]
        all_pret_partial_features = [all_pret_partial_features[i] for i in self.dist_indices]

        stream_logits, buf_outputs = all_logits[:B], all_logits[B:]
        stream_partial_features = [p[:B] for p in all_partial_features]
        stream_pret_partial_features = [p[:B]
                                        for p in all_pret_partial_features]

        output_mask = self.seen_y_so_far.unsqueeze(0).expand_as(stream_logits).detach().clone()
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        output = stream_logits[:, :offset_2]
        idx = labels.sum(0).nonzero().squeeze(1)
        filtered_output = output[:, idx]
        filtered_target = labels[:, idx]
        loss = self.loss(filtered_output, filtered_target.float())
        
        loss_clf = loss.detach().clone()

        self.seen_y_so_far[:offset_2] |= labels[:, :offset_2].any(dim=0).data

        loss_er = torch.tensor(0.)
        loss_der = torch.tensor(0.)
        loss_afd = torch.tensor(0.)

        if len(self.buffer) == 0:
            loss_afd, stream_attention_maps, losses = self.partial_distill_loss(
                stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, labels, task_labels=stream_task_labels)
        else:
            buffer_teacher_forcing = buf_task_labels != self.current_task
            teacher_forcing = torch.cat(
                (torch.zeros((B)).bool().to(self.device), buffer_teacher_forcing))
            attention_maps = [
                [torch.ones_like(map) for map in buf_attention_maps[0]]]*B + buf_attention_maps

            loss_afd, all_attention_maps, losses = self.partial_distill_loss(all_partial_features[-len(
                all_pret_partial_features):], all_pret_partial_features, all_labels,
                teacher_forcing, attention_maps, task_labels=all_task_labels)

            stream_attention_maps = [ap[:B] for ap in all_attention_maps]

            loss_er = self.loss(buf_outputs[:, :offset_2], buf_labels[:, :offset_2].float())
            der_buf_outputs = buf_outputs.clone()
            der_buf_outputs[~buf_logits_mask] = 0.0
            der_buf_logits = buf_logits.clone()
            der_buf_logits[~buf_logits_mask] = 0.0
            loss_der = F.mse_loss(der_buf_outputs, der_buf_logits)

        for k, v in losses.items():
            locals()[k] = v

        loss += self.args.der_beta * loss_er
        loss += self.args.der_alpha * loss_der        
        loss += self.args.lambda_fp * loss_afd
        loss += self.args.lambda_fp_replay * loss_attention_maps_replay

        self.opt.zero_grad()
        if self.opt_adapters:
            self.opt_adapters.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad)
        self.opt.step()
        if self.opt_adapters:
            self.opt_adapters.step()

        if output_mask.sum() > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                labels=labels,
                logits=stream_logits.data,
                attention_maps=stream_not_aug_attention_maps,
                task_labels=torch.ones(B)*self.current_task,
                logits_mask=output_mask.data)

        return loss.item()
    
    def forward(self, x):
        offset_1, offset_2 = self._compute_offsets(self.current_task - 1)
        return self.net(x)[:, :offset_2]

