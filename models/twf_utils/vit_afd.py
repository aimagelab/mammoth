import os
from typing import Tuple
import torch
from torch import nn
import wandb

import sys
import warnings

conf_path = os.getcwd()
sys.path.append(conf_path)

from models.twf_utils.afd import DiverseLoss, TestLoss, ChannelDiverseLoss, DiverseLossFixDiagonal, DiverseLossFixNormalized, DiverseLossFixNormalizedDiagonal
from models.twf_utils.afd import Normalize
from models.twf_utils.afd import TeacherTransform
from models.twf_utils.afd import TeacherForcingLoss
from models.twf_utils.adapters import MixerAttention
from models.twf_utils.adapters import ChannelAttention
from models.twf_utils.adapters import DoubleAttention
from models.twf_utils.adapters import MockAttention
from models.twf_utils.adapters import MimickingAttention
from models.twf_utils.adapters import DoubleAttentionViT
from models.twf_utils.adapters import TransformerAttention, TransformerAttentionLayerNorm, AttentionProbeCls, AttentionProbeClsNorm
from models.twf_utils.adapters import ClipCrossAttention,TransformerAttentionClip, TransformerAttentionProj, TaT, TaTV2, TaTNorm
from models.twf_utils.adapters import AttentionProbeClsNoGumbel, AttentionProbeClsNormNoGumbel
from models.twf_utils.adapters_v2 import ClipCrossAttentionV2
from models.twf_utils.adapters_v2 import ClipCrossAttentionV3, ClipCrossAttentionV4, ClipCrossAttentionV5, ClipCrossAttentionV6, ClipCrossAttentionV7

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class TokenizedDistillation(nn.Module):

    def __init__(self, input_shape: Tuple[int], n_tasks: int,
                 num_classes: int, adatype: str = 'mixer',
                 lambda_diverse_loss: float = 0.0,
                 teacher_forcing_or: bool = False, 
                 lambda_forcing_loss: float = 0.0,
                 use_prompt: bool = False,
                 use_conditioning: bool = True,
                 exclude_class_token: bool = False,
                 lambda_ignition_loss: float = 0.0,
                 ignition_loss_temp: float = 0.2,
                 diverse_loss_mode='original'):

        super().__init__()

        self.input_shape = input_shape
        self.adatype = adatype
        self.seq_len = input_shape[0]
        self.embed_dim = input_shape[1]
        self.n_tasks = n_tasks
        self.num_classes = num_classes
        self.teacher_forcing_or = teacher_forcing_or
        self.lambda_forcing_loss = lambda_forcing_loss
        self.use_prompt = use_prompt
        self.use_conditioning = use_conditioning
        self.exclude_class_token = exclude_class_token
        self.lambda_ignition_loss = lambda_ignition_loss
        self.ignition_loss_temp = ignition_loss_temp
        self.diverse_loss_mode = diverse_loss_mode

        if self.exclude_class_token is False and self.adatype == 'double_convit':
            warnings.warn("exclude_class_token is False, but adatype is double_convit. \
                           This is not supported. Set exclude_class_token to True.")
            self.exclude_class_token = True

        if self.exclude_class_token:
            self.seq_len -= 1

        self.attn_fn = self.build_attn_module()

        self.teacher_forcing_loss = TeacherForcingLoss(self.teacher_forcing_or, self.lambda_forcing_loss)
        self.teacher_transform = TeacherTransform(dims=(0, 1))
        self.norm = Normalize(dims=(1,))
        if self.diverse_loss_mode == 'original':
            self.diverse_loss = DiverseLoss(lambda_diverse_loss)
        elif self.diverse_loss_mode == 'fix_diagonal':
            self.diverse_loss = DiverseLossFixDiagonal(lambda_diverse_loss)
        elif self.diverse_loss_mode == 'fix_normalized':
            self.diverse_loss = DiverseLossFixNormalized(lambda_diverse_loss)
        elif self.diverse_loss_mode == 'fix_normalized_diagonal':
            self.diverse_loss = DiverseLossFixNormalizedDiagonal(lambda_diverse_loss)
        else:
            raise NotImplementedError("diverse_loss_mode unknown")
        #self.test_loss = TestLoss()
        #self.channel_diverse_loss = ChannelDiverseLoss(lambda_diverse_loss)

        self.resize_maps = None

    def build_attn_module(self):
        if self.adatype == 'attention_probe_cls_norm_no_gumbel':
            return AttentionProbeClsNormNoGumbel(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'attention_probe_cls_no_gumbel':
            return AttentionProbeClsNoGumbel(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'attention_probe_cls_norm':
            return AttentionProbeClsNorm(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'attention_probe_cls':
            return AttentionProbeCls(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'transformer_pretrained_layer_norm':
            return TransformerAttentionLayerNorm(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'tat_norm':
            return TaTNorm(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'tat_v2':
            return TaTV2(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'tat':
            return TaT(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'transformer_pretrained_proj':
            return TransformerAttentionProj(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'transformer_pretrained_clip':
            return TransformerAttentionClip(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v7':
            return ClipCrossAttentionV7(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v6':
            return ClipCrossAttentionV6(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v5':
            return ClipCrossAttentionV5(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v4':
            return ClipCrossAttentionV4(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'transformer_pretrained':
            return TransformerAttention(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention':
            return ClipCrossAttention(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v2':
            return ClipCrossAttentionV2(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'clip_cross_attention_v3':
            return ClipCrossAttentionV3(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'transformer':
            return TransformerAttention(self.embed_dim, self.n_tasks, self.use_conditioning)
        if self.adatype == 'mixer':
            return MixerAttention(self.seq_len, self.input_shape[1], self.n_tasks, \
                                          self.num_classes, use_prompt=self.use_prompt, \
                                          use_conditioning=self.use_conditioning)

        elif self.adatype == 'channel':
            assert self.use_prompt is False, "not supported"
            return ChannelAttention(c_in=self.input_shape[1], n_tasks=self.n_tasks, reduction_rate=1,
                                            use_conditioning=self.use_conditioning, activated_with_softmax=True)

        elif self.adatype == 'double_mixer':
            return DoubleAttention(self.seq_len, self.input_shape[1], self.n_tasks, \
                                           self.num_classes, use_prompt=self.use_prompt, \
                                           use_conditioning=self.use_conditioning, sp_attn_type='mixer')

        elif self.adatype == 'double_convit':
            assert self.use_prompt is False, 'not supported'
            assert self.use_conditioning is False, 'not supported'
            assert self.exclude_class_token is True, 'not supported'
            return DoubleAttention(self.seq_len, self.input_shape[1], self.n_tasks, \
                                           self.num_classes, use_prompt=self.use_prompt, \
                                           use_conditioning=self.use_conditioning, sp_attn_type='convit')
        elif self.adatype == 'mock':
            return MockAttention()
        elif self.adatype == 'mimicking':
            return MimickingAttention()
        elif self.adatype == 'twf_original':
            self.exclude_class_token = True
            return DoubleAttentionViT(self.seq_len, self.embed_dim, self.n_tasks, use_conditioning=self.use_conditioning)
        else:
            raise ValueError
        return None

    def get_tasks_id(self, targets):
        if 'ablation_type' in os.environ and os.environ['ablation_type'] == 'non_cond':
            return torch.zeros_like(targets)
        return torch.div(targets, self.cpt, rounding_mode='floor')

    def extend_like(self, teacher_forcing, y):
        dest_shape = (-1,) + (1,) * (len(y.shape) - 1)
        return teacher_forcing.view(dest_shape).expand(y.shape)
    
    def compute_ignition_loss(self, x, tau=0.2):
        # TODO: magari fare la media per ogni singolo canale, poi fare il log, poi fare la media su tutti i canali, poi fare la media su tutto il batch
        # x (b, n, d)
        x = torch.mean(x, dim=(1, 2))
        x = -torch.log(x + 1e-8) / tau
        x = torch.mean(x)
        return x

    def forward(self, fm_s, fm_t, targets, teacher_forcing, attention_map, task_labels):

        assert len(targets) == len(fm_s) == len(fm_t) == len(teacher_forcing) == len(attention_map)

        if self.exclude_class_token:
            fm_s = fm_s[:, 1:]
            fm_t = fm_t[:, 1:]
        
        #task_ids = self.get_tasks_id(targets)

        # if self.adatype == 'twf_original':
        #     fm_t = fm_t[:, 1:].view(fm_t.shape[0], 14, 14, 768).permute(0, 3, 1, 2)
        #     fm_s = fm_s[:, 1:].view(fm_s.shape[0], 14, 14, 768).permute(0, 3, 1, 2)
        #     output_rho, logits = self.attn_fn(fm_t, task_ids)
        # else:
        #     output_rho, logits = self.attn_fn(fm_t, targets, task_ids)
        output_rho, logits = self.attn_fn(fm_t, targets, task_labels)

        loss = .0
        losses = {}

        #ignition_loss = -torch.log(output_rho.mean() + 1e-8)
        # ignition_loss = self.compute_ignition_loss(output_rho, self.ignition_loss_temp)
        # loss += self.lambda_ignition_loss * ignition_loss
        
        losses['output_rho_sum_loss'] = output_rho.sum().item()
        losses['output_rho_mean_loss'] = output_rho.mean().item()
        # losses['ignition_loss'] = ignition_loss.item()

        rho = output_rho

        if not self.lambda_forcing_loss > 0.0:
            if teacher_forcing.any():
                p1 = torch.max(attention_map, output_rho) if self.teacher_forcing_or else attention_map
                rho = torch.where(self.extend_like(teacher_forcing, attention_map), p1, output_rho)
            else:
                rho = output_rho
        # elif teacher_forcing.any():
        #     if 'ablation_type' not in os.environ or os.environ['ablation_type'] != 'no_mask_replay':
        #         loss_tf = self.teacher_forcing_loss(logits, output_rho,
        #                                       attention_map, teacher_forcing)
        #         losses['teacher_forcing_loss'] = loss_tf.item()
        #         loss += loss_tf

        #fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)

        loss_dist = self.attn_fn.compute_distance(fm_s, fm_t, rho)
        losses['distillation_loss'] = loss_dist.item()
        loss += loss_dist

        if ('ablation_type' not in os.environ or os.environ['ablation_type'] != 'no_diverse') and type(self.attn_fn) != MockAttention:
            loss_div = self.diverse_loss(rho[~teacher_forcing])
            losses['diverse_loss'] = loss_div.item()
            loss += loss_div
        
        # loss_channel_diverse = self.channel_diverse_loss(rho[~teacher_forcing])
        # losses['loss_channel_diverse'] = loss_channel_diverse.item()
        # loss += loss_channel_diverse
        
        #test_loss = self.test_loss(rho)

        return loss, output_rho, losses


if __name__ == '__main__':
    x_s = torch.randn((32, 197, 768))
    x_t = torch.randn((32, 197, 768))
    targets = torch.randint(0, 10, (32,))
    teacher_forcing = torch.zeros(len(targets), dtype=torch.bool)
    attention_map = torch.empty((32, 197, 768)).random_(2)
    n_tasks = 5
    cpt = 2

    # with torch.no_grad():
    #     d = DistillationMLPMixer(x_s.shape[1:], adatype='mixer',
    #                              n_tasks=n_tasks, cpt=cpt, clear_grad=False,
    #                              teacher_forcing_or=False,
    #                              lambda_forcing_loss=0.1,
    #                              lambda_diverse_loss=0.1,
    #                              use_conditioning=False,
    #                              use_prompt=False,
    #                              exclude_class_token=True)

    #     loss, output_rho = d(x_s, x_t, targets, teacher_forcing=teacher_forcing, attention_map=attention_map)
    #     print(output_rho[0])