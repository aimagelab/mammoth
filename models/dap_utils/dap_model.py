"""
vit with DAP
"""
import copy
import math
import os
from os.path import join as pjoin

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

from scipy import ndimage

from backbone.vit import Block as MammothBlock, VisionTransformer as MammothVit
from models.dap_utils.head import MLP

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class ADPT_Block(nn.Module):
    def __init__(self, embed_dim, task_emb, num_dap_tokens, is_imgr, ext_block: MammothBlock, disable_dap_block=False):
        super().__init__()
        self.disable_dap_block = disable_dap_block
        self.embed_dim = embed_dim
        if not disable_dap_block:
            self.is_imgr = is_imgr
            self.num_dap_tokens = num_dap_tokens

            # domain-adaptive prompts
            self.dap_downsample = nn.Linear(197, num_dap_tokens)
            nn.init.zeros_(self.dap_downsample.weight)
            nn.init.zeros_(self.dap_downsample.bias)
            self.dap_film = nn.Linear(task_emb, self.embed_dim * 2)
            self.dap_norm = LayerNorm(self.embed_dim, eps=1e-6)

        self.attn = copy.deepcopy(ext_block.attn)
        self.mlp = copy.deepcopy(ext_block.mlp)
        self.norm1 = copy.deepcopy(ext_block.norm1)
        self.norm2 = copy.deepcopy(ext_block.norm2)

    def forward(self, x, task_id_estimated_emb=None, layer_index=None):
        if not self.disable_dap_block:
            if x.shape[1] == 197:  # first layer
                x_norm = self.dap_norm(x)
                x_tran = torch.transpose(x_norm, 2, 1)
                down = self.dap_downsample(x_tran)

                film = self.dap_film(task_id_estimated_emb)
                gamma4 = film[:, :self.embed_dim]
                beta4 = film[:, self.embed_dim:]
                gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
                beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

                gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
                beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
                down = gamma4 * down + beta4
                down = torch.transpose(down, 2, 1)

                x = torch.cat((
                    x[:, :1, :],
                    down,
                    x[:, 1:, :]
                ), dim=1)
            else:
                x = torch.cat((
                    x[:, :1, :],
                    x[:, (1 + self.num_dap_tokens):, :]
                ), dim=1)

                x_norm = self.dap_norm(x)
                x_tran = torch.transpose(x_norm, 2, 1)
                down = self.dap_downsample(x_tran)

                film = self.dap_film(task_id_estimated_emb)
                gamma4 = film[:, :self.embed_dim]
                beta4 = film[:, self.embed_dim:]
                gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
                beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

                gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
                beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
                down = gamma4 * down + beta4
                down = torch.transpose(down, 2, 1)

                if not (layer_index == 11 and self.is_imgr):
                    # for imagenet_r, do not append prompts on the last layer
                    x = torch.cat((
                        x[:, :1, :],
                        down,
                        x[:, 1:, :]
                    ), dim=1)

        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.embed_dim,
                                                                                   self.embed_dim).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.embed_dim, self.embed_dim).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.embed_dim,
                                                                                   self.embed_dim).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.embed_dim,
                                                                                   self.embed_dim).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            qkv_weight = torch.cat((query_weight, key_weight, value_weight))
            qkv_bias = torch.cat((query_bias, key_bias, value_bias))

            self.attn.qkv.weight.copy_(qkv_weight)
            self.attn.proj.weight.copy_(out_weight)
            self.attn.qkv.bias.copy_(qkv_bias)
            self.attn.proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.norm1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.norm1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.norm2.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.norm2.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ADPT_Encoder(MammothVit):
    def __init__(self, args, disable_dap_block=False, **kwargs):
        super().__init__(**kwargs)
        self.disable_dap_block = disable_dap_block

        for i in range(len(self.blocks)):
            layer = ADPT_Block(self.embed_dim, args.task_emb, args.num_dap_tokens,
                               args.dataset == 'seq-imagenet-r', self.blocks[i], disable_dap_block=disable_dap_block)
            self.blocks[i] = layer

    def forward(self, x: torch.Tensor, task_id_estimated_emb=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        # NOTE: grad checkpointing was removed from the original timm impl
        for idx, blk in enumerate(self.blocks):
            if self.disable_dap_block:
                x = blk(x)
            else:
                x = blk(x, task_id_estimated_emb=task_id_estimated_emb, layer_index=idx)
        x = self.norm(x)

        return x


def expand_to_batch(x, batch_size, dim=0, device=None):
    shape = [1 for _ in x.shape]
    shape.insert(dim, batch_size)
    return torch.tile(torch.unsqueeze(x, dim=dim), shape).to(device)


class ADPT_Transformer(nn.Module):
    def __init__(self, backbone, args, n_tasks, **kwargs):
        super(ADPT_Transformer, self).__init__()
        self.enable_test_time_majority_voting = args.enable_test_time_majority_voting
        self.encoder = ADPT_Encoder(args, **kwargs)
        self.load_original_checkpoint = args.load_original_checkpoint

        if args.load_original_checkpoint:
            self.pretrained_enc = ADPT_Encoder(args, disable_dap_block=True, **kwargs)
        else:
            self.pretrained_enc = backbone
            self.pretrained_enc.head = nn.Identity()

        self.patch_size = _pair(kwargs['patch_size'])
        self.prompt_dim = kwargs['embed_dim']
        self.pool_size = n_tasks

        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))
        self.dap_key_embeddings = nn.Parameter(torch.zeros(self.pool_size, self.prompt_dim))
        nn.init.uniform_(self.dap_key_embeddings.data, -val, val)
        self.dap_emb = torch.nn.Embedding(n_tasks, 16)

        self.top_k = 1

    def forward(self, input_ids, task_id=None, is_train=None):
        B = input_ids.shape[0]
        x_cls_embed = self.pretrained_enc(input_ids).detach()
        if self.load_original_checkpoint:
            x_cls_embed = x_cls_embed[:, 0]

        if is_train:
            start = task_id * self.top_k
            end = (task_id + 1) * self.top_k
            prompt_mask = torch.arange(start, end).to(input_ids.device)
            if end > self.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None

        dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1)

        x_embed_norm = F.normalize(x_cls_embed, dim=-1)
        sim = torch.matmul(dap_prompt_key_norm,
                           torch.transpose(x_embed_norm, 1, 0))

        sim = torch.transpose(sim, 1, 0)
        (sim_top_k, idx) = torch.topk(sim, self.top_k)
        idx = idx.squeeze(dim=-1)

        if is_train or self.enable_test_time_majority_voting:
            # majority voting
            prompt_id, id_counts = torch.unique(idx, return_counts=True)
            _, major_idx = torch.topk(id_counts, self.top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = expand_to_batch(major_prompt_id, x_cls_embed.shape[0], device=input_ids.device).squeeze(dim=-1)

        if prompt_mask is not None:
            idx = prompt_mask
            idx = expand_to_batch(idx, x_cls_embed.shape[0], device=input_ids.device).squeeze(dim=-1)

        task_id_estimated_emb = self.dap_emb(idx)

        i = torch.arange(B).reshape(B, 1, 1)
        l = torch.arange(self.prompt_dim).reshape(1, 1, self.prompt_dim)

        selected_prompt_key = dap_prompt_key_norm.repeat(B, 1, 1)[
            i, idx.unsqueeze(-1), l]

        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim_pull = selected_prompt_key * x_embed_norm
        reduce_sim = torch.sum(sim_pull) / x_cls_embed.shape[0]

        # embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(input_ids, task_id_estimated_emb=task_id_estimated_emb)

        return encoded, reduce_sim


class ADPT_VisionTransformer(nn.Module):
    def __init__(self, backbone, args, n_tasks, **kwargs):
        super().__init__()
        self.n_tasks = n_tasks

        self.transformer = ADPT_Transformer(backbone, args, n_tasks, **kwargs)
        self.head = nn.Identity()

        self.top_k = 1

    def forward(self, x, task_id=None, vis=False, is_train=None):
        x, reduce_sim = self.transformer(x, task_id, is_train=is_train)

        logits = self.head(x[:, 0])

        if not vis:
            return logits, reduce_sim
        return logits, reduce_sim


def dap_load_from(model: ADPT_Transformer, weights):
    with torch.no_grad():
        model.patch_embed.proj.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
        model.patch_embed.proj.bias.copy_(np2th(weights["embedding/bias"]))
        model.cls_token.copy_(np2th(weights["cls"]))
        model.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        model.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        posemb_new = model.pos_embed
        if posemb.size() == posemb_new.size():
            model.pos_embed.copy_(posemb)
        else:
            ntok_new = posemb_new.size(1)

            if model.classifier == "token":
                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new -= 1
            else:
                posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
            model.pos_embed.copy_(np2th(posemb))

        for idx, block in enumerate(model.blocks):
            block.load_from(weights, idx)


class DAPModel(nn.Module):
    def __init__(self, backbone, n_tasks, args, num_classes, device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.args = args

        self.enc = ADPT_VisionTransformer(backbone=backbone, n_tasks=self.n_tasks, args=args,
                                          img_size=224, patch_size=16, embed_dim=768, depth=12,
                                          num_heads=12, drop_path_rate=0, num_classes=0).to(self.device)

        if not args.load_original_checkpoint:
            st = backbone.state_dict()
            st = {k: v for k, v in st.items() if "dap" not in k}
            missing, unexpected = self.enc.transformer.encoder.load_state_dict(st, strict=False)
            assert len([m for m in missing if 'head' not in m and 'dap' not in m]) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        else:
            dap_load_from(self.enc.transformer.encoder, np.load(os.path.join("./data", "imagenet21k_ViT-B_16.npz")))
            dap_load_from(self.enc.transformer.pretrained_enc, np.load(os.path.join("./data", "imagenet21k_ViT-B_16.npz")))

        for k, p in self.enc.named_parameters():
            if "dap" not in k:
                p.requires_grad = False

        self.head = MLP(
            input_dim=768,
            mlp_dims=[num_classes],
            special_bias=True
        ).to(self.device)

    def forward(self, x, task_id=None, is_train=False, n_past_classes=None, n_cur_classes=None):
        x, reduce_sim = self.enc(x, task_id=task_id, is_train=is_train)
        x = self.head(x)

        if not is_train:
            return x

        if self.args.dataset == 'seq-imagenet-r':
            # only for imagenet_r
            offset1 = n_past_classes
            offset2 = n_cur_classes
            if offset1 > 0:
                x[:, :offset1].data.fill_(-10e10)
            if offset2 < x.shape[1]:
                x[:, int(offset2):].data.fill_(-10e10)

        return x, reduce_sim
