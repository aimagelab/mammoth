import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models import named_apply
from timm.layers import trunc_normal_

from backbone.vit import LoRAAttention, create_vision_transformer, VisionTransformer as MammothVP, get_init_weights_vit
from models.dualprompt_utils.prompt import EPrompt
from models.dualprompt_utils.attention import PreT_Attention


class VisionTransformer(MammothVP):

    def __init__(
            self, prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False, use_g_prompt=False,
            g_prompt_length=None, g_prompt_layer_idx=None, use_prefix_tune_for_g_prompt=False, use_e_prompt=False, e_prompt_layer_idx=None,
            use_prefix_tune_for_e_prompt=False, same_key_value=False, args=None, **kwargs):

        if not (use_g_prompt or use_e_prompt):
            attn_layer = LoRAAttention
        elif not (use_prefix_tune_for_g_prompt or use_prefix_tune_for_e_prompt):
            # Prompt tunning
            attn_layer = LoRAAttention
        else:
            # Prefix tunning
            attn_layer = PreT_Attention

        super().__init__(args=args, attn_layer=attn_layer, **kwargs)

        self.prompt_pool = prompt_pool
        self.use_permute_fix = args.use_permute_fix if hasattr(args, 'use_permute_fix') else False
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.use_g_prompt = use_g_prompt
        self.g_prompt_layer_idx = g_prompt_layer_idx

        # num_g_prompt : The actual number of layers to which g-prompt is attached.
        # In official code, create as many layers as the total number of layers and select them based on the index
        num_g_prompt = len(self.g_prompt_layer_idx) if self.g_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt

        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt

        if not self.use_prefix_tune_for_g_prompt and not self.use_prefix_tune_for_g_prompt:
            self.use_g_prompt = False
            self.g_prompt_layer_idx = []

        if use_g_prompt and g_prompt_length is not None and len(g_prompt_layer_idx) != 0:
            if not use_prefix_tune_for_g_prompt:
                g_prompt_shape = (num_g_prompt, g_prompt_length, self.embed_dim)
                if prompt_init == 'zero':
                    self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                elif prompt_init == 'uniform':
                    self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                    nn.init.uniform_(self.g_prompt, -1, 1)
            else:
                if same_key_value:
                    g_prompt_shape = (num_g_prompt, 1, g_prompt_length, self.num_heads, self.embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
                    self.g_prompt = self.g_prompt.repeat(1, 2, 1, 1, 1)
                else:
                    g_prompt_shape = (num_g_prompt, 2, g_prompt_length, self.num_heads, self.embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
        else:
            self.g_prompt = None

        if use_e_prompt and e_prompt_layer_idx is not None:
            self.e_prompt = EPrompt(length=prompt_length, embed_dim=self.embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                                    prompt_key_init=prompt_key_init, num_layers=num_e_prompt, use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                                    num_heads=self.num_heads, same_key_value=same_key_value, use_permute_fix=self.use_permute_fix)

        self.total_prompt_len = 0
        if self.prompt_pool:
            if not self.use_prefix_tune_for_g_prompt:
                self.total_prompt_len += g_prompt_length * len(self.g_prompt_layer_idx)
            if not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)

        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        if self.weight_init != 'skip':
            self.init_weights(self.weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        if self.use_g_prompt or self.use_e_prompt:
            if self.use_prompt_mask and train:
                start = task_id * self.e_prompt.top_k
                end = (task_id + 1) * self.e_prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.e_prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None

            g_prompt_counter = -1
            e_prompt_counter = -1

            res = self.e_prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            e_prompt = res['batched_prompt']

            for i, block in enumerate(self.blocks):
                if i in self.g_prompt_layer_idx:
                    if self.use_prefix_tune_for_g_prompt:
                        g_prompt_counter += 1
                        # Prefix tunning, [B, 2, g_prompt_length, num_heads, embed_dim // num_heads]
                        idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                        g_prompt = self.g_prompt[idx]
                    else:
                        g_prompt = None
                    x = block(x, prompt=g_prompt)

                elif i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1
                    if self.use_prefix_tune_for_e_prompt:
                        # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                        x = block(x, prompt=e_prompt[e_prompt_counter])
                    else:
                        # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                        prompt = e_prompt[e_prompt_counter]
                        x = torch.cat([prompt, x], dim=1)
                        x = block(x)
                else:
                    x = block(x)
        else:
            x = self.blocks(x)

            res = dict()

        x = self.norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')

        res['pre_logits'] = x

        x = self.fc_norm(x)

        res['logits'] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # modify
    logging.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        # ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > gs_old ** 2:
        ntok_new -= gs_old ** 2
        # expand cls's pos embedding for prompt tokens
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    logging.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def vit_base_patch16_224_dualprompt(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', base_class=VisionTransformer, filter_fn=checkpoint_filter_fn, pretrained=pretrained, **model_kwargs)
    return model
