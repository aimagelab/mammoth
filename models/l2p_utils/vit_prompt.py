""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from backbone.vit import create_vision_transformer, VisionTransformer as MammothVP
from models.l2p_utils.prompt import Prompt


class VisionTransformer(MammothVP):
    def __init__(
            self, prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False, prompt_shuffle=False, args=None, **kwargs):

        super().__init__(args=args, **kwargs)

        self.num_prefix_tokens = 1 if self.class_token else 0

        embed_len = self.pos_embed.shape[1]
        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.prompt_shuffle = prompt_shuffle

        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(length=prompt_length, embed_dim=self.embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                                 prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                                 prompt_key_init=prompt_key_init, prompt_shuffle=self.prompt_shuffle)

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        if self.weight_init != 'skip':
            self.init_weights(self.weight_init)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        perv_pos_embed = self.pos_embed.clone()
        c_device = next(self.parameters()).device
        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, self.embed_dim, device=c_device) * .02)
        super().load_pretrained(checkpoint_path, prefix)
        self.pos_embed.data = resize_pos_embed(self.pos_embed, perv_pos_embed, num_prefix_tokens=self.num_prefix_tokens, gs_new=self.patch_embed.grid_size)
        # TODO: CHECK

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        if hasattr(self, 'prompt'):
            if self.use_prompt_mask and train:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res = dict()
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)

        x = self.norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
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

    def forward(self, x, task_id=-1, cls_features=None, train=False, returnt='out'):
        assert returnt in ('out', 'features', 'both')

        feats = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        if returnt == 'features':
            return feats

        res = self.forward_head(feats)
        if returnt == 'both':
            return res, feats
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


def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', base_class=VisionTransformer,
                                      pretrained=pretrained, filter_fn=checkpoint_filter_fn, **model_kwargs)
    return model
