import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.vit import VisionTransformer as MammothVP, Block as MammothViTBlock
from models.coda_prompt_utils.vit import Attention as PrefixTuningAttention


class ResidualPromptAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompts=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompts is not None:
            prompts = prompts.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + prompts

        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(MammothViTBlock):
    def forward(self, x, prompts=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompts)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(MammothVP):

    def __init__(self, *args, prompt_mode='residual', **kwargs):
        super().__init__(*args, **kwargs)
        assert prompt_mode in ['residual', 'concat'], 'prompt_mode should be either residual or concat'

        attn_layer = ResidualPromptAttention if prompt_mode == 'residual' else PrefixTuningAttention

        self.blocks = nn.Sequential(*[
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                init_values=self.init_values,
                drop=self.pos_drop.p,
                attn_drop=self.attn_drop_rate,
                attn_layer=attn_layer,
                drop_path=self.dpr[i],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer
            )
            for i in range(self.depth)])

        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        if self.weight_init != 'skip':
            self.init_weights(self.weight_init)

    def forward_features(self, x, first_stage_query, prompter, cur_classes: int, frozen_past_classes=0):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        for idx, blk in enumerate(self.blocks):
            prompts = prompter.get_prompts(idx, first_stage_query, frozen_past_classes=frozen_past_classes, cur_classes=cur_classes)
            if prompts is not None:
                x = blk(x, prompts)
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, first_stage_query: torch.Tensor, prompter, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:
        """
        Compute the forward of STAR-Prompt.

        Args:
            x: input image
            query: the output of the visual encoder of CLIP, to be used as query for the second stage's prompter
            prompter: the prompter of the second stage
            train: whether the model is in training mode. If True, the prompts of the past tasks will be frozen and only the current task's prompts will be updated. Else, all prompts will be frozen.
        """
        x = self.forward_features(x, first_stage_query, prompter, cur_classes, frozen_past_classes)
        x = self.forward_head(x)
        return x
