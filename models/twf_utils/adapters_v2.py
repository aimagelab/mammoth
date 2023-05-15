import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import math

import einops
from einops import rearrange

from models.twf_utils.afd import BinaryGumbelSoftmax
from models.twf_utils.afd import HardAttentionSoftmax
from models.twf_utils.utils import ConditionalLinear
from models.twf_utils.utils import WrapperNOTConditionalLinear
from models.twf_utils.utils import TaskPrompter
from models.twf_utils.utils import MLPMixer
from models.twf_utils.utils import PiecewiseRect
from models.twf_utils.utils import MLPMixerWithBottleneck
from models.twf_utils.utils import ConVitWithBottleneck
from models.twf_utils.utils import MyGumbelSoftmax
from models.twf_utils.arol import Linear as LandiLinear
from models.twf_utils.afd import ConditionalLinear as ConditionalLinearOriginal
from models.twf_utils.afd import ConditionalBatchNorm2d

from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Block, LayerScale

class LandiAttention(nn.Module):

    def __init__(
            self,
            dim,
            ctx_dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            r: int = 32,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False
    ):

        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = LandiLinear(dim, dim * 3, ctx_features=ctx_dim, bias=qkv_bias, r=r,
                               lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                               fan_in_fan_out=fan_in_fan_out)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ctx):

        B, N, C = x.shape
        B1, C1 = ctx.shape

        assert B1 == B and C == C1

        qkv = self.qkv(x, ctx).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LandiBlock(nn.Module):

    def __init__(
            self,
            dim,
            ctx_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LandiAttention(
            dim,
            ctx_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, ctx):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), ctx)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ClipCrossAttentionV2(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = Block(self.embed_dim, num_heads=4, qkv_bias=True)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        clip_emb = x[:, -1:, :]
        x = x[:, :-1, :]

        x = self.self_attn(x)

        Q = clip_emb.repeat(1, x.shape[1], 1)
        x = self.proj(x) * Q

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist[:, :-1, :]
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class ClipCrossAttentionV4(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = Block(self.embed_dim, num_heads=4, qkv_bias=True)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        clip_emb = x[:, -1:, :]
        x = x[:, :-1, :]

        Q = clip_emb.repeat(1, x.shape[1], 1)
        x = self.proj(x) * Q

        x = self.self_attn(x)

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist[:, :-1, :]
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class ClipCrossAttentionV5(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.embed_dim = embed_dim

        self.self_attn = Block(dim=192, num_heads=3, qkv_bias=True)    # ViT tiny block

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)        
        self.blck_proj_1 = nn.Linear(self.embed_dim, 192)
        self.blck_proj_2 = nn.Linear(192, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        clip_emb = x[:, -1:, :]
        x = x[:, :-1, :]

        Q = clip_emb.repeat(1, x.shape[1], 1)
        x = self.proj(x) * Q

        x = self.blck_proj_1(x) # 768 -> 192
        x = self.self_attn(x) # 192 -> 192
        x = self.blck_proj_2(x) # 192 -> 768

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist[:, :-1, :]
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist
    

class ClipCrossAttentionV6(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.embed_dim = embed_dim

        self.self_attn = Block(dim=192, num_heads=3, qkv_bias=True)    # ViT tiny block

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.blck_proj_1 = nn.Linear(self.embed_dim, 192)
        self.blck_proj_2 = nn.Linear(192, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        clip_emb = x[:, -1:, :]
        x = x[:, :-1, :]

        x = self.blck_proj_1(x) # 768 -> 192
        x = self.self_attn(x) # 192 -> 192
        x = self.blck_proj_2(x) # 192 -> 768

        Q = clip_emb.repeat(1, x.shape[1], 1)
        x = self.proj(x) * Q

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist[:, :-1, :]
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class ClipCrossAttentionV7(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.embed_dim = embed_dim

        self.self_attn = Block(dim=192, num_heads=3, qkv_bias=True)    # ViT tiny block

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.blck_proj_1 = nn.Linear(self.embed_dim, 192)
        self.blck_proj_2 = nn.Linear(192, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        x = self.blck_proj_1(x) # 768 -> 192
        x = self.self_attn(x) # 192 -> 192
        x = self.blck_proj_2(x) # 192 -> 768

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist


class ClipCrossAttentionV3(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = LandiBlock(self.embed_dim, ctx_dim=self.embed_dim,
                                    num_heads=4, qkv_bias=True, mlp_ratio=1)

        #print("Num parameters", sum([t.numel() for t in self.self_attn.attn.qkv.lora_A.parameters()]))
        #print("Num parameters", sum([t.numel() for t in self.self_attn.attn.qkv.lora_B.parameters()]))

        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        x, clip_emb = x[:, :-1, :], x[:, -1, :]

        x = self.self_attn(x, clip_emb)

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist[:, :-1, :]
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist



if __name__ == '__main__':

    B = 32
    N_TOKENS = 196
    N_CHANNELS = 768
    N_CLASSES = 10
    N_TASKS = 5

    fm_t = torch.rand((B, N_TOKENS, N_CHANNELS))
    y = torch.randint(0, N_CLASSES, (B,))
    tasks_id = y // N_TASKS

    clip_cross_attn = ClipCrossAttentionV3(embed_dim=N_CHANNELS, n_tasks=5,
                                         use_conditioning=True)

    rho, logits = clip_cross_attn(fm_t, y, tasks_id)
    print(rho.shape, logits.shape)