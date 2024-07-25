import collections.abc
from itertools import repeat
from torch import nn
import torch
import torch.nn.functional as F


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class LoRALayer():
    def __init__(
            self,
            lora_dropout: float,
    ):
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoRALinear(nn.Linear, LoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor, AB: dict = None):

        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            A = None
            if isinstance(AB, dict):
                B = AB['B']
                A = AB.get('A')
            else:
                B = AB
            if A is not None:
                return result + (B @ (A @ x.transpose(1, 2).unsqueeze(1))).sum(1).transpose(1, 2)
            return result + (B @ x.transpose(1, 2).unsqueeze(1)).sum(1).transpose(1, 2)

        return result


class LoRAAttention(nn.Module):
    """
    Attention layer as used in Vision Transformer.
    Adapted to support LoRA-style parameters.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to q, k, v
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate after the final projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = LoRALinear(dim, dim * 3, 0., bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LoRALinear(dim, dim, 0.)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, AB: dict = None, **kwargs):
        """
        Forward pass of the attention layer.
        Supports `AB` for LoRA-style parameters (checkout docs for `VisionTransformer.forward`).

        Args:
            x: Input tensor
            AB: Dictionary containing LoRA-style parameters for the layer
        """

        B, N, C = x.shape

        AB_qkv = None

        if AB is not None:
            AB_qkv = AB.get("qkv")

        qkv = self.qkv(x, AB_qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # NOTE: flash attention is less debuggable than the original. Use the commented code below if in trouble.
        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)

        AB_proj = None

        if AB is not None:
            AB_proj = AB.get("proj")

        x = self.proj(x, AB_proj)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LoRAMlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    Adapted to support LoRA-style parameters.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        assert use_conv is False

        self.fc1 = LoRALinear(in_features, hidden_features, bias=bias[0], lora_dropout=0.)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = LoRALinear(hidden_features, out_features, bias=bias[1], lora_dropout=0.)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor, AB: dict = None, **kwargs):
        """
        Forward pass of the MLP layer.
        Supports `AB` for LoRA-style parameters (checkout docs for `VisionTransformer.forward`).

        Args:
            x: Input tensor
            AB: Dictionary containing LoRA-style parameters for the layer
        """
        AB_fc1 = None
        AB_fc2 = None

        if AB is not None:
            AB_fc1 = AB.get("fc1")
            AB_fc2 = AB.get("fc2")

        x = self.fc1(x, AB_fc1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, AB_fc2)
        x = self.drop2(x)

        return x
