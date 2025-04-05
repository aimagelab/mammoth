import torch
from copy import deepcopy
from kornia.augmentation import Normalize

from models.lora_prototype_utils_v2.utils import IncrementalClassifier
from models.lora_prototype_utils_v2.loralib.layers import Linear as LoraLinear

import models.lora_prototype_utils_v2.clip_coda_reborn.clip as clip


class MultiheadAttention(torch.nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = LoraLinear(dim, dim * 3, 0., bias=qkv_bias)

        self.attn_drop = torch.nn.Dropout(attn_drop)

        self.proj = LoraLinear(dim, dim, 0.)

        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, AB: dict = None):
        N, B, C = query.shape
        query = query.transpose(0, 1)

        AB_qkv = None

        if AB is not None:
            AB_qkv = AB.get("qkv")

        qkv = self.qkv(query, AB_qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        AB_proj = None

        if AB is not None:
            AB_proj = AB.get("proj")

        x = self.proj(x, AB_proj)
        x = self.proj_drop(x)

        x = x.transpose(1, 0)
        return x, attn


class ClipVit(torch.nn.Module):

    def __init__(self, args, seq_dataset, num_classes):
        super().__init__()

        visual, self.clip_preprocess, self.visual_dtype = \
            self.create_visual(args)
        self.embed_dim = visual.class_embedding.shape[0]

        # self.visual_original = deepcopy(visual)

        self.surgery(visual)
        self.visual = visual

        self.output_dim = self.visual.output_dim

        self.mlp_ratio = 4

        self.denorm_transform = seq_dataset.get_denormalization_transform()

        self.clip_normalization = Normalize(self.clip_preprocess.transforms[-1].mean,
                                            self.clip_preprocess.transforms[-1].std)

        for p in self.visual.parameters():
            p.requires_grad = False

        self.head = IncrementalClassifier(self.output_dim, num_classes) \
            if num_classes > 0 else torch.nn.Identity()

    def create_visual(self, args):
        clip_model, clip_preprocess = clip.load(args.backbone_type, device='cpu')

        if args.force_fp32:
            clip_model = clip_model.float()

        visual = deepcopy(clip_model.visual)
        dtype = clip_model.dtype

        del clip_model
        return visual, clip_preprocess, dtype

    @torch.no_grad()
    def surgery(self, visual):

        num_blocks = len(visual.transformer.resblocks)

        for block_id in range(num_blocks):
            old_ma = visual.transformer.resblocks[block_id].attn
            old_ma_sd = old_ma.state_dict()

            new_ma = MultiheadAttention(self.embed_dim, old_ma.num_heads, True)

            new_ma.qkv.weight.zero_()
            new_ma.qkv.weight.add_(old_ma_sd['in_proj_weight'])

            new_ma.qkv.bias.zero_()
            new_ma.qkv.bias.add_(old_ma_sd['in_proj_bias'])

            new_ma.proj.weight.zero_()
            new_ma.proj.weight.add_(old_ma_sd['out_proj.weight'])

            new_ma.proj.bias.zero_()
            new_ma.proj.bias.add_(old_ma_sd['out_proj.bias'])

            del visual.transformer.resblocks[block_id].attn
            visual.transformer.resblocks[block_id].attn = new_ma

    def encode_image(self, image, AB):
        x = image.type(self.visual_dtype)
        out = self.visual(x, AB)
        # with torch.no_grad():
        #     out2 = self.visual_original(x, AB)
        return out

    def forward_features(self, x, AB={}):
        with torch.no_grad():
            x = self.denorm_transform(x)
            x = self.clip_normalization(x)
        _, clip_out = self.encode_image(x, AB)
        return clip_out.unsqueeze(1)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x[:, 0])

    def forward(self, x, AB: dict = {}):
        x = self.forward_features(x, AB)
        x = self.forward_head(x)
        return x
