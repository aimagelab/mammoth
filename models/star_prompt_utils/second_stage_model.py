import os
import sys
import json
import torch
import torch.nn as nn
from typing import List
from kornia.augmentation import Normalize

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")

from datasets.utils.continual_dataset import ContinualDataset
from models.star_prompt_utils.vision_transformer import VisionTransformer


class Prompter(torch.nn.Module):
    keys: torch.Tensor

    def __init__(self, args, dataset: ContinualDataset,
                 num_classes: int, target_embed_len: int,
                 target_embed_dim: int, prompt_layers: List[int]):
        super().__init__()
        assert args.prompt_mode in ['residual', 'concat'], 'This prompter supports only STAR-Prompt residual-style prompts (`residual`) or Prefix tuning-style prompts (`concat`).'
        self.args = args
        self.prompt_layers = prompt_layers
        self.target_embed_len = target_embed_len
        self.target_embed_dim = target_embed_dim
        self.device = args.device
        self.num_classes = num_classes
        self.prompt_mode = args.prompt_mode

        clip_backbone = 'ViT-L/14'
        if args.keys_ckpt_path is not None:
            if args.keys_ckpt_path.endswith('.json'):
                try:
                    key_jobnum = json.load(open(os.path.join(os.path.dirname(__file__), 'first_stage_keys.json'), 'r'))[args.dataset][str(args.seed)]
                except BaseException:
                    print("key missing", args.dataset, args.seed, file=sys.stderr)
                    raise ValueError

                t = dataset.N_TASKS - 1
                self.keys_ckpt_path = f"coop_keys/coop_keys_{t}_{key_jobnum}.pt"
            else:
                self.keys_ckpt_path = args.keys_ckpt_path

            if not os.path.exists(self.keys_ckpt_path):
                raise ValueError(f'Keys checkpoint `{self.keys_ckpt_path}` does not exist')

            self.keys, first_stage_args = self.load_keys()
            print("Keys loaded. Loading CLIP version:", first_stage_args.clip_backbone)
            clip_backbone = first_stage_args.clip_backbone
            self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
            self.clip_model = self.clip_model.float()  # force fp32 when used for eval
        else:  # use prompt templates
            self.keys_ckpt_path = None
            print("No keys loaded. Using default CLIP version:", clip_backbone)
            self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
            self.clip_model = self.clip_model.float()  # force fp32 when used for eval
            self.keys = self.load_default_prompt_templates(dataset.get_class_names())

        self.clip_normalization = Normalize(self.clip_preprocess.transforms[-1].mean,
                                            self.clip_preprocess.transforms[-1].std).to(self.device)
        self.denorm_transform = dataset.get_denormalization_transform()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        for l in self.prompt_layers:
            if args.prompt_mode == 'residual':
                setattr(self, f'p_{l}', self.get_parameter((self.num_classes, self.target_embed_dim)))
            else:
                setattr(self, f'p_concat_{l}', self.get_parameter((self.num_classes, 2 * self.args.prefix_tuning_prompt_len,
                                                                   self.target_embed_dim)))

            setattr(self, f'a_{l}', self.get_parameter((self.num_classes, self.clip_model.visual.output_dim)))

    def get_parameter(self, shape, type_init: str = 'orto'):
        param = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=self.device))
        if type_init == 'orto':
            torch.nn.init.orthogonal_(param)
        if type_init == 'gaussian':
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        return param

    @torch.no_grad()
    def load_default_prompt_templates(self, dataset_classes: List[str]) -> torch.Tensor:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset_classes]).to(self.device)
        text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    @torch.no_grad()
    def load_keys(self):
        print(f'Loading keys from {self.keys_ckpt_path}', file=sys.stderr)
        st = torch.load(self.keys_ckpt_path)
        keys = st['keys'].to(self.device)
        old_args = st['args']
        assert self.num_classes == keys.shape[0]
        print('Keys loaded successfully', file=sys.stderr)
        return keys.float(), old_args

    @torch.no_grad()
    def get_query(self, x, disable_renorm=False):
        if not disable_renorm:
            x = self.denorm_transform(x)
            x = self.clip_normalization(x)
        clip_out = self.clip_model.encode_image(x)
        return clip_out

    def compute_maps(self, clip_out, a, k):
        filter_values = torch.softmax(a, dim=-1)

        clip_out = clip_out.unsqueeze(1).expand(clip_out.shape[0], a.shape[0], clip_out.shape[-1])
        clip_out_a = clip_out * filter_values[None, :, :]
        clip_out_a_norm = torch.nn.functional.normalize(clip_out_a, dim=-1)

        clip_out = torch.einsum('bcd,cd->bc', clip_out_a_norm, k) * 5

        return clip_out

    def get_masked_clip_out(self, sim_act_map):
        with torch.no_grad():
            mask = torch.ones_like(sim_act_map, dtype=torch.bool)
            mask.scatter_(1, sim_act_map.argmax(dim=1, keepdim=True), False)
            sim_act_map[mask] = 0.0

        return sim_act_map

    def compute_super_prompts(self, p, sim_act_map, start_idx, end_idx):
        sim_act_map = sim_act_map[:, start_idx:end_idx]
        p = p[start_idx:end_idx]

        if self.args.prompt_mode == 'residual':
            sp = torch.einsum('bc,cd->bd', sim_act_map, p)
        else:
            sp = torch.einsum('bc,cmd->bmd', sim_act_map, p)
        return sp

    def get_prompts(self, layer_idx, clip_out, cur_classes: int, frozen_past_classes=0):

        if layer_idx in self.prompt_layers:

            a: torch.Tensor = getattr(self, f'a_{layer_idx}')
            if self.prompt_mode == 'residual':
                pv: torch.Tensor = getattr(self, f'p_{layer_idx}')
            else:
                clip_out = clip_out[:, :1]  # only use class token for prefix tuning
                p_concat: torch.Tensor = getattr(self, f'p_concat_{layer_idx}')
                p_concat_k, p_concat_v = torch.split(p_concat, self.args.prefix_tuning_prompt_len, dim=1)

            if frozen_past_classes > 0:
                with torch.no_grad():
                    clip_out_prev = self.compute_maps(clip_out, a[:frozen_past_classes].detach(), self.keys[:frozen_past_classes].detach())
                clip_out_curr = self.compute_maps(clip_out, a[frozen_past_classes:cur_classes], self.keys[frozen_past_classes:cur_classes])
                clip_out = torch.cat((clip_out_prev.detach(), clip_out_curr), dim=1)
                clip_out = self.get_masked_clip_out(clip_out)

                with torch.no_grad():
                    if self.prompt_mode == 'residual':
                        sp_past = self.compute_super_prompts(pv, clip_out, 0, frozen_past_classes)
                    else:
                        sp_concat_k_past = self.compute_super_prompts(p_concat_k, clip_out, 0, frozen_past_classes).squeeze(2)
                        sp_concat_v_past = self.compute_super_prompts(p_concat_v, clip_out, 0, frozen_past_classes).squeeze(2)

                if self.prompt_mode == 'residual':
                    sp_curr = self.compute_super_prompts(pv, clip_out, frozen_past_classes, cur_classes)
                    super_prompt = sp_past.detach() + sp_curr
                else:
                    sp_concat_k_curr = self.compute_super_prompts(p_concat_k, clip_out, frozen_past_classes, cur_classes).squeeze(2)
                    sp_concat_v_curr = self.compute_super_prompts(p_concat_v, clip_out, frozen_past_classes, cur_classes).squeeze(2)
                    super_prompt = (sp_concat_k_past.detach() + sp_concat_k_curr, sp_concat_v_past.detach() + sp_concat_v_curr)
            else:
                clip_out = self.compute_maps(clip_out, a[:cur_classes], self.keys[:cur_classes])
                clip_out = self.get_masked_clip_out(clip_out)

                if self.prompt_mode == 'residual':
                    super_prompt = self.compute_super_prompts(pv, clip_out, 0, cur_classes)
                else:
                    sp_concat_k = self.compute_super_prompts(p_concat_k, clip_out, 0, cur_classes).squeeze(2)
                    sp_concat_v = self.compute_super_prompts(p_concat_v, clip_out, 0, cur_classes).squeeze(2)
                    super_prompt = (sp_concat_k, sp_concat_v)

            return super_prompt, clip_out
        else:
            return None

    def compute_ortho_loss(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:

        if frozen_past_classes == 0:  # No ortho to compute between present and past
            return 0.

        ortho_loss_list = []
        weight_loss_list = []

        def _compute_loss(p, frozen_past_classes, cur_classes):
            past_pv = p[:frozen_past_classes].detach()
            cur_pv = p[frozen_past_classes:cur_classes]

            eye_intra = torch.eye(cur_classes - frozen_past_classes).bool()

            intra_ortho_loss = (torch.matmul(cur_pv, cur_pv.T)[eye_intra] - 1).pow(2).mean()
            inter_ortho_loss = (torch.matmul(cur_pv, past_pv.T)).pow(2).mean()
            return intra_ortho_loss + inter_ortho_loss

        for layer_idx in self.prompt_layers:

            if self.prompt_mode == 'residual':
                p = getattr(self, f'p_{layer_idx}')
                current_loss = _compute_loss(p, frozen_past_classes, cur_classes)
            else:
                p_concat = getattr(self, f'p_concat_{layer_idx}')
                p_concat_k, p_concat_v = torch.split(p_concat, self.args.prefix_tuning_prompt_len, dim=1)

                p_concat_k = p_concat_k.view(p_concat_k.shape[0], -1)
                p_concat_v = p_concat_v.view(p_concat_v.shape[0], -1)

                current_loss_k = _compute_loss(p_concat_k, frozen_past_classes, cur_classes)
                current_loss_v = _compute_loss(p_concat_v, frozen_past_classes, cur_classes)

                current_loss = current_loss_k + current_loss_v

            current_weight = 1.
            if layer_idx < self.args.ortho_split_val:
                current_weight = 0.

            current_loss = current_weight * current_loss

            weight_loss_list.append(current_weight)
            ortho_loss_list.append(current_loss)

        total_ortho_loss = sum(ortho_loss_list) / sum(weight_loss_list)

        return total_ortho_loss


class Model(nn.Module):
    prompter: Prompter

    def __init__(self, args, backbone: nn.Module, dataset: ContinualDataset, num_classes):
        super().__init__()

        assert 'resnet' not in str(type(backbone)).lower(), "ResNet not supported"

        self.args = args
        self.num_classes = num_classes
        self.device = backbone.device

        # get feature encoder
        vit_model = VisionTransformer(embed_dim=768,
                                      depth=12,
                                      num_heads=12,
                                      drop_path_rate=0,
                                      num_classes=num_classes,
                                      prompt_mode=args.prompt_mode)

        # load pretrained weights
        load_dict = backbone.state_dict()
        for k in list(load_dict.keys()):
            if 'head' in k:
                del load_dict[k]
        missing, unexpected = vit_model.load_state_dict(load_dict, strict=False)
        assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        # classifier
        self.last = nn.Linear(768, num_classes)

        self.vit = vit_model

        self.prompt_layers = list(range(len(self.vit.blocks)))

        self.prompter = Prompter(args,
                                 dataset,
                                 num_classes=num_classes,
                                 target_embed_len=self.vit.patch_embed.num_patches,
                                 target_embed_dim=self.vit.embed_dim,
                                 prompt_layers=self.prompt_layers)

        for n, p in self.vit.named_parameters():
            if n != 'head.weight' and n != 'head.bias':
                p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        self.prompter.train(False)
        self.vit.train(mode)

        return self

    def forward(self, x, cur_classes: int, frozen_past_classes=0, return_features=False):
        clip_out = self.prompter.get_query(x, disable_renorm=False)
        features = self.vit.forward_features(x, first_stage_query=clip_out, prompter=self.prompter, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes)
        if return_features:
            return features

        out = self.vit.forward_head(features)
        return out
