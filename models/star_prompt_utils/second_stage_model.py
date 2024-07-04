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
        self.args = args
        self.prompt_layers = prompt_layers
        self.target_embed_len = target_embed_len
        self.target_embed_dim = target_embed_dim
        self.device = args.device
        self.num_classes = num_classes

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
            self.clip_model = self.clip_model.float() # force fp32 when used for eval
        else: # use prompt templates
            self.keys_ckpt_path = None
            print("No keys loaded. Using default CLIP version:", clip_backbone)
            self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
            self.clip_model = self.clip_model.float() # force fp32 when used for eval
            self.keys = self.load_default_prompt_templates(dataset.get_prompt_templates(), dataset.get_class_names())

        self.clip_normalization = Normalize(self.clip_preprocess.transforms[-1].mean,
                                            self.clip_preprocess.transforms[-1].std).to(self.device)
        self.denorm_transform = dataset.get_denormalization_transform()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        for l in self.prompt_layers:
            setattr(self, f'p_{l}', self.get_parameter((self.num_classes, self.target_embed_dim)))
            setattr(self, f'a_{l}', self.get_parameter((self.num_classes, self.clip_model.visual.output_dim),
                                                       type_init='orto'))

    def get_parameter(self, shape, type_init: str = 'orto'):
        param = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=self.device))
        if type_init == 'orto':
            torch.nn.init.orthogonal_(param)
        if type_init == 'gaussian':
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        return param

    @torch.no_grad()
    def load_default_prompt_templates(self, prompt_templates: List[str], dataset_classes: List[str]) -> torch.Tensor:
        all_features = []
        for t in prompt_templates:
            text_inputs = torch.cat([clip.tokenize(t.format(c)) for c in dataset_classes]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            all_features.append(text_features)
        text_features = torch.stack(all_features).mean(dim=0)
        return text_features

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

        sp = torch.einsum('bc,cd->bd', sim_act_map, p)
        return sp

    def get_prompts(self, layer_idx, clip_out, cur_classes: int, frozen_past_classes=0):

        if layer_idx in self.prompt_layers:

            pv: torch.Tensor = getattr(self, f'p_{layer_idx}')
            a: torch.Tensor = getattr(self, f'a_{layer_idx}')

            if frozen_past_classes > 0:
                with torch.no_grad():
                    clip_out_prev = self.compute_maps(clip_out, a[:frozen_past_classes].detach(), self.keys[:frozen_past_classes].detach())
                clip_out_curr = self.compute_maps(clip_out, a[frozen_past_classes:cur_classes], self.keys[frozen_past_classes:cur_classes])
                clip_out = torch.cat((clip_out_prev.detach(), clip_out_curr), dim=1)
                clip_out = self.get_masked_clip_out(clip_out)

                with torch.no_grad():
                    sp_past = self.compute_super_prompts(pv, clip_out, 0, frozen_past_classes)
                sp_curr = self.compute_super_prompts(pv, clip_out, frozen_past_classes, cur_classes)

                super_prompt = sp_past.detach() + sp_curr
            else:
                clip_out = self.compute_maps(clip_out, a[:cur_classes], self.keys[:cur_classes])
                clip_out = self.get_masked_clip_out(clip_out)

                super_prompt = self.compute_super_prompts(pv, clip_out, 0, cur_classes)

            return super_prompt, clip_out
        else:
            return None

    def compute_ortho_loss(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:

        if frozen_past_classes == 0:  # No ortho to compute between present and past
            return 0.

        ortho_loss_list = []
        weight_loss_list = []

        for layer_idx in self.prompt_layers:

            p = getattr(self, f'p_{layer_idx}')

            past_pv = p[:frozen_past_classes].detach()
            cur_pv = p[frozen_past_classes:cur_classes]

            eye_intra = torch.eye(cur_classes - frozen_past_classes).bool()

            intra_ortho_loss = (torch.matmul(cur_pv, cur_pv.T)[eye_intra] - 1).pow(2).mean()
            inter_ortho_loss = (torch.matmul(cur_pv, past_pv.T)).pow(2).mean()

            current_loss = intra_ortho_loss + inter_ortho_loss
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
        vit_model = VisionTransformer(embed_dim=768, depth=12, num_heads=12, drop_path_rate=0, num_classes=num_classes)

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
