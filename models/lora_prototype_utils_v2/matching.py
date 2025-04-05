import json
import os
import torch
import sys

from torch.nn.functional import normalize as torch_normalize
from kornia.augmentation import Normalize

import models.lora_prototype_utils_v2.clip_coda_reborn.clip as clip


class MatchingEngine(torch.nn.Module):

    def __init__(self, args, device, seq_dataset):

        super().__init__()

        self.device = device

        self.keys_ckpt_path = args.keys_ckpt_path
        self.use_templates = args.use_templates

        self.nucleus_top_k = 5

        if args.auto_load_keys:
            self.keys_ckpt_path = self.get_chk_name(args)

        self.prompt_templates = seq_dataset.get_prompt_templates()
        self.dataset_classes = seq_dataset.get_class_names()
        self.num_classes = seq_dataset.N_CLASSES

        if args.clip_backbone_type is not None:
            backbone_type = args.clip_backbone_type
        else:
            backbone_type = args.backbone_type if hasattr(args, 'backbone_type') else 'RN50'

        self.clip_model, self.clip_preprocess = clip.load(backbone_type, self.device)

        if args.force_fp32:
            self.clip_model = self.clip_model.float()

        self.clip_normalization = Normalize(self.clip_preprocess.transforms[-1].mean,
                                            self.clip_preprocess.transforms[-1].std).to(self.device)

        self.denorm_transform = seq_dataset.get_denormalization_transform()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.register_buffer('keys', self.init_keys())
        self.clip_dim = self.clip_model.visual.output_dim

    def get_chk_name(self, args):
        clip_backbone_type = args.clip_backbone_type if args.clip_backbone_type is not None else args.backbone_type
        try:
            key_jobnum = json.load(open(os.path.join(os.path.dirname(__file__), 'icoop_keys.json'), 'r'))[args.dataset][
                clip_backbone_type][args.distr_alignment][str(args.seed)]
        except BaseException:
            print("key missing", args.dataset, clip_backbone_type, args.seed, file=sys.stderr)
            raise ValueError

        return f"coop_keys/coop_keys_9_{key_jobnum}.pt" if 'joint' not in args.dataset \
            else f"coop_keys/coop_keys_0_{key_jobnum}.pt"

    def get_clip_dim(self):
        return self.clip_dim

    def load_keys(self):
        print(f'Loading keys from {self.keys_ckpt_path}', file=sys.stderr)
        keys = torch.load(self.keys_ckpt_path).to(self.device)
        assert self.num_classes == keys.shape[0], \
            "we guarda che stai caricando il checkpoint sbagliato pistola"
        print('Keys loaded successfully', file=sys.stderr)
        return keys.float()

    @torch.no_grad()
    def init_keys(self):

        if self.keys_ckpt_path is not None:
            return self.load_keys()

        if self.use_templates:
            all_features = []
            for t in self.prompt_templates:
                text_inputs = torch.cat([clip.tokenize(t.format(c)) for c in self.dataset_classes]).to(self.device)
                text_features = self.clip_model.encode_text(text_inputs)
                all_features.append(text_features)
            text_features = torch.stack(all_features).mean(dim=0)
        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.dataset_classes]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def top_k(self, query, start_idx, end_idx,
              train, top_k: int = None):

        sim_act_map = self.compute_maps(query, start_idx, end_idx)

        if top_k is None:
            top_k = self.nucleus_top_k

        num_elems = min(top_k, sim_act_map.size(1))

        _, top_k_indices = torch.topk(sim_act_map, k=num_elems, dim=1)

        return top_k_indices

    def compute_maps(self, act_map, start_idx, end_idx):
        B, D = act_map.shape
        mymap = act_map.unsqueeze(1).expand(B, (end_idx - start_idx), D)
        mymap = torch.einsum('bcd,cd->bc',
                             torch_normalize(mymap, dim=-1),
                             self.keys[start_idx: end_idx])

        return mymap

    def get_keys(self, start_idx, end_idx):
        return self.keys[start_idx: end_idx]

    @torch.no_grad()
    def get_query(self, x):
        x = self.denorm_transform(x)
        x = self.clip_normalization(x)
        _, clip_out = self.clip_model.encode_image(x)
        return clip_out

    def forward(self, x):
        return self.get_query(x)


class NullEngine:

    def __init__(self):
        pass

    def get_clip_dim(self):
        raise NotImplementedError

    def top_k(self, query, start_idx, end_idx,
              train, top_k: int = None):
        raise NotImplementedError

    def get_keys(self, start_idx, end_idx):
        raise NotImplementedError

    def get_query(self, x):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError
