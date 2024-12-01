from argparse import Namespace
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
                 target_embed_dim: int, prompt_layers: List[int],
                 clip_model: clip.model.CLIP = None, clip_preprocess=None,
                 device='cpu'):
        super().__init__()
        assert args.prompt_mode in ['residual', 'concat'], 'This prompter supports only STAR-Prompt residual-style prompts (`residual`) or Prefix tuning-style prompts (`concat`).'
        self.args = args
        self.prompt_layers = prompt_layers
        self.target_embed_len = target_embed_len
        self.target_embed_dim = target_embed_dim
        self.device = device
        self.num_classes = num_classes
        self.prompt_mode = args.prompt_mode

        if clip_model is not None:
            assert clip_preprocess is not None, 'Preprocess must be provided if the model is provided'

        print("Loading CLIP visual encoder and the pre-computed text features...")
        clip_backbone = 'ViT-L/14' if not hasattr(args, 'clip_backbone') else args.clip_backbone
        if hasattr(args, 'keys_ckpt_path') and args.keys_ckpt_path is not None:
            if args.keys_ckpt_path.endswith('.json'):
                try:
                    key_jobnum = json.load(open(args.keys_ckpt_path, 'r'))[args.dataset][str(args.seed)]
                except BaseException:
                    print("key missing", args.dataset, args.seed, file=sys.stderr)
                    raise ValueError

                t = dataset.N_TASKS - 1
                self.keys_ckpt_path = f"coop_keys/coop_keys_{t}_{key_jobnum}.pt"
            elif args.keys_ckpt_path.endswith('.pt'):
                self.keys_ckpt_path = args.keys_ckpt_path
            else:
                t = dataset.N_TASKS - 1
                self.keys_ckpt_path = f"coop_keys/coop_keys_{t}_{args.keys_ckpt_path}.pt"

            if not os.path.exists(self.keys_ckpt_path):
                raise ValueError(f'Keys checkpoint `{self.keys_ckpt_path}` does not exist')

            self.keys, first_stage_args = self.load_keys()
            if first_stage_args is not None:
                print("Keys loaded. Loading CLIP version:", first_stage_args.clip_backbone)
                clip_backbone = first_stage_args.clip_backbone
            if clip_model is None:
                self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
                self.clip_model = self.clip_model.float()  # force fp32 when used for eval
            else:
                self.clip_model = clip_model
                self.clip_preprocess = clip_preprocess
        else:  # use prompt templates
            self.keys_ckpt_path = None
            print("No keys loaded. Using default CLIP version:", clip_backbone)
            if clip_model is None:
                self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
                self.clip_model = self.clip_model.float()  # force fp32 when used for eval
            else:
                self.clip_model = clip_model
                self.clip_preprocess = clip_preprocess
            self.keys = self.load_default_prompt_templates(dataset.get_prompt_templates(), dataset.get_class_names())

        self.clip_normalization = Normalize(self.clip_preprocess.transforms[-1].mean,
                                            self.clip_preprocess.transforms[-1].std).to(self.device)
        self.denorm_transform = dataset.get_denormalization_transform()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        for l in self.prompt_layers:
            if args.prompt_mode == 'residual':
                # NOTE: this initialization follows that of CODA-Prompt.
                # We originally initialize a prompt for key, query, and value of the MHA layer.
                tmp = self.get_parameter((self.num_classes, 3, self.target_embed_dim))
                # We only use value at the end, so we keep only a single tensor.
                tmp.data = tmp.data[:, 0]
                # HOWEVER: Since the orthogonal_ of pytorch flattens the tensor, the value prompt is not orthogonal anymore.
                # orthogonal_ made (C, 3, D) -> (C, 3*D) -> orthogonal -> (C, 3, D), thus each 3*D is orthogonal, but not each D.
                # This is intended and maked the orthogonalization loss being optimized at the beginning.
                setattr(self, f'p_{l}', tmp)
            else:
                setattr(self, f'p_concat_{l}', self.get_parameter((self.num_classes, 2 * self.args.prefix_tuning_prompt_len,
                                                                   self.target_embed_dim)))

            setattr(self, f'a_{l}', self.get_parameter((self.num_classes, self.clip_model.visual.output_dim)))

    def set_keys(self, keys: torch.Tensor, start_class: int, end_class: int):
        """
        Set the keys for the classes in the range `[start_class, end_class)`.
        """
        assert end_class - start_class == keys.shape[0], 'Number of classes in the keys tensor does not match the range'

        self.keys[start_class:end_class] = keys

    def get_parameter(self, shape, type_init: str = 'orto') -> torch.nn.Parameter:
        """
        Create and initialize a parameter tensor. Code courtesy from CODA-Prompt.
        """
        param = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=self.device))
        if type_init == 'orto':
            torch.nn.init.orthogonal_(param)
        if type_init == 'gaussian':
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        return param

    @torch.no_grad()
    def load_default_prompt_templates(self, templates: List[str], dataset_classes: List[str]) -> torch.Tensor:
        """
        Pre-computes the CLIP's text-encoder features if the keys are not loaded from a checkpoint.
        """
        if hasattr(self.args, 'statc_keys_use_templates') and self.args.statc_keys_use_templates:
            all_features = []
            for t in templates:
                text_inputs = torch.cat([clip.tokenize(t.format(c)) for c in dataset_classes]).to(self.device)
                text_features = self.clip_model.encode_text(text_inputs)
                all_features.append(text_features)
            text_features = torch.stack(all_features).mean(dim=0)
        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset_classes]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    @torch.no_grad()
    def load_keys(self):
        """
        Load the keys from a `first_stage_starprompt` checkpoint file (run with `--save_first_stage_keys=1`).
        The checkpoint file can be either:
        - A path to a checkpoint file (.pt) containing ONLY THE FIRST STAGE KEYS.
        The number of classes and the dataset must match the current run, but we cannot check if the seed was the same.
        - A path to the checkpoint made by `first_stage_starprompt` or the job-id (`conf_jobnum`) of the `first_stage_starprompt` run that made the keys.
        Checks will prevent loading keys with a different order of the classes or dataset.
        - A JSON file containing the job-id (`conf_jobnum`) of the `first_stage_starprompt` run that made the keys.
        The JSON is expected to contain an entry for each dataset and seed: `{dataset: {seed: job-id}}`.

        Returns:
            The keys tensor
            The arguments used in the first stage
        """
        print(f'Loading keys from {self.keys_ckpt_path}', file=sys.stderr)
        st = torch.load(self.keys_ckpt_path, weights_only=True)
        if isinstance(st, dict):
            keys = st['keys'].to(self.device)
            self.old_args = Namespace(**st['args'])
            assert self.num_classes == keys.shape[0]
            assert self.args.dataset == self.old_args.dataset
            assert self.args.permute_classes == self.old_args.permute_classes
            if self.args.permute_classes:
                assert self.args.seed == self.old_args.seed
        else:
            keys = st.to(self.device)
            self.old_args = None
            assert self.num_classes == keys.shape[0]
        print('Keys loaded successfully', file=sys.stderr)
        return keys.float(), self.old_args

    @torch.no_grad()
    def get_query(self, x, disable_renorm=True):
        """
        Compute the CLIP features for the input image `x`.

        Args:
            x: the input image tensor
            disable_renorm: if False, the final normalization applied to `x` will be swapped with the CLIP's one.
        """
        if not disable_renorm:
            x = self.denorm_transform(x)
            x = self.clip_normalization(x)
        clip_out = self.clip_model.encode_image(x)
        return clip_out

    def compute_maps(self, clip_query: torch.Tensor, modulation_coeffs: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute the CLIP output given the `clip_query` and the `keys`. The queries are modulated by the `modulation_coeffs`.
        """
        filter_values = torch.softmax(modulation_coeffs, dim=-1)

        clip_query = clip_query.unsqueeze(1).expand(clip_query.shape[0], modulation_coeffs.shape[0], clip_query.shape[-1])
        clip_out_a = clip_query * filter_values[None, :, :]
        clip_out_a_norm = torch.nn.functional.normalize(clip_out_a, dim=-1)

        clip_query = torch.einsum('bcd,cd->bc', clip_out_a_norm, keys) * 5

        return clip_query

    def get_masked_clip_out(self, sim_act_map):
        """
        We only need the output of the CLIP model for the most similar class, so we mask the rest.
        """
        with torch.no_grad():
            mask = torch.ones_like(sim_act_map, dtype=torch.bool)
            mask.scatter_(1, sim_act_map.argmax(dim=1, keepdim=True), False)
            sim_act_map[mask] = 0.0

        return sim_act_map

    def compute_super_prompts(self, class_prompts: torch.Tensor, masked_clip_out: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Compute the actual super-prompt by merging the individual prompts for the classes in the range `[start_idx, end_idx)`.
        The merge is made according to the similarity map `sim_act_map` and scaled by it if `enable_confidence_modulation` is set.

        Args:
            class_prompts: the prompt parameters for each class in the range `[start_idx, end_idx)`
            masked_clip_out: the masked CLIP output for the classes in the range `[start_idx, end_idx)`,
containing the similarity value for the most similar class for each image.
            start_idx: the start index of the classes to consider
            end_idx: the end index of the classes to consider

        Returns:
            The super-prompt for the classes in the range `[start_idx, end_idx)`.
        """
        masked_clip_out = masked_clip_out[:, start_idx:end_idx]
        class_prompts = class_prompts[start_idx:end_idx]

        if not self.args.enable_confidence_modulation:
            masked_clip_out = (masked_clip_out != 0).float()  # make it binary if not using confidence modulation

        if self.args.prompt_mode == 'residual':
            sp = torch.einsum('bc,cd->bd', masked_clip_out, class_prompts)
        else:
            sp = torch.einsum('bc,cmd->bmd', masked_clip_out, class_prompts)
        return sp

    def get_prompts(self, layer_idx: int, clip_query: torch.Tensor, cur_classes: int, frozen_past_classes=0):
        """
        Compute the prompts for the `layer_idx`-th layer for `cur_classes` classes.
        The prompts until `frozen_past_classes` are detached to prevent gradients from flowing back.
        By default, all the layers require prompting. This can be changed by adjusting the `prompt_layers` attribute.

        Returns:
            The computed prompt, if the layer requires prompting. Else, returns None.
        """

        if layer_idx in self.prompt_layers:

            a: torch.Tensor = getattr(self, f'a_{layer_idx}')
            if self.prompt_mode == 'residual':
                pv: torch.Tensor = getattr(self, f'p_{layer_idx}')
            else:
                p_concat: torch.Tensor = getattr(self, f'p_concat_{layer_idx}')
                p_concat_k, p_concat_v = torch.split(p_concat, self.args.prefix_tuning_prompt_len, dim=1)

            if frozen_past_classes > 0:
                with torch.no_grad():  # detach the past prompts to prevent gradients from flowing back
                    clip_out_prev = self.compute_maps(clip_query, a[:frozen_past_classes].detach(), self.keys[:frozen_past_classes].detach())
                clip_out_curr = self.compute_maps(clip_query, a[frozen_past_classes:cur_classes], self.keys[frozen_past_classes:cur_classes])
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
                clip_out = self.compute_maps(clip_query, a[:cur_classes], self.keys[:cur_classes])
                clip_out = self.get_masked_clip_out(clip_out)

                if self.prompt_mode == 'residual':
                    super_prompt = self.compute_super_prompts(pv, clip_out, 0, cur_classes)
                else:
                    sp_concat_k = self.compute_super_prompts(p_concat_k, clip_out, 0, cur_classes).squeeze(2)
                    sp_concat_v = self.compute_super_prompts(p_concat_v, clip_out, 0, cur_classes).squeeze(2)
                    super_prompt = (sp_concat_k, sp_concat_v)

            return super_prompt
        else:
            return None

    def compute_ortho_loss(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:
        """
        Compute the orthogonality loss for the prompts of the layers in `prompt_layers`.
        The loss is computed in two parts:
        - The intra-orthogonality loss between the prompts of the current classes (between `frozen_past_classes` and `cur_classes`).
        - The inter-orthogonality loss between the prompts of the past classes (before `frozen_past_classes`).
        If `frozen_past_classes` is 0, the loss is skipped and not computed.

        The argument `ortho_split_val` is used to manage the orthogonality loss computation between the layers.
        The layers before `ortho_split_val` will have a weight of 0, thus the loss will not have any effect on them.
        """

        if frozen_past_classes == 0:  # No ortho to compute between present and past
            return 0.

        ortho_loss_list = []
        weight_loss_list = []

        def _compute_loss(p: torch.Tensor, frozen_past_classes: int, cur_classes: int) -> torch.Tensor:
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

    def __init__(self, args, backbone: nn.Module, dataset: ContinualDataset, num_classes, device='cpu',
                 clip_model: clip.model.CLIP = None, clip_preprocess=None):
        super().__init__()

        assert 'resnet' not in str(type(backbone)).lower(), "ResNet not supported"

        self.args = args
        self.num_classes = num_classes
        self.device = device

        # get feature encoder
        vit_model = VisionTransformer(embed_dim=768,
                                      depth=12,
                                      num_heads=12,
                                      drop_path_rate=0,
                                      num_classes=num_classes,
                                      prompt_mode=args.prompt_mode).to(device)

        print("Loading the Vision Transformer backbone...")
        load_dict = backbone.state_dict()
        for k in list(load_dict.keys()):
            if 'head' in k:
                del load_dict[k]
        missing, unexpected = vit_model.load_state_dict(load_dict, strict=False)
        assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        self.vit = vit_model

        self.prompt_layers = list(range(len(self.vit.blocks)))

        print("Initializing the prompter and prompt parameters...")
        self.prompter = Prompter(args,
                                 dataset,
                                 num_classes=num_classes,
                                 target_embed_len=self.vit.patch_embed.num_patches,
                                 target_embed_dim=self.vit.embed_dim,
                                 prompt_layers=self.prompt_layers,
                                 clip_model=clip_model,
                                 clip_preprocess=clip_preprocess,
                                 device=device)

        # freeze the backbone
        for n, p in self.vit.named_parameters():
            if n != 'head.weight' and n != 'head.bias':
                p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        self.prompter.train(False)
        self.vit.train(mode)

        return self

    def forward(self, x: torch.Tensor, cur_classes: int, frozen_past_classes=0, query_x=None, return_features=False, return_query=False) -> torch.Tensor:
        """
        Compute the forward of the second-stage of STAR-Prompt.
        Classes from `frozen_past_classes` to `cur_classes` will have a gradient, while all those before `frozen_past_classes` will be detached.

        If `query_x` is provided, it will be used as the query for the CLIP's visual encoder.
        Otherwise, the input image `x` will be used as the query. Note that the CLIP's pre-processing will applied to `x` in this case.

        Args:
            x: the input image tensor
            cur_classes: the number of classes up to the current task
            frozen_past_classes: the number of classes from the past tasks that will be frozen
            query_x: (optional) the query tensor for the CLIP's visual encoder
            return_features: if True, the features from the Vision Transformer will be returned instead of the classification output
            return_query: if True, the query tensor will be returned with the output
        """
        enable_renorm = query_x is None
        query_x = x if query_x is None else query_x
        clip_query = self.prompter.get_query(query_x, disable_renorm=not enable_renorm)
        features = self.vit.forward_features(x, first_stage_query=clip_query, prompter=self.prompter, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes)
        if return_features:
            return features

        out = self.vit.forward_head(features)
        if return_query:
            return out, clip_query
        return out
