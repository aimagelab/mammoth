import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.cgil_utils.diffusion import DiffusionCA
from models.cgil_utils.generative_replay import (FeaturesDataset, Gaussian,
                                                 MixtureOfGaussiansModel)
from models.cgil_utils.vae import VariationalAutoEncoderModel
from utils.conf import get_device

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")
try:
    import wandb
except ImportError:
    wandb = None


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Model(torch.nn.Module):
    def __init__(self, args, num_classes: int):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device = get_device()

        self.prompter = Prompter(args)

    def train(self, mode=True):
        super().train(False)
        self.prompter.train(False)

        return self

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        clip_out = self.prompter.get_query(x)

        keys = self.prompter.get_keys(train=train, image_embeds=clip_out)

        return self.prompter.get_clip_logits(clip_out, keys)

    def future_forward(self, x: torch.Tensor) -> torch.Tensor:
        clip_out = self.prompter.get_query(x)
        trained_keys = self.prompter.get_keys(train=False, image_embeds=clip_out)
        untrained_keys = self.prompter.just_text_features[len(trained_keys):]
        keys = torch.cat((trained_keys, untrained_keys), dim=0)

        return self.prompter.get_clip_logits(clip_out, keys)


class Prompter(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = get_device()

        self.seq_dataset = get_dataset(self.args)
        self.num_classes = self.seq_dataset.N_CLASSES

        self.clip_model, self.clip_preprocess = clip.load(args.clip_backbone, self.device)
        self.clip_model = self.clip_model.float()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.current_task = 0
        self.class_names = self.seq_dataset.get_class_names()
        self.setup_text_prompting()
        self.clip_logit_scale = self.clip_model.logit_scale

        embed_dim = self.clip_model.visual.output_dim

        if self.args.g_models == 'gauss':
            self.distributions = torch.nn.ModuleList([Gaussian(embed_dim) for _ in range(self.num_classes)]).to(self.device)
        elif self.args.g_models == 'mog':
            self.distributions = torch.nn.ModuleList([MixtureOfGaussiansModel(embed_dim, n_components=self.args.gr_mog_n_components,
                                                                              n_iters=self.args.gr_mog_n_iters)
                                                      for _ in range(self.num_classes)]).to(self.device)
        elif self.args.g_models == 'vae':
            self.distributions = torch.nn.ModuleList([VariationalAutoEncoderModel(input_dim=embed_dim,
                                                                                  hidden_dim=self.args.gr_vae_hidden_dim,
                                                                                  latent_dim=self.args.gr_vae_latent_dim,
                                                                                  lr=self.args.lr_vae,
                                                                                  n_iters=self.args.gr_vae_n_iters,
                                                                                  class_idx=i)
                                                      for i in range(self.num_classes)])
        elif self.args.g_models == 'diffusion':
            self.distributions = torch.nn.ModuleList([DiffusionCA(embed_dim,
                                                                  self.device,
                                                                  target="img",
                                                                  num_hidden=5,
                                                                  hidden_dim=self.args.gr_vae_latent_dim,
                                                                  n_iters=self.args.gr_vae_n_iters,
                                                                  class_idx=i) for i in range(self.num_classes)])

    def compute_ortho_loss(self) -> torch.Tensor:
        """Computes the orthogonality loss between the prompt parameters.

        Returns:
            torch.Tensor: The orthogonality loss.
        """
        offset_1, offset_2 = self.seq_dataset.get_offsets(self.current_task)

        if not self.args.train_only_current_prompts:
            coop_p = torch.cat([getattr(self, f'prompt_parameters_{i}') for i in range(0, offset_2)], dim=0)
            I = torch.eye(coop_p.shape[0], device=self.device, dtype=coop_p.dtype)
            ortho_loss_coop = (coop_p @ coop_p.t() - I).pow(2).mean()
        else:
            cur_coop_p = torch.cat([getattr(self, f'prompt_parameters_{i}') for i in range(offset_1, offset_2)], dim=0).unsqueeze(1)
            if self.current_task > 0:
                past_coop_p = torch.cat([getattr(self, f'prompt_parameters_{i}').detach() for i in range(offset_1)], dim=0)
                ortho_loss_coop = (torch.matmul(cur_coop_p.permute(1, 0, 2), past_coop_p.permute(1, 2, 0))**2).mean()

        return ortho_loss_coop

    @torch.no_grad()
    def build_features_dataset(self) -> torch.utils.data.DataLoader:
        """Builds a dataset of features and labels for the alignment task using the current distributions.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the alignment task.
        """

        labels, features = [], []

        for _ti in range(self.current_task + 1):

            prev_t_size, cur_t_size = self.seq_dataset.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):

                curr_dist = self.distributions[class_idx]
                prev_train = curr_dist.training
                curr_dist.eval()
                curr_dist = curr_dist.to(self.device)
                current_samples = curr_dist(256)
                curr_dist.train(prev_train)
                curr_dist = curr_dist.to('cpu')
                features.append(current_samples)
                labels.append(torch.ones((256)) * class_idx)

        features = torch.cat(features, dim=0).detach()
        labels = torch.cat(labels, dim=0).long()

        return torch.utils.data.DataLoader(FeaturesDataset(features, labels),
                                           batch_size=self.args.batch_size_alignment, shuffle=True, num_workers=0)

    def train_alignment_epoch(self, optim: torch.optim.Optimizer) -> None:
        """Trains the alignment task for one epoch.

        Args:
            optim (torch.optim.Optimizer): The optimizer to use for training.
        """
        offset_1, offset_2 = self.seq_dataset.get_offsets(self.current_task)

        data_loader = self.build_features_dataset()

        for i, (image_features, labels) in enumerate(data_loader):
            if self.args.debug_mode and i > 3:
                break
            optim.zero_grad()

            image_features, labels = image_features.to(self.device, dtype=self.clip_model.dtype), labels.to(self.device)
            image_features = F.normalize(image_features, dim=-1)
            if self.args.train_only_current_prompts and self.current_task > 0:
                with torch.no_grad():
                    past_keys = self.compute_keys(0, offset_1, image_features)
                cur_keys = self.compute_keys(offset_1, offset_2, image_features)
                text_features = torch.cat((past_keys.detach(), cur_keys), dim=0)
            else:
                text_features = self.compute_keys(0, offset_2, image_features)

            text_features = F.normalize(text_features, dim=-1)

            if self.args.generated_context and self.args.cocoop:
                text_features = text_features.reshape(image_features.shape[0], -1, text_features.shape[-1])
                image_features = image_features.unsqueeze(1)
                clip_logits = (text_features * image_features).sum(-1)
            else:
                clip_logits = torch.einsum('bd,cd->bc', image_features, text_features)
            clip_logits = clip_logits * self.clip_logit_scale.exp()
            loss = F.cross_entropy(clip_logits, labels)

            wandb_log = {'alignment_loss_ce': loss.item()}

            if self.args.align_with_ortholoss and not self.args.generated_context:
                ortho_loss = self.compute_ortho_loss()
                loss += self.args.lambda_ortho_first_stage * ortho_loss
                wandb_log['alignment_loss_ortho'] = ortho_loss.item()

            wandb_log['alignment_loss'] = loss.item()
            if wandb.run:
                wandb.log(wandb_log)

            loss.backward()
            optim.step()

    def align(self) -> None:
        """Trains the alignment task for the current task."""
        offset_1, offset_2 = self.seq_dataset.get_offsets(self.current_task)
        if not self.args.train_only_current_prompts:
            offset_1 = 0

        if self.args.generated_context:
            parameters = self.context_generator.parameters()
        elif self.args.combo_context:
            parameters = [getattr(self, f'prompt_parameters_{i}') for i in range(offset_1, offset_2)]
            parameters += self.context_generator.parameters()
        elif self.args.general_context == 0:
            parameters = [getattr(self, f'prompt_parameters_{i}') for i in range(offset_1, offset_2)]
        else:
            parameters = [self.prompt_parameters]

        if self.args.optim_alignment == 'sgd':
            optim = torch.optim.SGD(lr=self.args.learning_rate_alignment, params=parameters, momentum=0.0, weight_decay=0.0)
        elif self.args.optim_alignment == 'adam':
            optim = torch.optim.Adam(lr=self.args.learning_rate_alignment, params=parameters, weight_decay=0.0)
        elif self.args.optim_alignment == 'adamw':
            optim = torch.optim.AdamW(lr=self.args.learning_rate_alignment, params=parameters, weight_decay=self.args.optim_alignment_wd)
        else:
            raise ValueError(f'Invalid optimizer: {self.args.optim_alignment}')

        for _ in trange(self.args.num_epochs_alignment, desc=f'Alignment Task {self.current_task}', unit='epoch'):
            self.train_alignment_epoch(optim)

    @torch.no_grad()
    def update_statistics(self, dataset: ContinualDataset) -> None:
        """Fit the distributions to the features of the current task.

        Args:
            dataset (ContinualDataset): The dataset to use for updating the statistics.
        """
        offset_1, offset_2 = dataset.get_offsets(self.current_task)

        features_dict = {i: [] for i in range(offset_1, offset_2)}

        was_training = self.training
        self.eval()

        Path('./cache').mkdir(parents=True, exist_ok=True)
        clip_backbone = self.args.clip_backbone.replace('/', '_')
        cache_path = Path(f'./cache/{dataset.NAME}_{self.current_task}_{clip_backbone}_{dataset.args.permute_classes}_features.pt')
        if dataset.args.seed is not None:
            cache_path = Path(f'./cache/{dataset.NAME}_{self.current_task}_seed_{dataset.args.seed}_{clip_backbone}_{dataset.args.permute_classes}_features.pt')

        if cache_path.exists():
            features_dict = torch.load(cache_path, weights_only=True)
            print(f'Loaded cached features from {cache_path}')
        else:
            with tqdm(total=len(dataset.train_loader), desc='Updating statistics for first stage Generative Replay') as pbar:
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3 and min([len(v) for v in features_dict.values()]) > self.args.gr_mog_n_components:
                        break
                    inputs, labels = data[0], data[1]
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()

                    clip_query = self.get_query(inputs)

                    for class_idx in labels.unique():
                        features_dict[int(class_idx)].append(clip_query[labels == class_idx])

                    pbar.update(1)
            if not self.args.debug_mode:
                torch.save(features_dict, cache_path)

        for class_idx in range(offset_1, offset_2):
            features_class_idx = torch.cat(features_dict[class_idx], dim=0)
            self.distributions[class_idx].fit(features_class_idx.to(self.device))

        if was_training:
            self.train()

    def compute_keys(self, start: int, end: int, image_embeds=None):
        prefix = self.token_prefix[start:end]
        suffix = self.token_suffix[start:end]
        tokenized_prompts = self.tokenized_prompts[start:end]
        if self.args.generated_context:
            if self.args.cocoop:
                ctx = self.context_generator(image_embeds).unsqueeze(1).unsqueeze(1).expand(-1, end - start, -1, -1).reshape(-1, 1, image_embeds.shape[-1])
                prefix = prefix.unsqueeze(0).expand(image_embeds.shape[0], -1, -1, -1).reshape(-1, prefix.shape[1], prefix.shape[2])
                suffix = suffix.unsqueeze(0).expand(image_embeds.shape[0], -1, -1, -1).reshape(-1, suffix.shape[1], suffix.shape[2])
                tokenized_prompts = tokenized_prompts.unsqueeze(0).expand(image_embeds.shape[0], -1, -1).reshape(-1, self.tokenized_prompts.shape[-1])
            else:
                ctx = self.context_generator(self.just_text_features[start:end]).unsqueeze(1)
        elif self.args.combo_context:
            ctx = torch.cat([getattr(self, f'prompt_parameters_{i}') for i in range(start, end)], dim=0)
            ctx = torch.cat([ctx, self.context_generator(self.just_text_features[start:end]).unsqueeze(1)], dim=1)
        elif self.args.general_context == 0:
            ctx = torch.cat([getattr(self, f'prompt_parameters_{i}') for i in range(start, end)], dim=0)
        else:
            ctx = self.prompt_parameters.unsqueeze(0).expand(end - start, -1, -1)
        prompts = torch.cat((prefix, ctx, suffix), dim=1)
        keys = self.text_encoder(prompts.to(self.clip_model.dtype), tokenized_prompts)
        keys = F.normalize(keys, dim=-1)
        return keys

    def get_keys(self, train: bool = True, image_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        task_id = self.current_task if train else self.current_task - 1
        offset_1, offset_2 = self.seq_dataset.get_offsets(task_id)
        if train and self.current_task > 0:
            with torch.no_grad():
                past_keys = self.compute_keys(0, offset_1, image_embeds)
            cur_keys = self.compute_keys(offset_1, offset_2, image_embeds)
            keys = torch.cat((past_keys.detach(), cur_keys), dim=0)
        else:
            keys = self.compute_keys(0, offset_2, image_embeds)
        return keys

    def setup_text_prompting(self) -> None:
        """Setup the text prompting for the model."""
        self.text_encoder = TextEncoder(self.clip_model)

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.class_names]).to(self.device)
        with torch.no_grad():
            self.just_text_tokens = self.clip_model.token_embedding(text_inputs)
        self.just_text_features = self.clip_model.encode_text(text_inputs)

        if self.args.generated_context or self.args.combo_context:
            in_dim = self.just_text_features.shape[-1]
            out_dim = self.clip_model.token_embedding.weight.shape[1]
            self.context_generator = torch.nn.Sequential(
                torch.nn.Linear(in_dim, in_dim),
                torch.nn.BatchNorm1d(in_dim),
                torch.nn.SELU(True),
                torch.nn.Linear(in_dim, in_dim),
                torch.nn.BatchNorm1d(in_dim),
                torch.nn.SELU(True),
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.BatchNorm1d(out_dim),
                torch.nn.SELU(True),
                torch.nn.Linear(out_dim, out_dim),
            ).to(self.device)

        n_ctx = max(self.args.n_context, 1)
        if self.args.combo_context:
            n_ctx += 1
        prefix = " ".join(["X"] * n_ctx)
        text_prompts = [prefix + " " + name + "." for name in self.class_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts], dim=0).to(self.device)
        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        if not self.args.generated_context:
            if not self.args.general_context:
                for i in range(self.num_classes):
                    prompt_parameter = torch.empty(1, self.args.n_context, self.clip_model.token_embedding.weight.shape[1], device=self.device, dtype=torch.float32)
                    torch.nn.init.normal_(prompt_parameter, std=0.02)
                    self.register_parameter(f"prompt_parameters_{i}", torch.nn.Parameter(prompt_parameter))
            else:
                prompt_parameter = torch.empty(self.args.n_context, self.clip_model.token_embedding.weight.shape[1], device=self.device, dtype=torch.float32)
                torch.nn.init.normal_(prompt_parameter, std=0.02)
                self.prompt_parameters = torch.nn.Parameter(prompt_parameter)

    @torch.no_grad()
    def get_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(x)

    def get_clip_logits(self, clip_out: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(clip_out, dim=-1)
        if self.args.generated_context and self.args.cocoop:
            keys = keys.reshape(image_features.shape[0], -1, keys.shape[-1])
            image_features = image_features.unsqueeze(1)
            clip_logits = (keys * image_features).sum(-1)
        else:
            clip_logits = torch.einsum('bd,cd->bc', image_features, keys)
        clip_logits = clip_logits * self.clip_logit_scale.exp()
        return clip_logits
