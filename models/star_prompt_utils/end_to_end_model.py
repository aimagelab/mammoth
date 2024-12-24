from copy import deepcopy
from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import TensorDataset

from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
try:
    from clip.model import convert_weights
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git (requires also `huggingface-hub`)")


from utils.conf import create_seeded_dataloader
from datasets.utils.continual_dataset import ContinualDataset
from models.star_prompt_utils.first_stage_model import Model as FirstStageModel
from models.star_prompt_utils.second_stage_model import Model as SecondStageModel
from models.star_prompt_utils.generative_replay import Gaussian, MixtureOfGaussiansModel


class STARPromptModel(nn.Module):
    first_stage: FirstStageModel
    second_stage: SecondStageModel

    def __init__(self, args, backbone: nn.Module, num_classes: int, dataset: ContinualDataset, device='cpu'):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device = device
        self.dataset = dataset
        print("Loading first stage...")
        self.first_stage = FirstStageModel(args=args, num_classes=num_classes, dataset=dataset, device=device)

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.first_stage.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        print("Loading second stage...")
        self.second_stage = SecondStageModel(args=args, num_classes=num_classes,
                                             dataset=dataset, backbone=backbone,
                                             clip_model=self.first_stage.prompter.clip_model,
                                             clip_preprocess=self.first_stage.prompter.clip_preprocess,
                                             device=device)

        embed_dim = self.second_stage.vit.embed_dim

        self.second_stage_distributions = torch.nn.ModuleList([self._get_dist(embed_dim)
                                                               for _ in range(self.num_classes)]).to(self.device)
        self.classifier_state_dict = None
        print("Done.")

    def _get_dist(self, embed_dim):
        assert self.args.gr_model in ['mog', 'gaussian'], f"Invalid GR model: {self.args.gr_model}"

        if self.args.gr_model == 'mog':
            return MixtureOfGaussiansModel(embed_dim, n_components=self.args.gr_mog_n_components,
                                           n_iters=self.args.gr_mog_n_iters_second_stage)
        else:
            return Gaussian(embed_dim)

    @torch.no_grad()
    def update_keys(self, start_c: int, end_c: int):
        print('Updating keys for second stage...')
        first_stage_keys = self.first_stage.prompter.compute_keys(start_c, end_c)
        self.second_stage.prompter.set_keys(first_stage_keys, start_c, end_c)

    def forward(self, x: torch.Tensor, cur_classes: int, frozen_past_classes=0, return_query=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the complete forward pass of STAR-Prompt.
        This assumes that the keys are already pre-computed.

        Args:
            x: The input tensor.
            cur_classes: The number of current classes.
            frozen_past_classes: The number of past classes.
            return_query: Whether to return the query tensor with the output.
        """
        return self.second_stage(x, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes, return_query=return_query)

    def train(self, mode: bool = True):
        self.first_stage.train(mode)
        self.second_stage.train(mode)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.first_stage.to(device, *args, **kwargs)
        self.second_stage.to(device, *args, **kwargs)
        self.device = device

        return self

    @torch.no_grad()
    def eval_first_stage_on_task(self, dataset: ContinualDataset, n_seen_classes: int) -> torch.Tensor:
        """
        Compute and return the accuracy on each task so far.
        """
        was_training = self.first_stage.training
        self.first_stage.eval()
        all_accs = []
        with tqdm(total=sum([len(test_loader) for test_loader in dataset.test_loaders]), desc='Eval first stage on seen tasks') as pbar:
            for t, test_loader in enumerate(dataset.test_loaders):
                total = 0
                correct = 0
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)
                    logits = self.first_stage(inputs, cur_classes=n_seen_classes)[:, :n_seen_classes]
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.update(1)
                all_accs.append(correct / total)
        self.first_stage.train(was_training)
        return torch.tensor(all_accs)

    def norm(self, t):
        return torch.norm(t, p=2, dim=-1, keepdim=True) + 1e-7

    @torch.no_grad()
    def create_features_dataset(self, current_task: int):

        labels, features = [], []

        for _ti in range(current_task + 1):

            prev_t_size, cur_t_size = self.dataset.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):
                current_samples = self.second_stage_distributions[class_idx](self.args.num_samples_gr)
                features.append(current_samples)
                labels.append(torch.ones(self.args.num_samples_gr) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return create_seeded_dataloader(self.args, TensorDataset(features, labels),
                                        batch_size=self.args.batch_size_gr,
                                        shuffle=True, num_workers=0, non_verbose=True)

    def train_alignment_epoch(self, classifier: torch.nn.Module, optim: torch.optim.Optimizer, n_seen_classes: int, current_task: int, loss_fn):

        dl = self.create_features_dataset(current_task)

        with tqdm(enumerate(dl), total=len(dl), desc='GR epoch') as pbar:
            for i, (x, labels) in pbar:
                optim.zero_grad()
                x, labels = x.to(self.device, dtype=torch.float32), labels.to(self.device)

                logits = classifier(x)

                logits = logits[:, :n_seen_classes]

                norm = self.norm(logits)
                logits = logits / (0.1 * norm)

                loss = loss_fn(logits, labels)
                loss.backward()
                optim.step()

                if not self.args.nowand:
                    assert wandb is not None, "wandb is not installed."
                    wandb.log({'ca_loss_second_stage': loss.item(), 'ca_lr_second_stage': optim.param_groups[0]['lr']})
                pbar.set_postfix({'loss': loss.item()}, refresh=False)

    def align(self, current_task: int, n_seen_classes: int, loss_fn):

        classifier = deepcopy(self.second_stage.vit.head)

        optim = torch.optim.SGD(lr=self.args.learning_rate_gr_second_stage,
                                params=classifier.parameters(),
                                momentum=0.0,
                                weight_decay=0.0)

        num_epochs = self.args.num_epochs_gr_second_stage + (5 * current_task)

        for e in range(num_epochs):
            self.train_alignment_epoch(classifier, optim, n_seen_classes=n_seen_classes, current_task=current_task, loss_fn=loss_fn)

        self.second_stage.vit.head.weight.data.copy_(classifier.weight.data)
        self.second_stage.vit.head.bias.data.copy_(classifier.bias.data)

    @torch.no_grad()
    def update_statistics(self, dataset: ContinualDataset, n_past_classes: int, n_seen_classes: int):

        features_dict = {i: [] for i in range(n_past_classes, n_seen_classes)}

        self.second_stage.eval()

        with tqdm(total=self.args.num_monte_carlo_gr_second_stage * len(dataset.train_loader), desc='GR update statistics') as pbar:
            for _ in range(self.args.num_monte_carlo_gr_second_stage):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3 and min([len(v) for k, v in features_dict.items()]) > self.args.gr_mog_n_components:
                        break

                    x, labels = data[0], data[1]
                    x, labels = x.to(self.device), labels.to(self.device, dtype=torch.long)
                    features = self.second_stage(x, return_features=True, cur_classes=n_seen_classes, frozen_past_classes=n_past_classes)
                    features = features[:, 0]

                    for class_idx in labels.unique():
                        features_dict[int(class_idx)].append(features[labels == class_idx])

                    pbar.update(1)

        for class_idx in range(n_past_classes, n_seen_classes):
            features_class_idx = torch.cat(features_dict[class_idx], dim=0)
            self.second_stage_distributions[class_idx].fit(features_class_idx.to(self.device))

    def backup(self, current_task: int, n_past_classes: int, n_seen_classes: int):
        print(f"BACKUP: Task - {current_task} - classes from "
              f"{n_past_classes} - to {n_seen_classes}")
        self.classifier_state_dict = deepcopy(self.second_stage.vit.head.state_dict())

    def recall_classifier_second_stage(self, current_task: int, n_past_classes: int, n_seen_classes: int):
        print(f"RECALL: Task - {current_task} - classes from "
              f"{n_past_classes} - to {n_seen_classes}")

        if current_task == 0 or not self.args.enable_gr:
            return

        assert self.classifier_state_dict

        self.second_stage.vit.head.weight.data.copy_(self.classifier_state_dict['weight'].data)
        self.second_stage.vit.head.bias.data.copy_(self.classifier_state_dict['bias'].data)

    @torch.enable_grad()
    def train_first_stage_on_task(self, dataset: ContinualDataset, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
        """
        Train the first stage on the current task.

        Args:
            dataset: The continual dataset for the current task, containing both train and test (validation) set.
            current_task: The current task index.
            n_past_classes: The number of past classes.
            n_seen_classes: The number of seen classes.
            loss_fn: The loss function.
        """
        print("Starting training of first stage on task", current_task)
        # BEGIN-TASK
        old_train_transform = dataset.train_loader.dataset.transform
        old_test_transform = dataset.test_loaders[-1].dataset.transform

        # use CLIP's preprocessing
        dataset.train_loader.dataset.transform = self.first_stage.prompter.clip_preprocess
        dataset.test_loaders[-1].dataset.transform = self.first_stage.prompter.clip_preprocess

        convert_weights(self.first_stage.prompter.clip_model)  # convert weights to float16 during training for speedup
        self.first_stage.prompter.text_encoder.dtype = torch.float16
        was_training = self.first_stage.training
        self.first_stage.train()

        first_stage_params = [v for k, v in self.first_stage.named_parameters() if 'prompt_parameters' in k]
        if self.args.first_stage_optim == 'sgd':
            opt = torch.optim.SGD(first_stage_params, lr=self.args.first_stage_lr, momentum=self.args.first_stage_momentum,
                                  weight_decay=self.args.first_stage_weight_decay)
        else:
            opt = torch.optim.Adam(first_stage_params, lr=self.args.first_stage_lr,
                                   weight_decay=self.args.first_stage_weight_decay)

        # MINI TRAINING LOOP FOR CURRENT TASK
        with tqdm(total=self.args.first_stage_epochs * len(dataset.train_loader), desc='First stage training') as pbar:
            for epoch in range(self.args.first_stage_epochs):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3:
                        break
                    inputs, labels = data[0].to(self.device), data[1].to(self.device, dtype=torch.long)
                    loss = torch.tensor(0.).to(self.device)

                    opt.zero_grad()
                    # Check cur and past classes
                    clip_logits = self.first_stage(inputs, frozen_past_classes=n_past_classes, cur_classes=n_seen_classes)

                    # compute clip loss
                    clip_logits[:, :n_past_classes] = -float('inf')
                    loss_clip = loss_fn(clip_logits[:, :n_seen_classes], labels)

                    loss += loss_clip

                    loss_ortho_coop = self.first_stage.prompter.compute_ortho_loss(frozen_past_classes=n_past_classes, cur_classes=n_seen_classes)
                    loss += self.args.lambda_ortho_first_stage * loss_ortho_coop

                    if i == 0:
                        opt.zero_grad()
                    (loss / self.args.virtual_bs_n).backward()
                    if (i > 0 or self.args.virtual_bs_n == 1) and i % self.args.virtual_bs_n == 0:
                        opt.step()
                        opt.zero_grad()

                    if not self.args.nowand:
                        assert wandb is not None, "wandb is not installed."
                        wandb.log({'first_stage_loss': loss.item(),
                                   'first_stage_lr': opt.param_groups[0]['lr'],
                                   'first_stage_epoch': epoch,
                                   'first_stage_loss_clip': loss_clip.item(),
                                   'first_stage_loss_ortho': loss_ortho_coop.item(),
                                   'first_stage_iteration': i})

                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()}, refresh=False)

        # END-TASK
        opt.zero_grad(set_to_none=True)
        del opt
        torch.cuda.empty_cache()

        # Generative replay after end of task
        if self.args.enable_gr:
            self.first_stage.prompter.update_statistics(dataset, current_task)
            self.first_stage.prompter.align(current_task)

        cur_acc = self.eval_first_stage_on_task(dataset, n_seen_classes)
        print(f'First stage accuracy: {[acc.item() for acc in cur_acc]}')
        print(f'\tAverage: {cur_acc.mean().item():.4f}')
        if not self.args.nowand:
            assert wandb is not None, "wandb is not installed."
            log_dict = {f'first_stage_acc_{i}': acc.item() for i, acc in enumerate(cur_acc)}
            log_dict['first_stage_acc'] = cur_acc.mean().item()
            wandb.log(log_dict)

        # restore original transforms
        dataset.train_loader.dataset.transform = old_train_transform
        dataset.test_loaders[-1].dataset.transform = old_test_transform

        self.first_stage.prompter.clip_model.float()  # convert back to float32
        self.first_stage.prompter.text_encoder.dtype = torch.float32
        self.first_stage.train(was_training)
