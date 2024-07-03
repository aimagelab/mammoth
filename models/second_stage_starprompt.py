import torch
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from argparse import ArgumentParser

from utils.schedulers import CosineSchedule
from models.utils.continual_model import ContinualModel
from models.star_prompt_utils.second_stage_model import Model
from models.star_prompt_utils.generative_replay import MixtureOfGaussiansModel


class SecondStageStarprompt(ContinualModel):
    NAME = 'second_stage_starprompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Second-stage of StarPrompt. Requires the keys saved from the first stage.')

        # Frozen hyperparameters
        parser.add_argument("--virtual_bs_n", type=int, default=1,
                            help="virtual batch size iterations")
        parser.add_argument('--gr_mog_n_components', type=int, default=5,
                            help="Number of components for GR with MOG.")

        # Tunable hyperparameters
        parser.add_argument("--ortho_split_val", type=int, default=0)
        parser.add_argument("--lambda_ortho", type=float, default=10,
                            help="orthogonality loss coefficient")
        parser.add_argument("--num_monte_carlo_gr", type=int, default=1,
                            help="how many times to sample from the dataset for alignment")
        parser.add_argument("--num_epochs_gr", type=int, default=10,
                            help="Num. of epochs for GR.")
        parser.add_argument("--learning_rate_gr", type=float, default=0.001,
                            help="Learning rate for GR.")
        parser.add_argument('--gr_mog_n_iters', type=int, default=500,
                            help="Number of EM iterations during fit for GR with MOG.")

        parser.add_argument('--keys_ckpt_path', type=str, required=True,
                            help="Path for first-stage keys. The keys can be saved by runninng `first_stage_starprompt` with `--save_first_stage_keys=1`.")

        parser.add_argument('--batch_size_gr', type=int, default=128,
                            help="Batch size for Generative Replay.")
        parser.add_argument('--num_samples_gr', type=int, default=256,
                            help="Number of samples for Generative Replay.")

        return parser

    net: Model

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.net = Model(args,
                         backbone=self.net,
                         dataset=self.dataset,
                         num_classes=self.num_classes)

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.net.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        embed_dim = self.net.vit.embed_dim

        self.distributions = torch.nn.ModuleList([self._get_dist(embed_dim)
                                                  for _ in range(self.num_classes)]).to(self.device)
        self.classifier_state_dict = None

    def _get_dist(self, embed_dim):
        return MixtureOfGaussiansModel(embed_dim, n_components=self.args.gr_mog_n_components,
                                       n_iters=self.args.gr_mog_n_iters)

    def norm(self, t):
        return torch.norm(t, p=2, dim=-1, keepdim=True) + 1e-7

    def per_task_norms(self, logits):

        per_task_norm = []

        for _ti in range(self.current_task + 1):
            prev_t_size, cur_t_size = self.compute_offsets(_ti)
            temp_norm = self.norm(logits[:, prev_t_size:cur_t_size])
            per_task_norm.append(temp_norm)

        per_task_norm = torch.cat(per_task_norm, dim=-1)
        norms = per_task_norm.mean(dim=-1, keepdim=True)

        return norms

    @torch.no_grad()
    def create_features_dataset(self):

        labels, features = [], []

        for _ti in range(self.current_task + 1):

            prev_t_size, cur_t_size = self.compute_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):

                current_samples = self.distributions[class_idx](self.args.num_samples_gr)
                features.append(current_samples)
                labels.append(torch.ones(self.args.num_samples_gr) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return DataLoader(TensorDataset(features, labels),
                          batch_size=self.args.batch_size_gr,
                          shuffle=True, num_workers=0)

    def train_alignment_epoch(self, classifier: torch.nn.Module, optim: torch.optim.Optimizer):

        dl = self.create_features_dataset()

        for i, (x, labels) in tqdm(enumerate(dl), total=len(dl), desc='GR epoch'):
            optim.zero_grad()
            x, labels = x.to(self.device), labels.to(self.device)

            logits = classifier(x)

            logits = logits[:, :self.n_seen_classes]

            norm = self.norm(logits)
            logits = logits / (0.1 * norm)

            loss = self.loss(logits, labels)
            loss.backward()
            optim.step()

    def align(self):

        classifier = deepcopy(self.net.vit.head)

        optim = torch.optim.SGD(lr=self.args.learning_rate_gr,
                                params=classifier.parameters(),
                                momentum=0.0,
                                weight_decay=0.0)

        num_epochs = self.args.num_epochs_gr + (5 * self.current_task)

        for e in range(num_epochs):
            self.train_alignment_epoch(classifier, optim)

        self.net.vit.head.weight.data.copy_(classifier.weight.data)
        self.net.vit.head.bias.data.copy_(classifier.bias.data)

    @torch.no_grad()
    def update_statistics(self, dataset):

        features_dict = {i: [] for i in range(self.n_past_classes, self.n_seen_classes)}

        self.net.eval()

        with tqdm(total=self.args.num_monte_carlo_gr * len(dataset.train_loader), desc='GR update statistics') as pbar:
            for _ in range(self.args.num_monte_carlo_gr):
                for i, data in enumerate(dataset.train_loader):
                    x, labels = data[0], data[1]
                    x, labels = x.to(self.device), labels.to(self.device, dtype=torch.long)
                    features = self.net(x, return_features=True, cur_classes=self.n_seen_classes, frozen_past_classes=self.n_past_classes)
                    features = features[:, 0]

                    for class_idx in labels.unique():
                        features_dict[int(class_idx)].append(features[labels == class_idx])

                    pbar.update(1)

        for class_idx in range(self.n_past_classes, self.n_seen_classes):
            features_class_idx = torch.cat(features_dict[class_idx], dim=0)
            self.distributions[class_idx].fit(features_class_idx.to(self.device))

    def backup(self):
        print(f"BACKUP: Task - {self.current_task} - classes from "
              f"{self.n_past_classes} - to {self.n_seen_classes}")
        self.classifier_state_dict = deepcopy(self.net.vit.head.state_dict())

    def recall(self):
        print(f"RECALL: Task - {self.current_task} - classes from "
              f"{self.n_past_classes} - to {self.n_seen_classes}")

        if self.current_task == 0:
            return

        assert self.classifier_state_dict

        self.net.vit.head.weight.data.copy_(self.classifier_state_dict['weight'].data)
        self.net.vit.head.bias.data.copy_(self.classifier_state_dict['bias'].data)

    def end_task(self, dataset):
        if hasattr(self, 'opt'):
            del self.opt  # free up some vram

        self.update_statistics(dataset)
        self.backup()

        if self.current_task > 0:
            self.align()

    def get_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def get_scheduler(self):
        return CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_task(self, dataset):
        self.recall()

        if hasattr(self, 'opt'):
            del self.opt

        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler() # TODO: check custom scheduler

    def forward(self, x):
        logits = self.net(x, cur_classes=self.n_seen_classes)
        logits = logits[:, :self.n_seen_classes]
        return logits

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        stream_inputs, stream_labels = inputs, labels
        stream_logits = self.net(stream_inputs, cur_classes=self.n_seen_classes, frozen_past_classes=self.n_past_classes)

        # mask old classes
        stream_logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(stream_logits[:, :self.n_seen_classes], stream_labels)

        loss_ortho = self.net.prompter.compute_ortho_loss(frozen_past_classes=self.n_past_classes, cur_classes=self.n_seen_classes)
        loss += self.args.lambda_ortho * loss_ortho

        if self.epoch_iteration == 0:
            self.opt.zero_grad()

        loss.backward()
        if (self.epoch_iteration > 0 or self.args.virtual_bs_n == 1) and \
                self.epoch_iteration % self.args.virtual_bs_n == 0:
            # NOTE: The virtual batch size is missing `loss = loss/self.virtual_bs_n`. We did not see any significant change with this as Adam will take care of it.
            self.opt.step()
            self.opt.zero_grad()

        self.epoch_iteration += 1

        return loss.item()
