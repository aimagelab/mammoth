import torch
from copy import deepcopy
from torch.utils.data import TensorDataset
from tqdm import tqdm
from argparse import ArgumentParser

try:
    import wandb
except ImportError:
    wandb = None

from utils import binary_to_boolean_type
from utils.augmentations import RepeatedTransform
from utils.conf import create_seeded_dataloader
from utils.schedulers import CosineSchedule
from models.utils.continual_model import ContinualModel
from models.star_prompt_utils.second_stage_model import Model
from models.star_prompt_utils.generative_replay import Gaussian, MixtureOfGaussiansModel


class SecondStageStarprompt(ContinualModel):
    """Second-stage of StarPrompt. Requires the keys saved from the first stage."""
    NAME = 'second_stage_starprompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(batch_size=128, optimizer='adam', lr=0.001)

        frozen_group = parser.add_argument_group('Frozen hyperparameters')
        frozen_group.add_argument("--virtual_bs_n", type=int, default=1,
                                  help="virtual batch size iterations")
        frozen_group.add_argument("--enable_data_aug_query", type=binary_to_boolean_type, default=1,
                                  help="Use default transform with data aug to generate the CLIP's response?")
        frozen_group.add_argument("--use_clip_preprocess_eval", type=binary_to_boolean_type, default=0,
                                  help="Use CLIP's transform during eval instead of the default test transform?")
        frozen_group.add_argument("--ortho_split_val", type=int, default=0)
        frozen_group.add_argument('--gr_mog_n_iters', '--gr_mog_n_iters_second_stage', dest='gr_mog_n_iters_second_stage',
                                  type=int, default=500, help="Number of EM iterations during fit for GR with MOG.")
        frozen_group.add_argument('--gr_mog_n_components', type=int, default=5,
                                  help="Number of components for GR with MOG.")
        frozen_group.add_argument('--batch_size_gr', type=int, default=128,
                                  help="Batch size for Generative Replay.")
        frozen_group.add_argument('--num_samples_gr', type=int, default=256,
                                  help="Number of samples for Generative Replay.")
        frozen_group.add_argument('--prefix_tuning_prompt_len', type=int, default=5,
                                  help="Prompt length for prefix tuning. Used only if `--prompt_mode==concat`.")

        ablation_group = parser.add_argument_group('Ablations hyperparameters')
        ablation_group.add_argument('--gr_model', type=str, default='mog', choices=['mog', 'gaussian'],
                                    help="Type of distribution model for Generative Replay. "
                                    "- `mog`: Mixture of Gaussian. "
                                    "- `gaussian`: Single Gaussian distribution.")
        ablation_group.add_argument("--enable_gr", type=binary_to_boolean_type, default=1,
                                    help="Enable Generative Replay.")
        ablation_group.add_argument('--statc_keys_use_templates', type=binary_to_boolean_type, default=1,
                                    help="Use templates for the second stage if no keys are loaded.")
        ablation_group.add_argument('--prompt_mode', type=str, default='residual', choices=['residual', 'concat'],
                                    help="Prompt type for the second stage. "
                                    "- `residual`: STAR-Prompt style prompting. "
                                    "- `concat`: Prefix-Tuning style prompting.")
        ablation_group.add_argument("--enable_confidence_modulation", type=binary_to_boolean_type, default=1,
                                    help="Enable confidence modulation with CLIP similarities (Eq. 5 of the main paper)?")

        # Tunable hyperparameters
        tunable_group = parser.add_argument_group('Tunable hyperparameters')
        tunable_group.add_argument("--lambda_ortho_second_stage", type=float, default=10,
                                   help="orthogonality loss coefficient")
        tunable_group.add_argument("--num_monte_carlo_gr", "--num_monte_carlo_gr_second_stage", dest="num_monte_carlo_gr_second_stage",
                                   type=int, default=1, help="how many times to sample from the dataset for alignment")
        tunable_group.add_argument("--num_epochs_gr", "--num_epochs_gr_second_stage", dest="num_epochs_gr_second_stage",
                                   type=int, default=10, help="Num. of epochs for GR.")
        tunable_group.add_argument("--learning_rate_gr", "--learning_rate_gr_second_stage", dest="learning_rate_gr_second_stage",
                                   type=float, default=0.001, help="Learning rate for GR.")

        # Very important parameter
        parser.add_argument('--keys_ckpt_path', type=str,
                            help="Path for first-stage keys. "
                            "The keys can be saved by runninng `first_stage_starprompt` with `--save_first_stage_keys=1`."
                            "This can be:"
                            "- A path to a checkpoint file (.pt) containing ONLY THE FIRST STAGE KEYS."
                            "- A path to the checkpoint made by `first_stage_starprompt`"
                            "- The job-id (`conf_jobnum`) of the `first_stage_starprompt` run that made the keys."
                            "- A JSON file containing the job-id (`conf_jobnum`) of the `first_stage_starprompt` run that made the keys."
                            "The JSON is expected to contain an entry for each dataset and seed: `{dataset: {seed: job-id}}`.")

        return parser

    net: Model

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = Model(args,
                         backbone=self.net,
                         dataset=self.dataset,
                         num_classes=self.num_classes,
                         device=self.device)

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.net.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        embed_dim = self.net.vit.embed_dim

        self.distributions = torch.nn.ModuleList([self._get_dist(embed_dim)
                                                  for _ in range(self.num_classes)]).to(self.device)
        self.classifier_state_dict = None

    def _get_dist(self, embed_dim):
        assert self.args.gr_model in ['mog', 'gaussian'], f"Invalid GR model: {self.args.gr_model}"

        if self.args.gr_model == 'mog':
            return MixtureOfGaussiansModel(embed_dim, n_components=self.args.gr_mog_n_components,
                                           n_iters=self.args.gr_mog_n_iters_second_stage)
        else:
            return Gaussian(embed_dim)

    def norm(self, t):
        return torch.norm(t, p=2, dim=-1, keepdim=True) + 1e-7

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

        return create_seeded_dataloader(self.args, TensorDataset(features, labels),
                                        batch_size=self.args.batch_size_gr,
                                        shuffle=True, num_workers=0, non_verbose=True)

    def train_alignment_epoch(self, classifier: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int):

        dl = self.create_features_dataset()

        with tqdm(enumerate(dl), total=len(dl), desc=f'GR second stage epoch {epoch + 1}/{self.args.num_epochs_gr_second_stage}', leave=False) as pbar:
            for i, (x, labels) in pbar:
                optim.zero_grad()
                x, labels = x.to(self.device, dtype=torch.float32), labels.to(self.device)

                logits = classifier(x)

                logits = logits[:, :self.n_seen_classes]

                norm = self.norm(logits)
                logits = logits / (0.1 * norm)

                loss = self.loss(logits, labels)
                loss.backward()
                optim.step()

                if not self.args.nowand:
                    assert wandb is not None, "wandb is not installed."
                    wandb.log({'ca_loss_second_stage': loss.item(), 'ca_lr_second_stage': optim.param_groups[0]['lr']})
                pbar.set_postfix({'loss': loss.item()}, refresh=False)

    def align(self):

        classifier = deepcopy(self.net.vit.head)

        optim = torch.optim.SGD(lr=self.args.learning_rate_gr_second_stage,
                                params=classifier.parameters(),
                                momentum=0.0,
                                weight_decay=0.0)

        num_epochs = self.args.num_epochs_gr_second_stage + (5 * self.current_task)

        for e in range(num_epochs):
            self.train_alignment_epoch(classifier, optim, e)

        self.net.vit.head.weight.data.copy_(classifier.weight.data)
        self.net.vit.head.bias.data.copy_(classifier.bias.data)

    @torch.no_grad()
    def update_statistics(self, dataset):

        features_dict = {i: [] for i in range(self.n_past_classes, self.n_seen_classes)}

        self.net.eval()

        with tqdm(total=self.args.num_monte_carlo_gr_second_stage * len(dataset.train_loader), desc='GR update statistics') as pbar:
            for _ in range(self.args.num_monte_carlo_gr_second_stage):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3 and min([len(v) for k, v in features_dict.items()]) > self.args.gr_mog_n_components:
                        break

                    x, labels = data[0], data[1]
                    x, labels = x.to(self.device), labels.to(self.device, dtype=torch.long)
                    x, query_x = x[:, 0], x[:, 1]
                    if self.args.enable_data_aug_query:
                        query_x = None
                    features = self.net(x, query_x=query_x, return_features=True, cur_classes=self.n_seen_classes, frozen_past_classes=self.n_past_classes)
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

        if self.current_task == 0 or not self.args.enable_gr:
            return

        assert self.classifier_state_dict

        self.net.vit.head.weight.data.copy_(self.classifier_state_dict['weight'].data)
        self.net.vit.head.bias.data.copy_(self.classifier_state_dict['bias'].data)

    def end_task(self, dataset):
        if hasattr(self, 'opt'):
            del self.opt  # free up some vram

        if self.args.enable_gr:
            self.update_statistics(dataset)
            self.backup()

            if self.current_task > 0:
                self.align()

    def get_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def get_scheduler(self):
        return CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_task(self, dataset):
        if self.args.permute_classes:
            if hasattr(self.net.prompter, 'old_args') and self.net.prompter.old_args is not None:
                assert self.args.seed == self.net.prompter.old_args.seed
                assert (self.args.class_order == self.net.prompter.old_args.class_order).all()

        dataset.train_loader.dataset.transform = RepeatedTransform([dataset.train_loader.dataset.transform, self.net.prompter.clip_preprocess])
        dataset.test_loaders[-1].dataset.transform = RepeatedTransform([dataset.test_loaders[-1].dataset.transform, self.net.prompter.clip_preprocess])

        # NOTE: Remove these comments if you want to check if the keys are loaded correctly and results are the same as the first stage
        # tot_data, tot_corr = 0, 0
        # for i, ts in enumerate(dataset.test_loaders):
        #     task_tot, task_corr = 0, 0
        #     for data in ts:
        #         inputs, labels = data[0], data[1]
        #         inputs, labels = inputs[:, 1].to(self.device), labels.to(self.device) # only clip-preprocessed input
        #         queries = self.net.prompter.get_query(inputs)
        #         queries = torch.nn.functional.normalize(queries, dim=-1)
        #         logits = torch.einsum('bd,cd->bc', queries, self.net.prompter.keys.type(self.net.prompter.clip_model.dtype))
        #         task_corr += (logits.argmax(dim=-1) == labels).sum().item()
        #         task_tot += labels.shape[0]
        #     print(f"CLIP on TASK {i+1}: {task_corr / task_tot}")
        #     tot_corr += task_corr
        #     tot_data += task_tot
        # print(f"AVG CLIP ON TASKS: {tot_corr / tot_data}") # the avg of the avg != the avg of the total

        # For later GR
        self.recall()

        if hasattr(self, 'opt'):
            del self.opt

        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def forward(self, x):
        x, query_x = x[:, 0], x[:, 1]  # from repeated transform
        if not self.args.use_clip_preprocess_eval:
            query_x = None
        logits = self.net(x, query_x=query_x, cur_classes=self.n_seen_classes)
        logits = logits[:, :self.n_seen_classes]
        return logits

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        stream_inputs, stream_labels = inputs, labels
        stream_inputs, query_stream_inputs = stream_inputs[:, 0], stream_inputs[:, 1]
        if self.args.enable_data_aug_query:
            query_stream_inputs = None
        stream_logits = self.net(stream_inputs, query_x=query_stream_inputs, cur_classes=self.n_seen_classes, frozen_past_classes=self.n_past_classes)

        # Compute accuracy on current training batch for logging
        with torch.no_grad():
            stream_preds = stream_logits[:, :self.n_seen_classes].argmax(dim=1)
            stream_acc = (stream_preds == stream_labels).sum().item() / stream_labels.shape[0]

        # mask old classes
        stream_logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(stream_logits[:, :self.n_seen_classes], stream_labels)

        loss_ortho = self.net.prompter.compute_ortho_loss(frozen_past_classes=self.n_past_classes, cur_classes=self.n_seen_classes)
        loss += self.args.lambda_ortho_second_stage * loss_ortho

        if self.epoch_iteration == 0:
            self.opt.zero_grad()

        (loss / self.args.virtual_bs_n).backward()
        # loss.backward()
        if (self.epoch_iteration > 0 or self.args.virtual_bs_n == 1) and \
                self.epoch_iteration % self.args.virtual_bs_n == 0:
            self.opt.step()
            self.opt.zero_grad()

        return {'loss': loss.item(),
                'stream_accuracy': stream_acc}
