import logging
import torch
from argparse import ArgumentParser

import torch

from models.star_prompt_utils.end_to_end_model import STARPromptModel
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.schedulers import CosineSchedule

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git (requires also `huggingface-hub`)")


class STARPrompt(ContinualModel):
    """Second-stage of StarPrompt. Requires the keys saved from the first stage."""
    NAME = 'starprompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    net: STARPromptModel

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(batch_size=128, optimizer='adam', lr=0.001)

        frozen_group = parser.add_argument_group('Frozen hyperparameters')
        frozen_group.add_argument("--virtual_bs_n", type=int, default=1,
                                  help="virtual batch size iterations")
        frozen_group.add_argument("--ortho_split_val", type=int, default=0)
        frozen_group.add_argument('--gr_mog_n_iters_second_stage', type=int, default=500,
                                  help="Number of EM iterations during fit for GR with MOG on the second stage.")
        frozen_group.add_argument('--gr_mog_n_iters_first_stage', type=int, default=200,
                                  help="Number of EM iterations during fit for GR with MOG on the first stage.")
        frozen_group.add_argument('--gr_mog_n_components', type=int, default=5,
                                  help="Number of components for GR with MOG (both first and second stage).")
        frozen_group.add_argument('--batch_size_gr', type=int, default=128,
                                  help="Batch size for Generative Replay (both first and second stage).")
        frozen_group.add_argument('--num_samples_gr', type=int, default=256,
                                  help="Number of samples for Generative Replay (both first and second stage).")
        frozen_group.add_argument('--prefix_tuning_prompt_len', type=int, default=5,
                                  help="Prompt length for prefix tuning. Used only if `--prompt_mode==concat`.")

        ablation_group = parser.add_argument_group('Ablations hyperparameters')
        ablation_group.add_argument('--gr_model', type=str, default='mog', choices=['mog', 'gaussian'],
                                    help="Type of distribution model for Generative Replay (both first and second stage). "
                                    "- `mog`: Mixture of Gaussian. "
                                    "- `gaussian`: Single Gaussian distribution.")
        ablation_group.add_argument("--enable_gr", type=binary_to_boolean_type, default=1,
                                    help="Enable Generative Replay (both first and second stage).")
        ablation_group.add_argument('--prompt_mode', type=str, default='residual', choices=['residual', 'concat'],
                                    help="Prompt type for the second stage. "
                                    "- `residual`: STAR-Prompt style prompting. "
                                    "- `concat`: Prefix-Tuning style prompting.")
        ablation_group.add_argument("--enable_confidence_modulation", type=binary_to_boolean_type, default=1,
                                    help="Enable confidence modulation with CLIP similarities (Eq. 5 of the main paper)?")

        tunable_group = parser.add_argument_group('Tunable hyperparameters')
        # second stage
        tunable_group.add_argument("--lambda_ortho_second_stage", type=float, default=10,
                                   help="orthogonality loss coefficient")
        tunable_group.add_argument("--num_monte_carlo_gr_second_stage", type=int, default=1,
                                   help="how many times to sample from the dataset for alignment")
        tunable_group.add_argument("--num_epochs_gr_second_stage", type=int, default=10,
                                   help="Num. of epochs for GR.")
        tunable_group.add_argument("--learning_rate_gr_second_stage", type=float, default=0.001,
                                   help="Learning rate for GR.")
        # first stage
        tunable_group.add_argument("--num_monte_carlo_gr_first_stage", type=int, default=1,
                                   help="how many times to sample from the dataset for alignment")
        tunable_group.add_argument("--learning_rate_gr_first_stage", type=float, default=0.05,
                                   help="Learning rate for Generative Replay.")
        tunable_group.add_argument("--lambda_ortho_first_stage", type=float, default=30,
                                   help="Orthogonality loss coefficient for coop")
        tunable_group.add_argument("--num_epochs_gr_first_stage", type=int, default=10,
                                   help="Num. of epochs for Generative Replay.")

        parser.add_argument("--clip_backbone", type=str, default='ViT-L/14', help="CLIP backbone architecture",
                            choices=clip.available_models())

        first_stage_optim_group = parser.add_argument_group('First stage optimization hyperparameters')
        first_stage_optim_group.add_argument("--first_stage_optim", type=str, default='sgd', choices=['sgd', 'adam'],
                                             help="First stage optimizer")
        first_stage_optim_group.add_argument("--first_stage_lr", type=float, default=0.002, help="First stage learning rate")
        first_stage_optim_group.add_argument("--first_stage_momentum", type=float, default=0, help="First stage momentum")
        first_stage_optim_group.add_argument("--first_stage_weight_decay", type=float, default=0, help="First stage weight decay")
        first_stage_optim_group.add_argument("--first_stage_epochs", type=int, help="First stage epochs. If not set, it will be the same as `n_epochs`.")

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if not hasattr(args, 'first_stage_epochs') or args.first_stage_epochs is None:
            logging.info("`first_stage_epochs` not set. Setting it to `n_epochs`.")
            args.first_stage_epochs = args.n_epochs

        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = STARPromptModel(args,
                                   backbone=self.net,
                                   dataset=self.dataset,
                                   num_classes=self.num_classes,
                                   device=self.device)

    def end_task(self, dataset):
        if hasattr(self, 'opt'):
            del self.opt  # free up some vram

        if self.args.enable_gr:
            self.net.update_statistics(dataset, self.n_past_classes, self.n_seen_classes)
            self.net.backup(self.current_task, self.n_past_classes, self.n_seen_classes)

            if self.current_task > 0:
                if self.args.seed is not None:
                    torch.manual_seed(self.args.seed)
                self.net.align(self.current_task, self.n_seen_classes, self.loss)

    def get_parameters(self):
        if not isinstance(self.net, STARPromptModel):  # during initialization
            return super().get_parameters()
        return [p for p in self.net.second_stage.parameters() if p.requires_grad]

    def get_scheduler(self):
        return CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_task(self, dataset):
        # clean junk on GPU
        if hasattr(self, 'opt'):
            del self.opt

        torch.cuda.empty_cache()

        # adapt CLIP on current task
        self.net.train_first_stage_on_task(dataset, self.current_task, self.n_past_classes, self.n_seen_classes, self.loss)
        self.net.update_keys(self.n_past_classes, self.n_seen_classes)
        self.net.second_stage.train()

        # initialize second stage

        # For later GR
        self.net.recall_classifier_second_stage(self.current_task, self.n_past_classes, self.n_seen_classes)

        self.opt = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def forward(self, x):
        logits = self.net(x, cur_classes=self.n_seen_classes)
        logits = logits[:, :self.n_seen_classes]
        return logits

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):  # second stage only
        stream_inputs, stream_labels = inputs, labels
        stream_logits = self.net(stream_inputs, cur_classes=self.n_seen_classes, frozen_past_classes=self.n_past_classes)

        # Compute accuracy on current training batch for logging
        with torch.no_grad():
            stream_preds = stream_logits[:, :self.n_seen_classes].argmax(dim=1)
            stream_acc = (stream_preds == stream_labels).sum().item() / stream_labels.shape[0]

        # mask old classes
        stream_logits[:, :self.n_past_classes] = -float('inf')
        loss = self.loss(stream_logits[:, :self.n_seen_classes], stream_labels)

        loss_ortho = self.net.second_stage.prompter.compute_ortho_loss(frozen_past_classes=self.n_past_classes, cur_classes=self.n_seen_classes)
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
