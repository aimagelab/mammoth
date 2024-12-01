import logging
import os
import sys
import torch
from argparse import ArgumentParser

from utils import binary_to_boolean_type
from utils.checkpoints import to_parsable_obj
try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git (requires also `huggingface-hub`)")

from models.utils.continual_model import ContinualModel
from models.star_prompt_utils.first_stage_model import Model


class FirstStageStarprompt(ContinualModel):
    NAME = 'first_stage_starprompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    net: Model

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(batch_size=128, optimizer='sgd', lr=0.002)

        frozen_group = parser.add_argument_group('Frozen hyperparameters')
        frozen_group.add_argument("--virtual_bs_n", type=int, default=1, help="Virtual batch size iterations")
        frozen_group.add_argument('--gr_mog_n_iters', '--gr_mog_n_iters_first_stage', dest='gr_mog_n_iters_first_stage',
                                  type=int, default=500, help="Number of EM iterations during fit for GR with MOG.")
        frozen_group.add_argument('--gr_mog_n_components', type=int, default=5,
                                  help="Number of components for Generative Replay with MOG.")
        frozen_group.add_argument("--enable_gr", type=binary_to_boolean_type, default=1,
                                  help="Enable Generative Replay.")
        frozen_group.add_argument('--batch_size_gr', type=int, default=128,
                                  help="Batch size for Generative Replay.")
        frozen_group.add_argument('--num_samples_gr', type=int, default=256,
                                  help="Number of samples for Generative Replay.")

        # Tunable hyperparameters
        tunable_group = parser.add_argument_group('Tunable hyperparameters')
        tunable_group.add_argument("--num_monte_carlo_gr", "--num_monte_carlo_gr_first_stage", dest="num_monte_carlo_gr_first_stage",
                                   type=int, default=2, help="How many times to sample from the dataset for Generative Replay")
        tunable_group.add_argument("--learning_rate_gr", "--learning_rate_gr_first_stage", dest="learning_rate_gr_first_stage",
                                   type=float, default=0.05, help="Learning rate for Generative Replay.")
        tunable_group.add_argument("--lambda_ortho_first_stage", type=float, default=30,
                                   help="Orthogonality loss coefficient for coop")
        tunable_group.add_argument("--num_epochs_gr", "--num_epochs_gr_first_stage", dest="num_epochs_gr_first_stage",
                                   type=int, default=10, help="Num. of epochs for Generative Replay.")

        # Useful flags
        parser.add_argument("--save_first_stage_keys", type=binary_to_boolean_type, default=1,
                            help="save text encoder outputs")
        parser.add_argument("--save_first_stage_keys_filename", type=str, help="filename for saving text encoder outputs. Default is:"
                            "coop_keys_<N_TASKS-1>_<conf_jobnum>.pt")

        # Backbone arguments
        parser.add_argument("--clip_backbone", type=str, default='ViT-L/14', help="CLIP backbone architecture",
                            choices=clip.available_models())

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        logging.info("The first stage of STAR-Prompt ignores the backbone as it uses CLIP")
        del backbone

        super().__init__(None, loss, args, transform, dataset=dataset)
        self.net = Model(args, num_classes=self.num_classes, dataset=self.dataset, device=self.device)
        self.opt = self.get_optimizer()

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.net.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        self.eye = torch.eye(self.num_classes).to(self.device)

    def end_task(self, dataset):
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            delattr(self, 'opt')

        # Generative replay
        if self.args.enable_gr:
            self.net.prompter.update_statistics(dataset, self.current_task)
            self.net.prompter.align(self.current_task)

        if self.current_task == (self.n_tasks - 1) and self.args.save_first_stage_keys:
            print('Saving text encoder outputs... ', end='', file=sys.stderr)
            te_outputs = self.net.prompter.compute_keys(0, self.num_classes)
            os.makedirs('./coop_keys', exist_ok=True)
            st = {
                'keys': te_outputs,
                'args': to_parsable_obj(self.args),
            }
            if self.args.save_first_stage_keys_filename is not None:
                fname = f'./coop_keys/{self.args.save_first_stage_keys_filename}'
            else:
                fname = f'./coop_keys/coop_keys_{self.current_task}_{self.args.conf_jobnum}.pt'
            torch.save(st, fname)
            print('Saved text-encoder keys in:', fname, file=sys.stderr)

    def get_parameters(self):
        return [v for k, v in self.net.named_parameters() if 'prompt_parameters' in k]

    def begin_task(self, dataset):
        # Disable transforms and set normalization as CLIP's preprocessing
        dataset.train_loader.dataset.transform = self.net.prompter.clip_preprocess
        dataset.test_loaders[-1].dataset.transform = self.net.prompter.clip_preprocess

        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            delattr(self, 'opt')

        self.opt = self.get_optimizer()

        torch.cuda.empty_cache()

    def forward(self, x):
        logits = self.net(x, cur_classes=self.n_seen_classes)
        return logits[:, :self.n_seen_classes]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        loss = torch.tensor(0.).to(self.device)

        stream_inputs, stream_labels = inputs, labels.long()
        clip_logits = self.net(stream_inputs, frozen_past_classes=self.n_past_classes, cur_classes=self.n_seen_classes)

        # compute clip loss
        clip_logits[:, :self.n_past_classes] = -float('inf')
        loss_clip = self.loss(clip_logits[:, :self.n_seen_classes], stream_labels)

        loss += loss_clip

        loss_ortho_coop = self.net.prompter.compute_ortho_loss(frozen_past_classes=self.n_past_classes, cur_classes=self.n_seen_classes)
        loss += self.args.lambda_ortho_first_stage * loss_ortho_coop

        if self.epoch_iteration == 0:
            self.opt.zero_grad()
        (loss / self.args.virtual_bs_n).backward()
        if (self.epoch_iteration > 0 or self.args.virtual_bs_n == 1) and self.epoch_iteration % self.args.virtual_bs_n == 0:
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()
