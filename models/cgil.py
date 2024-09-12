from argparse import ArgumentParser

import torch

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.cgil_utils.cgil_utils import Model
from models.utils.future_model import FutureModel


class CGIL(FutureModel):
    NAME = 'cgil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument("--clip_backbone", type=str, default='ViT-L/14', help="Clip backbone")
        parser.add_argument("--learning_rate_alignment", type=float, default=0.05, help="Learning rate for GR.")
        parser.add_argument("--optim_alignment", type=str, default='adamw', choices=('sgd', 'adam', 'adamw'), help="Optimizer for GR.")
        parser.add_argument("--optim_alignment_wd", type=float, default=0, help="Weight decay for GR.")
        parser.add_argument("--lambda_ortho_first_stage", type=float, default=1, help="Orthogonality loss coefficient for coop")
        parser.add_argument("--num_epochs_alignment", type=int, default=30, help="Num. of epochs for GR.")
        parser.add_argument("--batch_size_alignment", type=int, default=128, help="Batch size for alignment.")
        parser.add_argument('--gr_mog_n_components', type=int, default=5, help="Number of components for GR with MOG.")
        parser.add_argument('--gr_mog_n_iters', type=int, default=500, help="Number of EM iterations during fit for GR with MOG.")
        parser.add_argument('--gr_vae_hidden_dim', type=int, default=512, help="Hidden dimension for GR with VAE.")
        parser.add_argument('--gr_vae_latent_dim', type=int, default=256, help="Latent dimension for GR with VAE.")
        parser.add_argument('--gr_vae_n_iters', type=int, default=500, help="Number of iterations for GR with VAE.")
        parser.add_argument('--train_only_current_prompts', type=int, default=0, choices=(0, 1), help="Train only current prompts.")
        parser.add_argument('--align_with_ortholoss', type=int, default=0, choices=(0, 1), help="Align with orthogonality loss.")
        parser.add_argument('--lr_vae', type=float, default=2e-4, help="Learning rate for VAE.")
        parser.add_argument('--general_context', type=int, default=0, help="Use general context (number of contexts created).")
        parser.add_argument('--generated_context', type=int, default=0, help="Use generated context.")
        parser.add_argument('--cocoop', type=int, default=0, help="Use image embedding to generate context.")
        parser.add_argument('--combo_context', type=int, default=1, help="Use both generated and prompt context.")
        parser.add_argument('--n_context', type=int, default=1, help="Use both generated and prompt context.")
        parser.add_argument("--g_models", type=str, default='vae', choices=('vae', 'mog', 'gauss', "diffusion"), help="Generative model to use for alignment")

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        args.n_epochs = 0

        if args.debug_mode:
            args.num_epochs_alignment = 1
            args.gr_mog_n_iters = 1
            args.gr_vae_n_iters = 10

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        backbone = Model(args, num_classes=tmp_dataset.N_CLASSES)
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # REMOVE ALL TRACK RUNNING STATS FROM CLIP
        for m in self.net.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

    def end_task(self, dataset: ContinualDataset) -> None:

        self.net.prompter.update_statistics(dataset)

        self.net.prompter.align()

        self.net.prompter.current_task += 1

    def begin_task(self, dataset: ContinualDataset) -> None:

        self.change_transform(dataset)

        self.old_epoch = 0
        self.iteration = 0

        torch.cuda.empty_cache()

    def change_transform(self, dataset: ContinualDataset) -> None:
        dataset.train_loader.dataset.transform = self.net.prompter.clip_preprocess
        dataset.test_loaders[-1].dataset.transform = self.net.prompter.clip_preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x, train=False)
        return logits[:, :self.n_seen_classes]

    def future_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.future_forward(x)

    def observe(self, *args, **kwargs):
        return 0
