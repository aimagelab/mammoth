"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import logging
import torch

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import ArgumentParser
from timm import create_model  # noqa
from models.l2p_utils.l2p_model import L2PModel


class L2P(ContinualModel):
    """Learning to Prompt (L2P)."""
    NAME = 'l2p'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
        # Prompt parameters
        parser.add_argument('--prompt_pool', default=True, type=bool,)
        parser.add_argument('--pool_size_l2p', default=10, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--length', default=5, type=int, help='length of prompt (L_p in paper)')
        parser.add_argument('--top_k', default=5, type=int, help='top k prompts to use (N in paper)')
        parser.add_argument('--prompt_key', default=True, type=bool, help='Use learnable prompt key')
        parser.add_argument('--prompt_key_init', default='uniform', type=str, help='initialization type for key\'s prompts')
        parser.add_argument('--use_prompt_mask', default=False, type=bool)
        parser.add_argument('--batchwise_prompt', default=0, type=binary_to_boolean_type,
                            help='Use batch-wise prompting (i.e., majority voting) during test? NOTE: this may lead to unfair comparison with other methods.')
        parser.add_argument('--embedding_key', default='cls', type=str)
        parser.add_argument('--predefined_key', default='', type=str)
        parser.add_argument('--pull_constraint', default=True)
        parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

        # Learning rate schedule parameters
        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')

        parser.add_argument('--use_original_ckpt', type=binary_to_boolean_type, default=0, help='Use original checkpoint from `https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz`')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        L2P re-defines the backbone model to include the prompt parameters.
        This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        if args.batchwise_prompt:
            logging.warning("Using batch-wise prompting (i.e., majority voting) during test may lead to unfair comparison with other methods.")

        del backbone
        print("-" * 20)
        print(f"WARNING: L2P USES A CUSTOM BACKBONE: `https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz` (vit_base_patch16_224_in21k_fn_in1k_old).")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        args.lr = args.lr * args.batch_size / 256.0  # scale learning rate by batch size
        backbone = L2PModel(args)

        super().__init__(backbone, loss, args, transform, dataset=dataset)

    def begin_task(self, dataset):
        self.net.original_model.eval()

        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        outputs = self.net(inputs, return_reduce_sim_loss=True)
        logits = outputs['logits']
        reduce_sim = outputs['reduce_sim']

        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint and reduce_sim is not None:
            loss = loss - self.args.pull_constraint_coeff * reduce_sim.mean()  # the mean is needed for data-parallel (concatenates instead of averaging)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()

    def get_parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]

    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
