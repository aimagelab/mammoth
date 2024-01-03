"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from timm import create_model  # noqa
from models.l2p_utils.l2p_model import L2PModel


class L2P(ContinualModel):
    NAME = 'l2p'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Learning to Prompt (L2P)')
        # Prompt parameters
        parser.add_argument('--prompt_pool', default=True, type=bool,)
        parser.add_argument('--pool_size_l2p', default=10, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--length', default=5, type=int, help='length of prompt (L_p in paper)')
        parser.add_argument('--top_k', default=5, type=int, help='top k prompts to use (N in paper)')
        parser.add_argument('--prompt_key', default=True, type=bool, help='Use learnable prompt key')
        parser.add_argument('--prompt_key_init', default='uniform', type=str, help='initialization type for key\'s prompts')
        parser.add_argument('--use_prompt_mask', default=False, type=bool)
        parser.add_argument('--batchwise_prompt', default=True, type=bool)
        parser.add_argument('--embedding_key', default='cls', type=str)
        parser.add_argument('--predefined_key', default='', type=str)
        parser.add_argument('--pull_constraint', default=True)
        parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

        # Learning rate schedule parameters
        parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
        parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
        parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
        parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
        parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
        parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
        parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
        parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
        parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

        parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        L2P re-defines the backbone model to include the prompt parameters. This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        del backbone
        print("-" * 20)
        print(f"WARNING: L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        args.lr = args.lr * args.batch_size / 256.0
        backbone = L2PModel(args)

        super().__init__(backbone, loss, args, transform)

    def begin_task(self, dataset):
        self.net.original_model.eval()

        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        outputs = self.net(inputs, return_outputs=True)
        logits = outputs['logits']

        # here is the trick to mask out classes of non-current tasks
        offset_1, offset_2 = self._compute_offsets(self.current_task)
        logits[:, :offset_1] = -float('inf')

        loss = self.loss(logits[:, :offset_2], labels)
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            loss = loss - self.args.pull_constraint_coeff * outputs['reduce_sim']

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()

    def get_parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]

    def forward(self, x):
        if self.current_task > 0:
            _, offset_2 = self._compute_offsets(self.current_task - 1)
        else:
            offset_2 = self.N_CLASSES
        return self.net(x)[:, :offset_2]
