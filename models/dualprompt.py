"""
DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning

Note:
    WARNING: DualPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch
from models.dualprompt_utils.model import Model

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser

from datasets import get_dataset


class DualPrompt(ContinualModel):
    NAME = 'dualprompt'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning')

        parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
        parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

        # Optimizer parameters
        parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')

        # G-Prompt parameters
        parser.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
        parser.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
        parser.add_argument('--g_prompt_layer_idx', default=[0, 1], type=int, nargs="+", help='the layer index of the G-Prompt')
        parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')

        # E-Prompt parameters
        parser.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
        parser.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

        # Use prompt pool in L2P to implement E-Prompt
        parser.add_argument('--prompt_pool', default=True, type=bool,)
        parser.add_argument('--size', default=10, type=int,)
        parser.add_argument('--length', default=5, type=int, )
        parser.add_argument('--top_k', default=1, type=int, )
        parser.add_argument('--initializer', default='uniform', type=str,)
        parser.add_argument('--prompt_key', default=True, type=bool,)
        parser.add_argument('--prompt_key_init', default='uniform', type=str)
        parser.add_argument('--use_prompt_mask', default=True, type=bool)
        parser.add_argument('--mask_first_epoch', default=False, type=bool)
        parser.add_argument('--shared_prompt_pool', default=True, type=bool)
        parser.add_argument('--shared_prompt_key', default=False, type=bool)
        parser.add_argument('--batchwise_prompt', default=True, type=bool)
        parser.add_argument('--embedding_key', default='cls', type=str)
        parser.add_argument('--predefined_key', default='', type=str)
        parser.add_argument('--pull_constraint', default=True)
        parser.add_argument('--pull_constraint_coeff', default=1.0, type=float)
        parser.add_argument('--same_key_value', default=False, type=bool)

        # ViT parameters
        parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
        parser.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
        parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')
        return parser

    def __init__(self, backbone, loss, args, transform):
        del backbone
        print("-" * 20)
        print(f"WARNING: DualPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        args.lr = args.lr * args.batch_size / 256.0
        backbone = Model(args, get_dataset(args).N_CLASSES)

        super().__init__(backbone, loss, args, transform)

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self.dataset.get_offsets(self.current_task)

        if self.current_task > 0:
            prev_start = (self.current_task - 1) * self.args.top_k
            prev_end = self.current_task * self.args.top_k

            cur_start = prev_end
            cur_end = (self.current_task + 1) * self.args.top_k

            if (prev_end > self.args.size) or (cur_end > self.args.size):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    self.net.model.e_prompt.prompt.grad.zero_()
                    self.net.model.e_prompt.prompt[cur_idx] = self.net.model.e_prompt.prompt[prev_idx]
                    self.opt.param_groups[0]['params'] = self.net.model.parameters()

        self.opt = self.get_optimizer()
        self.net.original_model.eval()

    def get_parameters(self):
        return [p for p in self.net.model.parameters() if p.requires_grad]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        log_dict = {}
        cur_lr = self.opt.param_groups[0]['lr']
        log_dict['lr'] = cur_lr

        outputs = self.net(inputs, task_id=self.current_task, train=True, return_outputs=True)
        logits = outputs['logits']

        # here is the trick to mask out classes of non-current tasks
        if self.args.train_mask:
            logits[:, :self.offset_1] = -float('inf')

        loss_clf = self.loss(logits[:, :self.offset_2], labels)
        loss = loss_clf
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            loss_pull_constraint = outputs['reduce_sim']
            loss = loss - self.args.pull_constraint_coeff * loss_pull_constraint

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.model.parameters(), self.args.clip_grad)
        self.opt.step()

        return loss.item()

    def forward(self, x):
        res = self.net(x, task_id=-1, train=False, return_outputs=True)
        logits = res['logits']
        return logits[:, :self.offset_2]
