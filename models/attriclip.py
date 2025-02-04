"""
DISCLAIMER: AttriCLIP **does not** reproduce the results in the paper (https://arxiv.org/pdf/2305.11488).
Unfortunately, the original implementation (https://github.com/bhrqw/AttriCLIP) did not reproduced the results either and is no longer available. This is a known issue (see https://github.com/bhrqw/SADA/issues/3).

This implementation is based on that code and on the information provided in the paper.
"""

from utils.args import *
from models.utils.continual_model import ContinualModel

from datasets import get_dataset
from models.attriclip_utils.model import CoOp
from models.attriclip_utils.utils import cosine_loss
from utils.conf import get_device


class Attriclip(ContinualModel):
    """Continual Learning via Progressive Neural Networks."""
    NAME = 'attriclip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
        parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
        parser.add_argument('--freeze_clip', type=int, default=1, help='freeze_clip')
        parser.add_argument('--matching_loss_lambda', type=float, default=0.7, help='lambda_k in the main paper')  # 0.5 in the original code
        parser.add_argument('--orthogonalize_loss_lambda', type=float, default=0.3, help='lambda_p in the main paper')  # 0.1 in the original code
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert args.lr_scheduler is None, 'Attriclip does not support lr_scheduler, uses a custom one.'
        seq_dataset = get_dataset(args) if dataset is None else dataset
        self.device = get_device()
        self.class_names = seq_dataset.get_class_names()
        backbone = CoOp(self.device, False, False, args)
        offset_1, offset_2 = seq_dataset.get_offsets(0)
        cur_class_names = self.class_names[offset_1:offset_2]
        backbone.init_model(class_names=cur_class_names, text_key=backbone.text_key, text_prompt=backbone.text_prompt)
        super().__init__(backbone, loss, args, transform, dataset=dataset)

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self.dataset.get_offsets(self.current_task)
        self.per_epoch_steps = len(dataset.train_loader)
        cur_class_names = self.class_names[self.offset_1:self.offset_2]
        self.net.init_model(class_names=cur_class_names, text_key=self.net.text_key, text_prompt=self.net.text_prompt)
        self.opt, self.custom_scheduler = self.net.get_optimizer(self.per_epoch_steps)
        self.net.model.eval()
        self.old_epoch = 0
        self.idx = 0
        self.iteration = 0
        self.opt.zero_grad()

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        if self.old_epoch != epoch:
            self.idx = 0
            self.old_epoch = epoch
        labels = labels.long()

        lr = self.opt.param_groups[0]['lr']

        cur_iter_idx = epoch * self.per_epoch_steps + self.idx
        self.custom_scheduler.step(cur_iter_idx)

        output, ima_feat, key_choose, loss_m = self.net.model(inputs)
        loss_main = self.loss(output, labels - self.offset_1)
        loss_k = cosine_loss(ima_feat, key_choose)
        loss = loss_main + self.args.matching_loss_lambda * loss_k + self.args.orthogonalize_loss_lambda * loss_m

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.idx += 1
        self.iteration += 1

        return {'loss': loss.item(), 'lr': lr}

    def forward(self, x):
        test_classes = self.class_names[:self.offset_2]
        logits = self.net.model(x, test_classes, test=True)
        return logits[:, :self.offset_2]
