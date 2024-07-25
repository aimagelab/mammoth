

from utils.args import *
from models.utils.continual_model import ContinualModel

from datasets import get_dataset
import wandb
from models.attriclip_utils.model import CoOp
from models.attriclip_utils.utils import cosine_loss
from utils.conf import get_device


class Attriclip(ContinualModel):
    NAME = 'attriclip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual Learning via'
                                            ' Progressive Neural Networks.')
        parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
        parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
        parser.add_argument('--freeze_clip', type=int, default=1, help='freeze_clip')
        parser.add_argument("--virtual_bs_n", type=int, default=1, help="virtual batch size iterations")
        return parser

    def __init__(self, backbone, loss, args, transform):
        self.seq_dataset = get_dataset(args)
        self.device = get_device()
        self.class_names = self.seq_dataset.get_class_names()
        backbone = CoOp(self.device, False, False, args)
        offset_1, offset_2 = self.seq_dataset.get_offsets(0)
        cur_class_names = self.class_names[offset_1:offset_2]
        backbone.init_model(class_names=cur_class_names, text_key=backbone.text_key, text_prompt=backbone.text_prompt)
        super().__init__(backbone, loss, args, transform)

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self.seq_dataset.get_offsets(self.current_task)
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

        log_dict = {}
        log_dict['lr'] = self.opt.param_groups[0]['lr']

        cur_iter_idx = epoch * self.per_epoch_steps + self.idx
        self.custom_scheduler.step(cur_iter_idx)

        output, ima_feat, key_choose, loss_m = self.net.model(inputs)
        loss_main = self.loss(output, labels - self.offset_1)
        loss_k = cosine_loss(ima_feat, key_choose)
        loss = loss_main + 0.7 * loss_k + 0.3 * loss_m

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.idx += 1
        self.iteration += 1

        if not self.args.nowand:
            wandb.log(log_dict)

        return loss.item()

    def forward(self, x):
        test_classes = self.class_names[:self.offset_2]
        logits = self.net.model(x, test_classes, test=True)
        return logits[:, :self.offset_2]
