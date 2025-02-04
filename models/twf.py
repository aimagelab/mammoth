import logging
import torch
from models.twf_utils.utils import init_twf
from utils import binary_to_boolean_type
from utils.augmentations import CustomRandomCrop, CustomRandomHorizontalFlip, DoubleCompose, DoubleTransform, apply_transform
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

from torchvision import transforms
import torch.nn.functional as F

from utils.kornia_utils import KorniaMultiAug


def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))


class TwF(ContinualModel):
    """Transfer without Forgetting: double-branch distillation + inter-branch skip attention."""

    NAME = 'twf'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # Griddable parameters
        parser.add_argument('--der_alpha', type=float, required=True,
                            help='Distillation alpha hyperparameter for student stream (`alpha` in the paper).')
        parser.add_argument('--der_beta', type=float, required=True,
                            help='Distillation beta hyperparameter (`beta` in the paper).')
        parser.add_argument('--lambda_fp', type=float, required=True,
                            help='weight of feature propagation loss replay')
        parser.add_argument('--lambda_diverse_loss', type=float, default=0,
                            help='Diverse loss hyperparameter.')
        parser.add_argument('--lambda_fp_replay', type=float, default=0,
                            help='weight of feature propagation loss replay')
        parser.add_argument('--resize_maps', type=binary_to_boolean_type, default=1,
                            help='Apply downscale and upscale to feature maps before save in buffer?')
        parser.add_argument('--min_resize_threshold', type=int, default=16,
                            help='Min size of feature maps to be resized?')
        parser.add_argument('--virtual_bs_iterations', type=int, default=1, help="virtual batch size iterations")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert "resnet" in str(type(backbone)).lower(), "Only resnet is supported for TwF"

        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.buf_transform = self.get_custom_double_transform(self.original_transform.transforms)

        if self.args.loadcheck is None:
            logging.warning("no checkpoint loaded!")

        if self.args.lambda_fp_replay == 0:
            logging.warning('lambda_fp_replay is 0, so no replay of attention masks will be used')

        if self.args.lambda_diverse_loss == 0:
            logging.warning('lambda_diverse_loss is 0, so no diverse loss will be used')

    def get_custom_double_transform(self, transform):
        tfs = []
        for tf in transform:
            if isinstance(tf, transforms.RandomCrop):
                tfs.append(CustomRandomCrop(tf.size, tf.padding, resize=self.args.resize_maps, min_resize_index=2))
            elif isinstance(tf, transforms.RandomHorizontalFlip):
                tfs.append(CustomRandomHorizontalFlip(tf.p))
            elif isinstance(tf, transforms.Compose):
                tfs.append(DoubleCompose(
                    self.get_custom_double_transform(tf.transforms)))
            else:
                tfs.append(DoubleTransform(tf))

        return DoubleCompose(tfs)

    def end_task(self, dataset):
        self.opt.zero_grad(set_to_none=True)
        delattr(self, 'opt')

        self.net.eval()

        torch.cuda.empty_cache()

        with torch.no_grad():
            # loop over buffer, recompute attention maps and save them
            for buf_idxs in batch_iterate(len(self.buffer), self.args.batch_size):

                buf_labels = self.buffer.labels[buf_idxs].to(self.device)

                buf_mask = torch.div(buf_labels, self.n_classes_current_task,
                                     rounding_mode='floor') == self.current_task

                if not buf_mask.any():
                    continue

                buf_inputs = self.buffer.examples[buf_idxs].to(self.device)[buf_mask]
                buf_labels = buf_labels[buf_mask]
                buf_inputs = apply_transform(buf_inputs, self.normalization_transform).to(self.device)

                if len(buf_inputs) < torch.cuda.device_count():
                    continue

                _, buf_partial_features = self.net(buf_inputs, returnt='full')
                pret_buf_partial_features = self.teacher(buf_inputs)

                _, attention_masks = self.partial_distill_loss(buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels)

                for idx in buf_idxs:
                    self.buffer.attention_maps[idx] = [
                        at[idx % len(at)].to(self.device) for at in attention_masks]

        self.net.train()
        self.opt = self.get_optimizer()

    def begin_task(self, dataset):

        if self.current_task == 0 or ("start_from" in self.args and self.args.start_from is not None and self.current_task == self.args.start_from):
            init_twf(self, dataset)

            self.opt = self.get_optimizer()
            self.net.train()

    def partial_distill_loss(self, net_partial_features: list, pret_partial_features: list,
                             targets, teacher_forcing: list = None, extern_attention_maps: list = None):

        assert len(net_partial_features) == len(
            pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

        if teacher_forcing is None or extern_attention_maps is None:
            assert teacher_forcing is None
            assert extern_attention_maps is None

        loss = 0
        attention_maps = []

        torch.cuda.empty_cache()

        for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
            assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

            adapter = getattr(
                self.net, f"adapter_{i+1}")

            pret_feat = pret_feat.detach()

            if teacher_forcing is None:
                curr_teacher_forcing = torch.zeros(
                    len(net_feat,)).bool().to(self.device)
                curr_ext_attention_map = torch.ones(
                    (len(net_feat), adapter.c)).to(self.device)
            else:
                curr_teacher_forcing = teacher_forcing
                curr_ext_attention_map = torch.stack(
                    [b[i] for b in extern_attention_maps], dim=0).float()

            adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                                  teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

            loss += adapt_loss
            attention_maps.append(adapt_attention.detach().cpu().clone().data)

        return loss / (i + 1), attention_maps

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        B = len(inputs)
        if len(inputs) < torch.cuda.device_count():
            return 0

        labels = labels.long()

        B = len(inputs)
        all_labels = labels

        with torch.no_grad():
            if len(self.buffer) > 0:
                # sample from buffer
                buf_choices, buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=None, return_index=True)
                buf_attention_maps = [self.buffer.attention_maps[c]
                                      for c in buf_choices]
                d = [self.buf_transform(ee, attn_map) for ee, attn_map in zip(buf_inputs, buf_attention_maps)]
                buf_inputs, buf_attention_maps = torch.stack(
                    [v[0] for v in d]).to(self.device), [[o.to(self.device) for o in v[1]] for v in d]
                buf_logits = buf_logits.to(self.device)
                buf_labels = buf_labels.to(self.device)

                inputs = torch.cat([inputs, buf_inputs])
                all_labels = torch.cat([labels, buf_labels])

        all_logits, all_partial_features = self.net(inputs, returnt='full')
        with torch.no_grad():
            all_pret_partial_features = self.teacher(inputs)

        stream_logits, buf_outputs = all_logits[:B], all_logits[B:]
        stream_partial_features = [p[:B] for p in all_partial_features]
        stream_pret_partial_features = [p[:B] for p in all_pret_partial_features]

        loss = self.loss(
            stream_logits[:, self.n_past_classes:self.n_seen_classes], labels - self.n_past_classes)

        loss_er = torch.tensor(0.)
        loss_der = torch.tensor(0.)
        loss_afd = torch.tensor(0.)

        torch.cuda.empty_cache()
        if len(self.buffer) == 0:
            loss_afd, stream_attention_maps = self.partial_distill_loss(
                stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, labels)
        else:
            buffer_teacher_forcing = torch.div(
                buf_labels, self.n_classes_current_task, rounding_mode='floor') != self.current_task
            teacher_forcing = torch.cat(
                (torch.zeros((B)).bool().to(self.device), buffer_teacher_forcing))
            attention_maps = [
                [torch.ones_like(map) for map in buf_attention_maps[0]]] * B + buf_attention_maps

            loss_afd, all_attention_maps = self.partial_distill_loss(all_partial_features[-len(
                all_pret_partial_features):], all_pret_partial_features, all_labels,
                teacher_forcing, attention_maps)

            stream_attention_maps = [ap[:B] for ap in all_attention_maps]

            loss_er = self.loss(buf_outputs[:, :self.n_seen_classes], buf_labels)

            loss_der = F.mse_loss(buf_outputs, buf_logits)

        loss += self.args.der_beta * loss_er
        loss += self.args.der_alpha * loss_der
        loss += self.args.lambda_fp * loss_afd

        if self.task_iteration == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        loss.backward()
        if self.task_iteration > 0 and self.task_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=stream_logits.data,
                             attention_maps=stream_attention_maps)

        return loss.item()
