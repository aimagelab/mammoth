import torch
from models.twf_utils.utils import init_twf
from utils.augmentations import CustomRandomCrop, CustomRandomHorizontalFlip, DoubleCompose, DoubleTransform
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

from torchvision import transforms
import torch.nn.functional as F


def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))


def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    # parser.add_argument('--pre_epochs', type=int, default=200,
    #                     help='pretrain_epochs.')
    # parser.add_argument('--pre_dataset', type=str, required=True,
    #                     choices=['cifar100', 'tinyimgR', 'imagenet'])
    # parser.add_argument('--stop_after_prep', action='store_true')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Double-branch distillation + inter-branch skip attention')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)

    # Griddable parameters
    parser.add_argument('--der_alpha', type=float, required=True,
                        help='Distillation alpha hyperparameter for student stream.')
    parser.add_argument('--der_beta', type=float, required=True,
                        help='Distillation beta hyperparameter.')
    parser.add_argument('--lambda_fp', type=float, required=True,
                        help='weight of feature propagation loss replay')
    parser.add_argument('--lambda_diverse_loss', type=float, required=False, default=0,
                        help='Diverse loss hyperparameter.')
    parser.add_argument('--lambda_fp_replay', type=float, required=False, default=0,
                        help='weight of feature propagation loss replay')
    parser.add_argument('--resize_maps', type=int, required=False, choices=[0, 1], default=0,
                        help='Apply downscale and upscale to feature maps before save in buffer?')
    parser.add_argument('--min_resize_threshold', type=int, required=False, default=16,
                        help='Min size of feature maps to be resized?')

    return parser


class TwF(ContinualModel):

    NAME = 'twf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        assert "resnet" in str(type(backbone)).lower(), "Only resnet is supported for TwF"

        super().__init__(
            backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)
        self.buf_transform = self.get_custom_double_transform(self.transform.transforms)

        ds = get_dataset(args)
        self.cpt = ds.N_CLASSES_PER_TASK
        self.not_aug_transform = transforms.Compose([transforms.ToPILImage(), ds.TEST_TRANSFORM]) if hasattr(ds, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), ds.get_normalization_transform()])
        self.num_classes = self.N_TASKS * self.cpt

        self.task = 0

    def get_custom_double_transform(self, transform):
        tfs = []
        for tf in transform:
            if isinstance(tf, transforms.RandomCrop):
                tfs.append(CustomRandomCrop(tf.size, tf.padding, resize=self.args.resize_maps == 1, min_resize_index=2))
            elif isinstance(tf, transforms.RandomHorizontalFlip):
                tfs.append(CustomRandomHorizontalFlip(tf.p))
            elif isinstance(tf, transforms.Compose):
                tfs.append(DoubleCompose(
                    self.get_custom_double_transform(tf.transforms)))
            else:
                tfs.append(DoubleTransform(tf))

        return DoubleCompose(tfs)

    def end_task(self, dataset):
        self.net.eval()

        with torch.no_grad():
            # loop over buffer, recompute attention maps and save them
            for buf_idxs in batch_iterate(len(self.buffer), self.args.batch_size):

                buf_labels = self.buffer.labels[buf_idxs].to(self.device)

                buf_mask = torch.div(buf_labels, self.cpt,
                                     rounding_mode='floor') == self.task

                if not buf_mask.any():
                    continue

                buf_inputs = self.buffer.examples[buf_idxs].to(self.device)[buf_mask]
                buf_labels = buf_labels[buf_mask]
                buf_inputs = torch.stack([self.not_aug_transform(
                    ee.cpu()) for ee in buf_inputs]).to(self.device)

                _, buf_partial_features = self.net(buf_inputs, returnt='full')
                pret_buf_partial_features = self.teacher(buf_inputs)

                _, attention_masks = self.partial_distill_loss(buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels)

                for idx in buf_idxs:
                    self.buffer.attention_maps[idx] = [
                        at[idx % len(at)].to(self.device) for at in attention_masks]

        self.net.train()  # TODO: check
        self.task += 1

    def begin_task(self, dataset):

        if self.task == 0 or ("start_from" in self.args and self.args.start_from is not None and self.task == self.args.start_from):
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
        labels = labels.long()

        B = len(inputs)
        all_labels = labels

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

        self.opt.zero_grad()

        loss = self.loss(
            stream_logits[:, self.task * self.cpt:(self.task + 1) * self.cpt], labels % self.cpt)

        loss_er = torch.tensor(0.)
        loss_der = torch.tensor(0.)
        loss_afd = torch.tensor(0.)

        if len(self.buffer) == 0:
            loss_afd, stream_attention_maps = self.partial_distill_loss(
                stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, labels)
        else:
            buffer_teacher_forcing = torch.div(
                buf_labels, self.cpt, rounding_mode='floor') != self.task
            teacher_forcing = torch.cat(
                (torch.zeros((B)).bool().to(self.device), buffer_teacher_forcing))
            attention_maps = [
                [torch.ones_like(map) for map in buf_attention_maps[0]]] * B + buf_attention_maps

            loss_afd, all_attention_maps = self.partial_distill_loss(all_partial_features[-len(
                all_pret_partial_features):], all_pret_partial_features, all_labels,
                teacher_forcing, attention_maps)

            stream_attention_maps = [ap[:B] for ap in all_attention_maps]

            loss_er = self.loss(buf_outputs[:, :(self.task + 1) * self.cpt], buf_labels)

            loss_der = F.mse_loss(buf_outputs, buf_logits)

        loss += self.args.der_beta * loss_er
        loss += self.args.der_alpha * loss_der
        loss += self.args.lambda_fp * loss_afd

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=stream_logits.data,
                             attention_maps=stream_attention_maps)

        return loss.item()
