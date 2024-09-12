# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.ring_buffer import RingBuffer

from datasets import get_dataset
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.mixup import mixup
from utils.triplet import batch_hard_triplet_loss, negative_only_triplet_loss
import torch
import torch.nn.functional as F


class Ccic(ContinualModel):
    """Continual Semi-Supervised Learning via Continual Contrastive Interpolation Consistency."""
    NAME = 'ccic'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'cssl']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(optimizer='adam')
        add_rehearsal_args(parser)

        parser.set_defaults(optimizer='adam')
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Unsupervised loss weight.')
        parser.add_argument('--knn_k', '--k', type=int, default=2, dest='knn_k',
                            help='k of kNN.')
        parser.add_argument('--memory_penalty', type=float,
                            default=1.0, help='Unsupervised penalty weight.')
        parser.add_argument('--k_aug', type=int, default=3,
                            help='Number of augumentation to compute label predictions.')
        parser.add_argument('--mixmatch_alpha', '--lamda', type=float, default=0.5, dest='mixmatch_alpha',
                            help='Regularization weight.')
        parser.add_argument('--sharp_temp', default=0.5,
                            type=float, help='Temperature for sharpening.')
        parser.add_argument('--mixup_alpha', default=0.75, type=float)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Ccic, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.embeddings = None

        self.eye = torch.eye(self.num_classes).to(self.device)
        self.sup_virtual_batch = RingBuffer(self.args.batch_size)
        self.unsup_virtual_batch = RingBuffer(self.args.batch_size)

    def get_debug_iters(self):
        """
        Returns the number of iterations to wait before logging.
        - CCIC needs a couple more iterations to initialize the KNN.
        """
        return 1000 if len(self.buffer) < self.args.buffer_size else 5

    def forward(self, x):
        if self.embeddings is None:
            with torch.no_grad():
                self.compute_embeddings()

        n_seen_classes = self.cpt * self.current_task if isinstance(self.cpt, int) else sum(self.cpt[:self.current_task])
        n_remaining_classes = self.N_CLASSES - n_seen_classes
        buf_labels = self.buffer.labels[:self.buffer.num_seen_examples]
        feats = self.net(x, returnt='features')
        feats = F.normalize(feats, p=2, dim=1)
        distances = (self.embeddings.unsqueeze(0) - feats.unsqueeze(1)).pow(2).sum(2)

        dist = torch.stack([distances[:, buf_labels == c].topk(1, largest=False)[0].mean(dim=1)
                            if (buf_labels == c).sum() > 0 else torch.zeros(x.shape[0]).to(self.device)
                            for c in range(n_seen_classes)] +
                           [torch.zeros(x.shape[0]).to(self.device)] * n_remaining_classes).T
        topkappas = self.eye[buf_labels[distances.topk(self.args.knn_k, largest=False)[1]]].sum(1)
        return topkappas - dist * 10e-6

    def end_task(self, dataset):
        self.embeddings = None

    def discard_unsupervised_labels(self, inputs, labels, not_aug_inputs):
        mask = labels != -1

        return inputs[mask], labels[mask], not_aug_inputs[mask]

    def discard_supervised_labels(self, inputs, labels, not_aug_inputs):
        mask = labels == -1

        return inputs[mask], labels[mask], not_aug_inputs[mask]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        sup_inputs, sup_labels, sup_not_aug_inputs = self.discard_unsupervised_labels(inputs, labels, not_aug_inputs)
        sup_inputs_for_buffer, sup_labels_for_buffer = sup_not_aug_inputs.clone(), sup_labels.clone()
        unsup_inputs, unsup_labels, unsup_not_aug_inputs = self.discard_supervised_labels(inputs, labels, not_aug_inputs)
        if len(sup_inputs) == 0 and self.buffer.is_empty():  # if there is no data to train on, just return 1.
            return 1.

        self.sup_virtual_batch.add_data(sup_not_aug_inputs, sup_labels)
        sup_inputs, sup_labels = self.sup_virtual_batch.get_data(self.args.batch_size, transform=self.transform, device=self.device)

        if self.current_task > 0 and unsup_not_aug_inputs.shape[0] > 0:
            self.unsup_virtual_batch.add_data(unsup_not_aug_inputs)
            unsup_inputs = self.unsup_virtual_batch.get_data(self.args.batch_size, transform=self.transform, device=self.device)[0]

        # BUFFER RETRIEVAL
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            sup_inputs = torch.cat((sup_inputs, buf_inputs))
            sup_labels = torch.cat((sup_labels, buf_labels))
            if self.current_task > 0:
                masked_buf_inputs = self.buffer.get_data(self.args.minibatch_size,
                                                         mask_task_out=self.current_task,
                                                         transform=self.transform,
                                                         cpt=self.n_classes_current_task,
                                                         device=self.device)[0]
                unsup_labels = torch.cat((torch.zeros(unsup_inputs.shape[0]).to(self.device),
                                          torch.ones(masked_buf_inputs.shape[0]).to(self.device))).long()
                unsup_inputs = torch.cat((unsup_inputs, masked_buf_inputs))

        # ------------------ K AUG ---------------------

        mask = labels != -1
        real_mask = mask[:real_batch_size]

        if (~real_mask).sum() > 0:
            unsup_aug_inputs = self.transform(not_aug_inputs[~real_mask].repeat_interleave(self.args.k_aug, 0))
        else:
            unsup_aug_inputs = torch.zeros((0,)).to(self.device)

        # ------------------ PSEUDO LABEL ---------------------

        self.net.eval()
        if len(unsup_aug_inputs):
            with torch.no_grad():
                unsup_aug_outputs = self.net(unsup_aug_inputs).reshape(self.args.k_aug, -1, self.eye.shape[0]).mean(0)
                unsup_sharp_outputs = unsup_aug_outputs ** (1 / self.args.sharp_temp)
                unsup_norm_outputs = unsup_sharp_outputs / unsup_sharp_outputs.sum(1).unsqueeze(1)
                unsup_norm_outputs = unsup_norm_outputs.repeat(self.args.k_aug, 1)
        else:
            unsup_norm_outputs = torch.zeros((0, len(self.eye))).to(self.device)
        self.net.train()

        # ------------------ MIXUP ---------------------

        self.opt.zero_grad()

        W_inputs = torch.cat((sup_inputs, unsup_aug_inputs))
        W_probs = torch.cat((self.eye[sup_labels], unsup_norm_outputs))
        perm = torch.randperm(W_inputs.shape[0])
        W_inputs, W_probs = W_inputs[perm], W_probs[perm]
        sup_shape = sup_inputs.shape[0]

        sup_mix_inputs, _ = mixup([(sup_inputs, W_inputs[:sup_shape]), (self.eye[sup_labels], W_probs[:sup_shape])], self.args.mixup_alpha)
        sup_mix_outputs = self.net(sup_mix_inputs)
        if len(unsup_aug_inputs):
            unsup_mix_inputs, _ = mixup(
                [(unsup_aug_inputs, W_inputs[sup_shape:]),
                 (unsup_norm_outputs, W_probs[sup_shape:])],
                self.args.mixup_alpha)
            unsup_mix_outputs = self.net(unsup_mix_inputs)

        effective_mbs = min(self.args.minibatch_size,
                            self.buffer.num_seen_examples)
        if effective_mbs == 0:
            effective_mbs = -self.N_CLASSES

        # ------------------ CIC LOSS ---------------------

        loss_X = 0
        if real_mask.sum() > 0:
            loss_X += self.loss(sup_mix_outputs[:-effective_mbs],
                                sup_labels[:-effective_mbs])
        if not self.buffer.is_empty():
            assert effective_mbs > 0
            loss_X += self.args.memory_penalty * self.loss(sup_mix_outputs[-effective_mbs:],
                                                           sup_labels[-effective_mbs:])

        if len(unsup_aug_inputs):
            loss_U = F.mse_loss(unsup_norm_outputs, unsup_mix_outputs) / self.eye.shape[0]
        else:
            loss_U = 0

        # CIC LOSS
        if self.current_task > 0 and epoch < self.args.n_epochs / 10 * 9:
            W_inputs = sup_inputs
            W_probs = self.eye[sup_labels]
            perm = torch.randperm(W_inputs.shape[0])
            W_inputs, W_probs = W_inputs[perm], W_probs[perm]

            sup_mix_inputs, _ = mixup([(sup_inputs, W_inputs), (self.eye[sup_labels], W_probs)], 1)
        else:
            sup_mix_inputs = sup_inputs

        # STANDARD TRIPLET
        sup_mix_embeddings = self.net.features(sup_mix_inputs)
        loss = batch_hard_triplet_loss(sup_labels, sup_mix_embeddings, self.args.batch_size // 10,
                                       margin=1, margin_type='hard')

        if loss is None:
            loss = loss_X + self.args.mixmatch_alpha * loss_U
        else:
            loss += loss_X + self.args.mixmatch_alpha * loss_U

        self.buffer.add_data(examples=sup_inputs_for_buffer,
                             labels=sup_labels_for_buffer)

        # SELF-SUPERVISED PAST TASKS NEGATIVE ONLY
        if self.current_task > 0 and epoch < self.args.n_epochs / 10 * 9:
            unsup_embeddings = self.net.features(unsup_inputs)
            loss_unsup = negative_only_triplet_loss(unsup_labels, unsup_embeddings, self.args.batch_size // 10,
                                                    margin=1, margin_type='hard')
            if loss_unsup is not None:
                loss += self.args.alpha * loss_unsup

        loss.backward()
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def compute_embeddings(self):
        """
        Computes a vector representing mean features for each class.
        """
        was_training = self.net.training
        self.net.eval()
        data = self.buffer.get_all_data(transform=self.normalization_transform)[0]
        outputs = []
        while data.shape[0] > 0:
            inputs = data[:self.args.batch_size]
            data = data[self.args.batch_size:]
            out = self.net(inputs, returnt='features')
            out = F.normalize(out, p=2, dim=1)
            outputs.append(out)

        self.embeddings = torch.cat(outputs)
        self.net.train(was_training)
