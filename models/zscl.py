"""
Implementation of ZSCL from the ICCV 2023 paper "Preventing zero-shot transfer degradation in continual learning of vision-language models"
Paper: https://arxiv.org/abs/2303.06628
Original code: https://github.com/Thunderbeee/ZSCL
"""

import copy
import logging
from pathlib import Path

import models.zscl_utils.clip as clip
import torch
from models.utils.future_model import FutureModel
from models.zscl_utils import conceptual_captions
from utils.schedulers import CosineSchedulerWithLinearWarmup
from torch.nn import functional as F

from utils.args import ArgumentParser
from utils.conf import base_path


class ZSCL(FutureModel):
    """ZSCL -- Preventing zero-shot transfer degradation in continual learning of vision-language models."""
    NAME = 'zscl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:

        parser.set_defaults(n_epochs=1, lr=1e-5, batch_size=8, eval_future=1, optimizer='adamw', scheduler_mode='iter')

        parser.add_argument("--clip_backbone", type=str, default='ViT-L/14', choices=['ViT-B/16', 'ViT-L/14'], help="Clip backbone")
        parser.add_argument("--prompt_template", type=str, default='a good photo of a {}.', help="Template string")

        parser.add_argument('--we', type=int, default=1, help='Whether to use weight averaging')
        parser.add_argument('--avg_freq', type=int, default=100, help='Frequency of weight averaging')
        parser.add_argument('--ls', type=float, default=0.2, help='Label smoothing')


        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        assert args.scheduler_mode == 'iter', "ZSCL only supports 'iter' scheduler mode."
        if args.lr_scheduler is not None:
            logging.warning("ZSCL does not require a learning rate scheduler and will use a custom one.")

        if args.optimizer != 'adamw':
            logging.warning("ZSCL should use AdamW optimizer.")

        logging.info("ZSCL redefines the tokenizer of CLIP. Check out the changes in models/zscl_utils/clip/tokenizer.py .")

        del backbone
        logging.info("ZSCL will override the backbone model.")
        super().__init__(None, loss, args, transform, dataset)
        self.model, self.transforms, self.clip_preprocess = clip.load(args.clip_backbone, device=self.device, jit=False)
        self.net = self.model
        self.current_class_names = []
        self.text_tokens = None

        self.classes_names = self.dataset.get_class_names()
        self.tot_text_tokens = clip.tokenize(
            [self.args.prompt_template.format(c) for c in self.classes_names]
        ).to(self.device)

        self.ref_model, _, self.test_preprocess = clip.load(args.clip_backbone, device=self.device, jit=False)

        assert Path(f"{base_path()}/conceptual_captions").exists(), f"Conceptual Captions dataset not found. Please follow the steps at https://github.com/Thunderbeee/ZSCL/blob/main/mtil/datasets.md (gather_cc.py is within models/zscl_utils) and put it in {base_path()}."
        self.ref_dataset = conceptual_captions(
            self.test_preprocess,
            location=f"{base_path()}/conceptual_captions",
            batch_size=self.args.batch_size,
        )
        self.ref_texts = self.ref_dataset.train_dataset.captions
        self.ref_texts = clip.tokenize(self.ref_texts).to(self.device)
        self.ref_model.eval()
        with torch.no_grad():
            step = 1000
            ref_embeddings = torch.cat(
                [self.ref_model(None, self.ref_texts[i:i + step]) for i in range(0, len(self.ref_texts), step)])
            self.ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)


    def get_parameters(self):
        exclude_params_name = ["logit_scale"]
        return [v for k, v in self.model.named_parameters() if k not in exclude_params_name]

    def get_optimizer(self):
        return super().get_optimizer()

    def begin_task(self, dataset):
        self.change_transform(dataset)

        self.current_class_names = self.classes_names[:self.n_seen_classes]
        self.text_tokens = self.tot_text_tokens[:self.n_seen_classes]

        self.logit_scale = self.model.logit_scale
        self.opt = self.get_optimizer()

        num_batches = len(dataset.train_loader)
        total_iterations = self.args.n_epochs * num_batches
        self.scheduler = CosineSchedulerWithLinearWarmup(self.opt, self.args.lr, 30, total_iterations)

        if self.args.we:
            self.we_model = copy.deepcopy(self.model).to(self.device)
            self.we_n = 0
        self.texts = self.tot_text_tokens[self.n_past_classes:self.n_seen_classes]

        self.ref_iter = iter(self.ref_dataset)
        self.ref_model.eval()
        self.model.train()

    def change_transform(self, dataset):
        dataset.train_loader.dataset.transform = self.clip_preprocess
        dataset.test_loaders[-1].dataset.transform = self.clip_preprocess

    def end_task(self, dataset):
        if self.args.we:
            for param_q, param_k in zip(self.model.parameters(), self.we_model.parameters()):
                param_q.data = param_k.data
        self.model.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        logits_per_image, _ = self.model(inputs, self.texts)
        loss = F.cross_entropy(logits_per_image, labels - self.n_past_classes, label_smoothing=self.args.ls)

        try:
            ref_images, ref_labels = next(self.ref_iter)
        except:
            ref_iter = iter(self.ref_dataset.train_loader)
            ref_images, ref_labels = next(ref_iter)
        ref_images, ref_labels = ref_images.to(self.device), ref_labels.to(self.device)

        with torch.no_grad():
            # -- get ref image embedding --
            ref_out = self.ref_model(ref_images, None)
            ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        ref_out_current = self.model(ref_images, None)
        ref_out_current = ref_out_current / ref_out_current.norm(dim=-1, keepdim=True)
        # -- image_loss --
        logits_current = self.logit_scale.exp() * ref_out_current @ self.ref_embeddings.t()
        logits_ref = self.logit_scale.exp() * ref_out @ self.ref_embeddings.t()
        loss_ZSCL = self.distillation(logits_ref, logits_current, T=2)
        # -- text_loss --
        logits_current_2 = logits_current.t()
        logits_ref_2 = logits_ref.t()
        loss_ZSCL_2 = self.distillation(logits_ref_2, logits_current_2, T=2)
        # -- final loss --
        loss = loss + 5 * loss_ZSCL + 5 * loss_ZSCL_2

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.args.we and self.task_iteration % self.args.avg_freq == 0:
            self.we_n += 1
            self.merge_we(self.model, self.we_model, self.we_n)

        return loss.item()

    @torch.no_grad()
    def forward(self, image):
        logits_per_image, _ = self.model(image, self.text_tokens)
        probs = logits_per_image.softmax(dim=-1)
        return probs

    @torch.no_grad()
    def future_forward(self, image):
        logits_per_image, _ = self.model(image, self.tot_text_tokens)
        probs = logits_per_image.softmax(dim=-1)
        return probs

    def distillation(self, t, s, T=2):
        p = F.softmax(t / T, dim=1)
        loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
        return loss

    def merge_we(self, model_0, model_1, sma_count):
        for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
            param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
        return model_1
