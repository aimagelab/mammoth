from argparse import Namespace
import torch
from torch import nn
from models.dualprompt_utils.vision_transformer import vit_base_patch16_224_dualprompt


class Model(nn.Module):
    def __init__(self, args: Namespace, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        self.original_model = vit_base_patch16_224_dualprompt(
            pretrained=args.pretrained,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
        )
        self.original_model.eval()

        self.model = vit_base_patch16_224_dualprompt(
            pretrained=args.pretrained,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
            prompt_length=args.length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=True,
            prompt_key=True,
            pool_size=args.size,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=True,
            use_g_prompt=True,
            g_prompt_length=args.g_prompt_length,
            g_prompt_layer_idx=args.g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
            use_e_prompt=True,
            e_prompt_layer_idx=args.e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
            same_key_value=args.same_key_value,
        )

        if args.freeze:
            for p in self.original_model.parameters():
                p.requires_grad = False

            for n, p in self.model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

    def forward(self, x, task_id, train=False, return_outputs=False):

        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=task_id, cls_features=cls_features, train=train)

        if return_outputs:
            return outputs
        else:
            return outputs['logits']
