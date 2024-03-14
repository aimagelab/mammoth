import torch
import torch.nn as nn
from datasets import get_dataset
from models.l2p_utils.vit_prompt import vit_base_patch16_224_l2p


class L2PModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        dataset = get_dataset(args)
        n_classes = dataset.N_CLASSES

        self.original_model = vit_base_patch16_224_l2p(
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        self.original_model.eval()

        self.model = vit_base_patch16_224_l2p(
            pretrained=True,
            num_classes=n_classes,
            prompt_length=args.length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=args.prompt_pool,
            prompt_key=args.prompt_key,
            pool_size=args.pool_size_l2p,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=args.use_prompt_mask,
        )

        if args.freeze:
            # all parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False

            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

    def forward(self, x, return_outputs=False):
        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=-1, cls_features=cls_features, train=self.training)
        # self.prompt_idx = outputs['prompt_idx']
        # print(self.prompt_idx)
        logits = outputs['logits']
        if return_outputs:
            return outputs
        else:
            return logits
