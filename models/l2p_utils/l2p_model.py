import torch
import torch.nn as nn
from timm import create_model

class L2PModel(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.args = args
        self.n_classes = n_classes

        self.model_name = f'{self.args.network}_l2p'
        self.original_model = create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.n_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
        self.original_model.eval()

        self.model = create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.n_classes,
            prompt_length=self.args.length,
            embedding_key=self.args.embedding_key,
            prompt_init=self.args.prompt_key_init,
            prompt_pool=self.args.prompt_pool,
            prompt_key=self.args.prompt_key,
            pool_size=self.args.pool_size_l2p,
            top_k=self.args.top_k,
            batchwise_prompt=self.args.batchwise_prompt,
            prompt_key_init=self.args.prompt_key_init,
            head_type=self.args.head_type,
            use_prompt_mask=self.args.use_prompt_mask,
        )

        if self.args.freeze:
            # all parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False
            
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.model.named_parameters():
                if n.startswith(tuple(self.args.freeze)):
                    p.requires_grad = False

    
    def forward(self, x, return_outputs=False):
        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=-1, cls_features=cls_features, train=self.training)
        #self.prompt_idx = outputs['prompt_idx']
        #print(self.prompt_idx)
        logits = outputs['logits']
        if return_outputs:
            return outputs
        else:
            return logits