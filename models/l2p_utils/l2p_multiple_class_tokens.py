import torch
import torch.nn as nn
from timm import create_model
import lifelong_methods.methods.l2p_utils.vit_prompt_multiple_class_tokens

class L2PModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config['freeze'] = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
        n_classes = self.config['num_classes']

        self.model_name = f'{self.config["network"]}_l2p'

        self.original_model = create_model(
            self.model_name,
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
            
            # MUST CREATE A MODEL WITH 1 CLS TOKEN
            many_cltoks=False
        )
        self.original_model.eval()

        self.model = create_model(
            self.model_name,
            pretrained=True,
            num_classes=n_classes,
            prompt_length=5,
            embedding_key='cls',
            prompt_init='uniform',
            prompt_pool=True,
            prompt_key=True,
            pool_size=self.config['pool_size_l2p'],
            top_k=5,
            batchwise_prompt=True,
            prompt_key_init='uniform',
            head_type=config['l2p_head_type'],
            use_prompt_mask=False,
            
            # THIS SHOULD HAVE 115 CLS TOKENS
            many_cltoks=True
        )

        if self.config['freeze']:
            # all parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False
            
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.model.named_parameters():
                if n.startswith(tuple(self.config['freeze'])):
                    p.requires_grad = False

    
    def forward(self, x, return_outputs=False):
        # x = transforms.Resize(224)(x)
        # if x.shape[1] != 3:
        #     x = x.expand(x.shape[0], 3, 224, 224)

        with torch.no_grad():
            if self.original_model is not None:
                # Si usa uin modello preaddestrato e congelato per estrarre un vettore di feature
                # per ciascun input. Viene estratto il feature vector che corrisponde al token [class]
                # Questo feature vector viene utilizzato come query, quindi in seguito si misura la similarità
                # tra questo vettore e i vettori delle prompt keys. 
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