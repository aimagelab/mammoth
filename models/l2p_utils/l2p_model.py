import os
import numpy as np
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

        if args.use_original_ckpt:
            # download ckpt from https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

            if not os.path.exists('./data/imagenet21k_ViT-B_16.npz'):
                os.system('wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P ./data/')

            lf = np.load('data/imagenet21k_ViT-B_16.npz')
            ckpt = {k: lf[k] for k in lf.files}

            def translate_name(name):
                name = name.replace('Transformer/', '')
                name = name.replace('encoderblock_', 'blocks.')
                name = name.replace('/', '.')
                name = name.replace('LayerNorm_', 'norm')
                name = name.replace('norm0', 'norm1')
                name = name.replace('MlpBlock_3', 'mlp')
                name = name.replace('Dense_0', 'fc1')
                name = name.replace('Dense_1', 'fc2')
                name = name.replace('MultiHeadDotProductAttention_1', 'attn')
                name = name.replace('kernel', 'weight')
                name = name.replace('out', 'proj')
                name = name.replace('scale', 'weight')
                name = name.replace('cls', 'cls_token')
                name = name.replace('embedding', 'patch_embed.proj')
                name = name.replace('posembed_input.pos_patch_embed.proj', 'pos_embed')
                name = name.replace('encoder_norm', 'norm')
                return name
            ckpt = {translate_name(k): v for k, v in ckpt.items()}

            for block_id in range(12):
                # convert qkv
                q = ckpt[f'blocks.{block_id}.attn.query.weight'].reshape(768, -1)
                k = ckpt[f'blocks.{block_id}.attn.key.weight'].reshape(768, -1)
                v = ckpt[f'blocks.{block_id}.attn.value.weight'].reshape(768, -1)
                qkv = np.concatenate([q, k, v], axis=1)
                ckpt[f'blocks.{block_id}.attn.qkv.weight'] = qkv
                ckpt.pop(f'blocks.{block_id}.attn.query.weight')
                ckpt.pop(f'blocks.{block_id}.attn.key.weight')
                ckpt.pop(f'blocks.{block_id}.attn.value.weight')

                q = ckpt[f'blocks.{block_id}.attn.query.bias'].reshape(-1)
                k = ckpt[f'blocks.{block_id}.attn.key.bias'].reshape(-1)
                v = ckpt[f'blocks.{block_id}.attn.value.bias'].reshape(-1)
                qkv = np.concatenate([q, k, v], axis=0)
                ckpt[f'blocks.{block_id}.attn.qkv.bias'] = qkv
                ckpt.pop(f'blocks.{block_id}.attn.query.bias')
                ckpt.pop(f'blocks.{block_id}.attn.key.bias')
                ckpt.pop(f'blocks.{block_id}.attn.value.bias')

                # permute
                ckpt[f'blocks.{block_id}.mlp.fc1.weight'] = ckpt[f'blocks.{block_id}.mlp.fc1.weight'].T
                ckpt[f'blocks.{block_id}.mlp.fc2.weight'] = ckpt[f'blocks.{block_id}.mlp.fc2.weight'].T

                ckpt[f'blocks.{block_id}.attn.qkv.weight'] = ckpt[f'blocks.{block_id}.attn.qkv.weight'].T
                ckpt[f'blocks.{block_id}.attn.proj.weight'] = ckpt[f'blocks.{block_id}.attn.proj.weight'].reshape(-1, 768).T
            ckpt['patch_embed.proj.weight'] = ckpt['patch_embed.proj.weight'].transpose(-1, -2, -4, -3)

            # remove head
            del ckpt['head.weight']
            del ckpt['head.bias']
            del ckpt['pre_logits.weight']
            del ckpt['pre_logits.bias']

            # convert to torch
            ckpt = {k: torch.from_numpy(v) for k, v in ckpt.items()}

            unexpected, missing = self.original_model.load_state_dict(ckpt, strict=False)
            assert len([x for x in missing if 'head' not in x]) == 0, f"Missing keys: {missing}"
            assert len([x for x in unexpected if 'head' not in x]) == 0, f"Unexpected keys: {unexpected}"

            # extend pos_embed for the prompts
            ckpt['pos_embed'] = torch.cat((ckpt['pos_embed'], self.model.pos_embed[:, ckpt['pos_embed'].shape[1]:]), dim=1)

            unexpected, missing = self.model.load_state_dict(ckpt, strict=False)
            assert len([x for x in missing if 'prompt' not in x and 'head' not in x]) == 0, f"Missing keys: {missing}"
            assert len([x for x in unexpected if 'prompt' not in x and 'head' not in x]) == 0, f"Unexpected keys: {unexpected}"

        if args.freeze:
            # all parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False

            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

    def forward(self, x, return_reduce_sim_loss=False):
        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=-1, cls_features=cls_features, train=self.training)

        logits = outputs['logits']
        reduce_sim = outputs['reduce_sim'] if 'reduce_sim' in outputs else None
        if return_reduce_sim_loss:
            return {'logits': logits, 'reduce_sim': reduce_sim}
        else:
            return logits
