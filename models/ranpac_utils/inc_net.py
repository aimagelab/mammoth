from argparse import Namespace
import copy
from functools import partial
import math
import torch
from torch import nn
import torch.nn.functional as F
from backbone import MammothBackbone
from backbone.vit import vit_base_patch16_224_prompt_prototype
from models.ranpac_utils.vit import VisionTransformer


class RanPACNet(MammothBackbone):

    def __init__(self, backbone: MammothBackbone):
        super(RanPACNet, self).__init__()

        self.fc = None
        self.device = backbone.device

        tuning_config = Namespace(ffn_adapt=True,
                                  ffn_option="parallel",
                                  ffn_adapter_layernorm_option="none",
                                  ffn_adapter_init_option="lora",
                                  ffn_adapter_scalar="0.1",
                                  ffn_num=64,
                                  d_model=768,
                                  vpt_on=False,
                                  vpt_num=0,
                                  )

        self.convnet = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                         norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)

        backbone.head = nn.Identity()
        state_dict = backbone.state_dict()

        for key in list(state_dict.keys()):
            if 'qkv.weight' in key:
                qkv_weight = state_dict.pop(key)
                q_weight = qkv_weight[:768]
                k_weight = qkv_weight[768:768 * 2]
                v_weight = qkv_weight[768 * 2:]
                state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
                state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
                state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
            elif 'qkv.bias' in key:
                qkv_bias = state_dict.pop(key)
                q_bias = qkv_bias[:768]
                k_bias = qkv_bias[768:768 * 2]
                v_bias = qkv_bias[768 * 2:]
                state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
                state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
                state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
        # second, modify the mlp.fc.weight to match fc.weight
        for key in list(state_dict.keys()):
            if 'mlp.fc' in key:
                fc_weight = state_dict.pop(key)
                state_dict[key.replace('mlp.', '')] = fc_weight

        missing, unexpected = self.convnet.load_state_dict(state_dict, strict=False)
        assert len([m for m in missing if 'adaptmlp' not in m]) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        # freeze all but the adapter
        for name, p in self.convnet.named_parameters():
            if name in missing:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.convnet.eval()

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.convnet.embed_dim, nb_classes).to(self.device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.convnet.embed_dim).to(self.device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn = input
                # inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn, self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
