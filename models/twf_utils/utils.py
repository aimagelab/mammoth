import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from models.twf_utils.afd import get_rnd_weight

from models.twf_utils.convit import BlockGPSA

class ConditionalLinear(nn.Module):

    def __init__(self, fin: int, fout: int, n_tasks: int,
                 axis=2, act_init: str = 'relu'):

        super(ConditionalLinear, self).__init__()
        self.fin, self.fout = fin, fout
        self.n_tasks = n_tasks
        self.weight = nn.Embedding(self.n_tasks, self.fin * self.fout) 
        self.axis = axis
        assert axis in [1, 2]
        self.init_parameters(act_init)

    def init_parameters(self, act_init: str):
        self.weight.weight.data.copy_(
            get_rnd_weight(self.n_tasks, fin=self.fin,
                           fout=self.fout, nonlinearity=act_init))

    def forward(self, inp):
        x, task_id = inp    # x: (b, n, d)
        weight = self.weight(task_id).view(-1, self.fout, self.fin)
        if self.axis == 2:
            x = torch.einsum('bnd,bcd->bnc', x, weight)
        else:
            x = torch.einsum('bnd,bcn->bcd', x, weight)
        return x, task_id


class WrapperNOTConditionalLinear(nn.Module):

    def __init__(self, fin: int, fout: int, n_tasks: int,
                 axis=2, act_init: str = 'relu'):

        super(WrapperNOTConditionalLinear, self).__init__()
        self.fin, self.fout = fin, fout
        self.n_tasks = n_tasks
        self.weight = nn.Embedding(1, self.fin * self.fout) 
        self.axis = axis
        assert axis in [1, 2]
        self.init_parameters(act_init)

    def init_parameters(self, act_init: str):
        self.weight.weight.data.copy_(
            get_rnd_weight(1, fin=self.fin,
                           fout=self.fout, nonlinearity=act_init))

    def forward(self, inp):
        x, tasks_id = inp    # x: (b, n, d)
        weight = self.weight(torch.zeros_like(tasks_id)).view(-1, self.fout, self.fin)
        if self.axis == 2:
            x = torch.einsum('bnd,bcd->bnc', x, weight)
        else:
            x = torch.einsum('bnd,bcn->bcd', x, weight)
        return x, tasks_id


class WrapperNOTConditionalLayerNorm(nn.Module):

    def __init__(self, num_features, num_tasks):
        super(WrapperNOTConditionalLayerNorm, self).__init__()
        self.num_features = num_features
        self.ln = nn.LayerNorm(num_features,
                                 elementwise_affine=False)
        self.embed = nn.Embedding(1, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, task_id):
        out = self.ln(x)
        gamma, beta = self.embed(torch.zeros_like(task_id)).chunk(2, 1)
        out = gamma.view(-1, 1, self.num_features) * out + \
            beta.view(-1, 1, self.num_features)
        return out


class ConditionalLayerNorm(nn.Module):

    def __init__(self, num_features, num_tasks):
        super(ConditionalLayerNorm, self).__init__()
        self.num_features = num_features
        self.ln = nn.LayerNorm(num_features,
                                 elementwise_affine=False)
        self.embed = nn.Embedding(num_tasks, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, task_id):
        out = self.ln(x)
        gamma, beta = self.embed(task_id).chunk(2, 1)
        out = gamma.view(-1, 1, self.num_features) * out + \
            beta.view(-1, 1, self.num_features)
        return out


class PreNormCondResidual(nn.Module):
    def __init__(self, dim, fn, num_tasks, use_conditioning: bool = True):
        super().__init__()
        self.fn = fn
        self.norm = ConditionalLayerNorm(dim, num_tasks=num_tasks) if use_conditioning \
            else WrapperNOTConditionalLayerNorm(dim, num_tasks=num_tasks)

    def forward(self, inp):
        x, task_id = inp
        out = self.norm(x, task_id)
        out = self.fn((out, task_id))
        return  out[0] + x, task_id


class CondFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor: float, dropout = 0., dense = nn.Linear):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.dense_1 = dense(dim, inner_dim)
        self.dense_2 = dense(inner_dim, dim)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, inp: tuple):
        x, task_id = self.dense_1(inp)
        x = F.gelu(x)
        #x = self.dropout(x)
        x, task_id = self.dense_2((x, task_id))
        #x = self.dropout(x)
        
        return x, task_id


class TaskPrompter(nn.Module):
    def __init__(self, n_tasks, embed_dim):
        super().__init__()
        self.n_tasks = n_tasks
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.randn(self.n_tasks, self.embed_dim), requires_grad=True)
        torch.nn.init.uniform_(self.prompt, -1, 1)

    def forward(self, x, task_id):
        prompt = self.prompt[task_id]
        x = torch.cat([x, prompt.unsqueeze(1)], dim=1)
        return x, task_id


class MLPMixer(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int,
                 n_tasks: int, use_conditioning: bool,
                 expansion_factor: float = 0.3, expansion_factor_token: float = 0.3,
                 depth: int = 1):

        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.expansion_factor = expansion_factor
        self.expansion_factor_token = expansion_factor_token
        self.depth = depth

        if self.use_conditioning:
            chan_first, chan_last = partial(ConditionalLinear, axis=1, n_tasks=n_tasks), \
                                    partial(ConditionalLinear, axis=2, n_tasks=n_tasks)
        else:
            chan_first, chan_last = partial(WrapperNOTConditionalLinear, axis=1, n_tasks=n_tasks), \
                                    partial(WrapperNOTConditionalLinear, axis=2, n_tasks=n_tasks)

        self.mixer = nn.Sequential(*[nn.Sequential(
            PreNormCondResidual(self.embed_dim, \
                                CondFeedForward(self.seq_len, self.expansion_factor, 0.0, chan_first),
                                num_tasks=n_tasks, use_conditioning=use_conditioning),
            PreNormCondResidual(self.embed_dim, \
                                CondFeedForward(self.embed_dim, self.expansion_factor_token, 0.0, chan_last),
                                num_tasks=n_tasks, use_conditioning=use_conditioning)
        ) for _ in range(self.depth)])

    def forward(self, x, tasks_id):
        return self.mixer((x, tasks_id))


class MLPMixerWithBottleneck(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int, n_tasks: int, \
                 use_conditioning: bool, reduction_rate: int = 4):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.bottleneck_dim = embed_dim // reduction_rate

        self.first_proj = ConditionalLinear(self.embed_dim, self.bottleneck_dim, self.n_tasks) if self.use_conditioning\
            else WrapperNOTConditionalLinear(self.embed_dim, self.bottleneck_dim, self.n_tasks)

        self.mlp_mixer = MLPMixer(seq_len, self.bottleneck_dim, n_tasks, \
                                  use_conditioning, expansion_factor = 0.5,\
                                  expansion_factor_token = 0.5, depth=1)

        self.snd_proj = ConditionalLinear(self.bottleneck_dim, 1, self.n_tasks) if self.use_conditioning \
            else WrapperNOTConditionalLinear(self.bottleneck_dim, 1, self.n_tasks)

    def forward(self, x, tasks_id):
        (x, _) = self.first_proj((x, tasks_id))
        (x, _) = self.mlp_mixer(x, tasks_id)
        (x, _) = self.snd_proj((x, tasks_id))
        return (x, tasks_id)


class ConVitWithBottleneck(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int, n_tasks: int, \
                 use_conditioning: bool, reduction_rate: int = 4):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.bottleneck_dim = embed_dim // reduction_rate

        self.first_proj = WrapperNOTConditionalLinear(self.embed_dim, self.bottleneck_dim, n_tasks)
        self.gpsa = BlockGPSA(self.bottleneck_dim, num_heads=8)
        self.snd_proj = WrapperNOTConditionalLinear(self.bottleneck_dim, 1, n_tasks)

    def forward(self, x, tasks_id):
        (x, _) = self.first_proj((x, tasks_id))
        x = self.gpsa(x)
        (x, _) = self.snd_proj((x, tasks_id))
        return (x, tasks_id)


class PiecewiseRect(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning: bool):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.weight = nn.Embedding(self.n_tasks, self.embed_dim * 4) if use_conditioning else\
            nn.Embedding(1, self.embed_dim * 4)

        self.init_parameters_of_weight(n_tasks if self.use_conditioning else 1, self.embed_dim)

    def init_parameters_of_weight(self, n_rows, fin):
        M = get_rnd_weight(n_rows, fin, 4)
        M = M.view(n_rows, 1, fin, 4)
        M[..., 1] = 0.0
        M[..., 3] = 0.0
        self.weight.weight.data.copy_(M.view(n_rows, -1))

    def forward(self, x: torch.Tensor, tasks_id: torch.long):

        if not self.use_conditioning:
            tasks_id = torch.zeros_like(tasks_id)

        weight = self.weight(tasks_id).view(-1, 1, self.embed_dim, 4)

        proj_1 = x * weight[..., 0] + weight[..., 1]
        proj_2 = x * weight[..., 2] + weight[..., 3]
        x = torch.stack([proj_1, proj_2], dim=-1)

        return x

class MyGumbelSoftmax(nn.Module):
    def __init__(self, tau=(2.0/3.0), dim=-1, hard=False) -> None:
        super().__init__()
        self.tau = tau
        self.dim = dim
        self.hard = hard

    def forward(self, x):
        p_1 = nn.functional.sigmoid(x)
        p_2 = 1 - p_1
        p = torch.stack([p_1, p_2], dim=-1)
        log_p = torch.log(p)

        gumbels = (-torch.empty_like(p, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0,1)
        (log_p + gumbels) / self.tau
        y_soft = gumbels.softmax(self.dim)

        if self.hard:
            # Straight through.
            index = y_soft.max(self.dim, keepdim=True)[1]
            y_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    

class ConvLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x

