import torch
from torch import nn

import einops
from einops import rearrange

from models.twf_utils.afd import BinaryGumbelSoftmax
from models.twf_utils.afd import HardAttentionSoftmax
from models.twf_utils.utils import ConditionalLinear
from models.twf_utils.utils import WrapperNOTConditionalLinear
from models.twf_utils.utils import TaskPrompter
from models.twf_utils.utils import MLPMixer
from models.twf_utils.utils import PiecewiseRect
from models.twf_utils.utils import MLPMixerWithBottleneck
from models.twf_utils.utils import ConVitWithBottleneck
from models.twf_utils.utils import MyGumbelSoftmax
from models.twf_utils.afd import ConditionalLinear as ConditionalLinearOriginal
from models.twf_utils.afd import ConditionalBatchNorm2d
from timm.models.vision_transformer import Block
import functools
import math
import torch.nn.functional as F


class Normalize(nn.Module):

    def __init__(self, eps: float = 1e-6, dims = (2, 3)):
        super(Normalize, self).__init__()
        self.eps = eps
        self.dims = dims

    def forward(self, x):
        norm = torch.norm(x, dim=self.dims, keepdim=True)
        return torch.div(x, norm + self.eps)
class MixerAttention(nn.Module):

    def __init__(self, seq_len, embed_dim: int, n_tasks, n_classes,
                 use_conditioning: bool = True, use_prompt: bool = False):

        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.use_prompt = use_prompt
        self.use_conditioning = use_conditioning

        self.prompter = nn.Sequential()

        if self.use_prompt:
            self.seq_len += 1
            self.prompter = TaskPrompter(self.n_classes, self.embed_dim)

        self.mlp_mixer = MLPMixer(self.seq_len, self.embed_dim, self.n_tasks, self.use_conditioning)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        #self.my_gumbel = MyGumbelSoftmax(dim=-1, hard=True)
        #self.my_conv = nn.Conv2d(1, 2, kernel_size=1, stride=1)
        #self.my_linear = nn.Linear(1, 2)

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        if self.use_prompt:
            fm_t, _ = self.prompter(fm_t, y)

        x, _ = self.mlp_mixer(fm_t, tasks_id)

        if self.use_prompt:
            x = x[:, :-1, :]
        x = self.piecewise_rect(x, tasks_id)

        #x = self.my_conv(x.unsqueeze(1)).permute(0, 2, 3, 1)
        #x = self.my_linear(x.unsqueeze(-1))
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

class ClipCrossAttention(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.clip_embs = None
        self.q_proj = nn.Linear(512, self.embed_dim)
    
    def set_clip_embs(self, clip_embs):
        self.clip_embs = clip_embs
    
    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        assert self.clip_embs is not None, "clip_embs is None"
        class_idxs = y.nonzero(as_tuple=True)[1]
        embs = self.clip_embs[class_idxs]
        Q = self.q_proj(embs)
        Q = Q.unsqueeze(1).repeat(1, fm_t.shape[1], 1)
        K = self.k_proj(fm_t)
        at = K * Q

        x = self.piecewise_rect(at, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits
    
    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class TaTV2(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.proj_1 = self.build_feature_connector(self.embed_dim, self.embed_dim)
        self.proj_2 = self.build_feature_connector(self.embed_dim, self.embed_dim)
        self.proj_3 = self.build_feature_connector(self.embed_dim, self.embed_dim)
    
    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv1d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(t_channel)]

        for m in C:
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return nn.Sequential(*C)

    def proj_fm(self, fm, proj):
        fm = proj(fm.permute(0, 2, 1)).permute(0, 2, 1)
        return fm
    
    def perform_tat(self, fm_s, fm_t):
        fm_t_1 = self.proj_fm(fm_t, self.proj_1)
        fm_s_1 = self.proj_fm(fm_s, self.proj_2)
        attn_filter = torch.bmm(fm_s_1, fm_t_1.permute(0, 2, 1))
        attn_filter = F.softmax(attn_filter, dim=-1)
        fm_s_2 = self.proj_fm(fm_s, self.proj_3)
        fm_s_new = torch.bmm(attn_filter, fm_s_2)
        return fm_s_new

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

    def compute_distance(self, fm_s, fm_t, rho):
        fm_s_new = self.perform_tat(fm_s, fm_t)
        dist = (fm_s_new - fm_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class TaT(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.proj_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_3 = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = Normalize(dims=(1,))
    
    def perform_tat(self, fm_s, fm_t):
        fm_t_1 = self.proj_1(fm_t)
        fm_s_1 = self.proj_2(fm_s)
        attn_filter = torch.bmm(fm_s_1, fm_t_1.permute(0, 2, 1))
        attn_filter = F.softmax(attn_filter, dim=-1)
        fm_s_2 = self.proj_3(fm_s)
        fm_s_new = torch.bmm(attn_filter, fm_s_2)
        return fm_s_new

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

    def compute_distance(self, fm_s, fm_t, rho):
        #fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        fm_s_new = self.perform_tat(fm_s, fm_t)
        dist = (fm_s_new - fm_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class TaTNorm(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.proj_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_3 = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = Normalize(dims=(1,))
    
    def perform_tat(self, fm_s, fm_t):
        fm_t_1 = self.proj_1(fm_t)
        fm_s_1 = self.proj_2(fm_s)
        attn_filter = torch.bmm(fm_s_1, fm_t_1.permute(0, 2, 1))
        attn_filter = F.softmax(attn_filter, dim=-1)
        fm_s_2 = self.proj_3(fm_s)
        fm_s_new = torch.bmm(attn_filter, fm_s_2)
        return fm_s_new

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        fm_s_new = self.perform_tat(fm_s, fm_t)
        dist = (fm_s_new - fm_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class AttentionProbeClsNormNoGumbel(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, seq_len=197) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.seq_len = seq_len
        self.norm = Normalize(dims=(1,))


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        x_s = torch.bmm(fm_s, fm_s.permute(0, 2, 1))
        x_t = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x_s = x_s[:, 0, 1:]
        x_t = x_t[:, 0, 1:]
        dist = (x_s - x_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.mean(0)
        return dist

class AttentionProbeClsNoGumbel(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, seq_len=197) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.seq_len = seq_len


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

    def compute_distance(self, fm_s, fm_t, rho):
        x_s = torch.bmm(fm_s, fm_s.permute(0, 2, 1))
        x_t = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x_s = x_s[:, 0, 1:]
        x_t = x_t[:, 0, 1:]
        dist = (x_s - x_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.mean(0)
        return dist

class AttentionProbeCls(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, seq_len=197) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.seq_len = seq_len

        self.piecewise_rect = PiecewiseRect(self.seq_len-1, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        x = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x = x[:, :1, 1:]
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 197]

        logits = rearrange(x, 'b 1 s t -> b 1 t s')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        x_s = torch.bmm(fm_s, fm_s.permute(0, 2, 1))
        x_t = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x_s = x_s[:, 0, 1:]
        x_t = x_t[:, 0, 1:]
        dist = (x_s - x_t) ** 2
        rho = rho.squeeze(1)
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.mean(0)
        return dist

class AttentionProbeClsNorm(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, seq_len=197) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.seq_len = seq_len

        self.piecewise_rect = PiecewiseRect(self.seq_len-1, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.norm = Normalize(dims=(1,))


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        fm_t = self.norm(fm_t)
        x = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x = x[:, :1, 1:]
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 197]

        logits = rearrange(x, 'b 1 s t -> b 1 t s')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        x_s = torch.bmm(fm_s, fm_s.permute(0, 2, 1))
        x_t = torch.bmm(fm_t, fm_t.permute(0, 2, 1))
        x_s = x_s[:, 0, 1:]
        x_t = x_t[:, 0, 1:]
        dist = (x_s - x_t) ** 2
        rho = rho.squeeze(1)
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.mean(0)
        return dist

class TransformerAttentionLayerNorm(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = Block(self.embed_dim, self.num_heads, qkv_bias=True,
                               norm_layer=functools.partial(nn.LayerNorm, eps=1e-6))

        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.norm = nn.LayerNorm(self.embed_dim)


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        x = self.self_attn(fm_t)
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class TransformerAttention(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = Block(self.embed_dim, self.num_heads, qkv_bias=True,
                               norm_layer=functools.partial(nn.LayerNorm, eps=1e-6))

        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.norm = Normalize(dims=(1,))


    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        x = self.self_attn(fm_t)
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist
    
class TransformerAttentionProj(nn.Module):
    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.self_attn = Block(self.embed_dim, self.num_heads, qkv_bias=True,
                               norm_layer=functools.partial(nn.LayerNorm, eps=1e-6))

        self.conv1x1 = self.build_feature_connector(self.embed_dim, self.embed_dim)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.norm = Normalize(dims=(1,))



    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        x = self.self_attn(fm_t)
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv1d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(t_channel)]

        for m in C:
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return nn.Sequential(*C)

    def proj_student(self, fm_s):
        fm_s = self.conv1x1(fm_s.permute(0, 2, 1)).permute(0, 2, 1)
        return fm_s
    
    def compute_distance(self, fm_s, fm_t, rho):
        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)
        fm_s = self.proj_student(fm_s)
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class TransformerAttentionClip(nn.Module):

    def __init__(self, embed_dim, n_tasks, use_conditioning, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning
        self.clip_embs = None
        self.self_attn = Block(self.embed_dim, self.num_heads, qkv_bias=True,
                               norm_layer=functools.partial(nn.LayerNorm, eps=1e-6))

        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
        self.proj = nn.Linear(512, self.embed_dim)

    def set_clip_embs(self, clip_embs):
        self.clip_embs = clip_embs

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        assert self.clip_embs is not None, "clip_embs is None"
        class_idxs = y.argmax(dim=1)
        embs = self.clip_embs[class_idxs]
        embs = self.proj(embs)
        fm_t = torch.cat((fm_t, embs.unsqueeze(1)), dim=1)
        x = self.self_attn(fm_t)
        x = x[:, :-1, :]
        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

class MHAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.mha_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()
    
    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        B, N, C = fm_t.shape
        qkv = self.qkv(fm_t).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        ret = self.mha_attn(q, k, v)

        x = self.piecewise_rect(ret, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

class ChannelAttention(nn.Module):

    def __init__(self, c_in: int, n_tasks: int, reduction_rate: int = 1,
                 activated_with_softmax: bool = False, use_conditioning: bool = True):

        super(ChannelAttention, self).__init__()

        self.c_in = c_in
        self.c_out = self.c_in // reduction_rate
        self.n_tasks = n_tasks
        self.eps = 1e-6
        self.activated_with_softmax = activated_with_softmax

        if use_conditioning:
            self.l1 = ConditionalLinear(self.c_in, self.c_out, n_tasks,
                                        axis=2, act_init='tanh')
            self.l2 = ConditionalLinear(self.c_in, self.c_out, n_tasks,
                                        axis=2, act_init='sigmoid')
            self.lres = ConditionalLinear(self.c_in, self.c_out, n_tasks, axis=2)  # C
        else:
            self.l1 = WrapperNOTConditionalLinear(self.c_in, self.c_out, n_tasks, axis=2, act_init='tanh')
            self.l2 = WrapperNOTConditionalLinear(self.c_in, self.c_out, n_tasks, axis=2, act_init='sigmoid')
            self.lres = WrapperNOTConditionalLinear(self.c_in, self.c_out, n_tasks, axis=2)

        self.attn_act = None

        if activated_with_softmax:
            self.attn_act = HardAttentionSoftmax(self.c_out, self.c_in, n_tasks)

    def upsample(self, x, desired_shape):
        return x

    def downsample(self, x, *args, **kwargs):
        return x

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist.mean(dim=(1,))
        dist = rho * dist
        dist = dist.sum(1).mean(0)
        return dist

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        x = torch.mean(fm_t, 1, keepdims=True)

        (rho_a, _) = self.l1((x, tasks_id))
        rho_a = torch.tanh(rho_a)

        (rho_b, _) = self.l2((x, tasks_id))
        rho_b = torch.sigmoid(rho_b)

        (res, _) = self.lres((x, tasks_id))
        rho = rho_a * rho_b + res

        if self.activated_with_softmax:
            rho = rho.squeeze(1)
            rho, logits = self.attn_act(rho, tasks_id)
            return rho, logits

        return rho


class DoubleAttention(nn.Module):

    def __init__(self, seq_len, embed_dim: int, n_tasks, n_classes,
                 use_conditioning: bool = True, use_prompt: bool = False,
                 sp_attn_type: str = 'mixer'):

        super().__init__()

        assert sp_attn_type in ['mixer', 'convit']

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.use_prompt = use_prompt
        self.use_conditioning = use_conditioning

        self.prompter = nn.Sequential()

        if self.use_prompt:
            self.seq_len += 1
            self.prompter = TaskPrompter(self.n_classes, self.embed_dim)

        if sp_attn_type == 'mixer':
            self.spatial_attn = MLPMixerWithBottleneck(self.seq_len, self.embed_dim, \
                                                       self.n_tasks, self.use_conditioning)
        elif sp_attn_type == 'convit':
            assert use_conditioning is False, 'not supported'
            assert use_prompt is False, 'not supported'
            self.spatial_attn = ConVitWithBottleneck(self.seq_len, self.embed_dim, \
                                                       self.n_tasks, self.use_conditioning)
        else:
            raise ValueError

        self.channel_branch = ChannelAttention(self.embed_dim, self.n_tasks, activated_with_softmax=False,
                                        use_conditioning=use_conditioning)

        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)

        self.gumbel = BinaryGumbelSoftmax()

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

    def spatial_branch(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        # print(fm_t.shape)

        if self.use_prompt:
            fm_t, _ = self.prompter(fm_t, y)

        x, _ = self.spatial_attn(fm_t, tasks_id)

        if self.use_prompt:
            x = x[:, :-1, :]

        return x

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):

        sp_attn = self.spatial_branch(fm_t, y, tasks_id)
        ch_attn = self.channel_branch(fm_t, y, tasks_id)

        x = ch_attn + sp_attn

        x = self.piecewise_rect(x, tasks_id)
        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

class ChannelAttentionViT(nn.Module):
    def __init__(self, c: int, n_tasks: int, reduction_rate: int = 1,
                 activated_with_softmax: bool = False):

        super().__init__()

        self.c_in = c
        self.c_out = self.c_in // reduction_rate
        self.n_tasks = n_tasks
        self.eps = 1e-6
        self.activated_with_softmax = activated_with_softmax

        self.l1 = ConditionalLinearOriginal(self.c_in, self.c_out, n_tasks,
                                    use_bn=True, act_init='tanh')
        self.l2 = ConditionalLinearOriginal(self.c_in, self.c_out, n_tasks,
                                    use_bn=True, act_init='sigmoid')
        self.lres = ConditionalLinearOriginal(self.c_in, self.c_out, n_tasks)  # C

        self.attn_act = None

        if activated_with_softmax:
            self.attn_act = HardAttentionSoftmax(self.c_out, self.c_in, n_tasks)

    def compute_distance(self, fm_s, fm_t, rho,
                         use_overhaul_fd):

        dist = (fm_s - fm_t) ** 2

        if use_overhaul_fd:
            mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
            dist = dist * mask

        dist = dist.mean(dim=(2, 3))
        dist = rho * dist
        dist = dist.sum(1).mean(0)

        return dist

    def forward(self, fm_t: torch.Tensor, tasks_id: torch.Tensor):

        c = fm_t.shape[1]  # b, c, h, w

        x = F.adaptive_avg_pool2d(fm_t, 1).view(-1, c)

        rho_a = self.l1(x, tasks_id)
        rho_a = torch.tanh(rho_a)

        rho_b = self.l2(x, tasks_id)
        rho_b = torch.sigmoid(rho_b)

        res = self.lres(x, tasks_id)
        rho = rho_a * rho_b + res

        if self.activated_with_softmax:
            rho, logits = self.attn_act(rho, tasks_id)
            return rho, logits

        return rho

class SpatialAttentionViT(nn.Module):

    def __init__(self, c: int, n_tasks: int, reduction_rate: int = 4):

        super().__init__()

        self.c_in = c
        self.c_out = self.c_in // reduction_rate
        self.n_tasks = n_tasks
        self.eps = 1e-6

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1, stride=1)
        self.condbn_1 = ConditionalBatchNorm2d(self.c_out, self.n_tasks)

        self.conv2 = nn.Conv2d(self.c_out, self.c_out, kernel_size=3, stride=1,
                               dilation=2, padding=2)
        self.condbn_2 = ConditionalBatchNorm2d(self.c_out, self.n_tasks)

        self.conv3 = nn.Conv2d(self.c_out, self.c_out, kernel_size=3, stride=1,
                               dilation=2, padding=2)
        self.condbn_3 = ConditionalBatchNorm2d(self.c_out, self.n_tasks)

        self.conv4 = nn.Conv2d(self.c_out, 1, kernel_size=1, stride=1)
        self.condbn_4 = ConditionalBatchNorm2d(1, self.n_tasks)

    def forward(self, fm_t: torch.Tensor, tasks_id: torch.Tensor):

        x = fm_t

        x = self.conv1(x)
        x = self.condbn_1(x, tasks_id)
        x = self.act(x)

        x = self.conv2(x)
        x = self.condbn_2(x, tasks_id)
        x = self.act(x)

        x = self.conv3(x)
        x = self.condbn_3(x, tasks_id)
        x = self.act(x)

        x = self.conv4(x)
        x = self.condbn_4(x, tasks_id)

        return x


class DoubleAttentionViT(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, n_tasks: int, reduction_rate: int = 4, use_conditioning: bool = True):

        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.use_conditioning = use_conditioning

        self.channel_attn = ChannelAttentionViT(self.embed_dim, n_tasks, reduction_rate=1,
                                        activated_with_softmax=False)
        self.spatial_attn = SpatialAttentionViT(self.embed_dim, n_tasks, reduction_rate=reduction_rate)

        #self.weight = nn.Embedding(self.n_tasks, self.embed_dim * (self.embed_dim * 2))
        self.piecewise_rect = PiecewiseRect(self.embed_dim, self.n_tasks, self.use_conditioning)
        self.gumbel = BinaryGumbelSoftmax()

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = rho * dist
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

    def forward(self, fm_t: torch.Tensor, targets, tasks_id: torch.Tensor):
        #fm_t = fm_t[:, 1:].view(fm_t.shape[0], 14, 14, 768).permute(0, 3, 1, 2)
        #fm_s = fm_s[:, 1:].view(fm_s.shape[0], 14, 14, 768).permute(0, 3, 1, 2)

        fm_t = fm_t.view(fm_t.shape[0], 14, 14, 768).permute(0, 3, 1, 2)

        ch_attn = self.channel_attn(fm_t, tasks_id)
        sp_attn = self.spatial_attn(fm_t, tasks_id)

        ch_attn = ch_attn.unsqueeze(2).unsqueeze(3)
        # Esempio:
        # ch_attn --> (64, 64, 1, 1)
        # sp_attn --> (64, 1, 32, 32)
        x = ch_attn + sp_attn       # x --> (64, 64, 32, 32)

        x = x.view(x.shape[0], self.embed_dim, self.seq_len-1).permute(0, 2, 1)
        x = self.piecewise_rect(x, tasks_id)

        rho = self.gumbel(x)  # rho [64, 197, 768]

        logits = rearrange(x, 'b s f t -> b s t f')  # [64, 197, 2, 768]

        return rho, logits

        weight = self.weight(tasks_id).view(-1, self.c * 2, self.c) # esempio: (64, 128, 64)
        logits = torch.einsum('bji,bixy->bjxy', weight, x)  # esempio: (64, 128, 32, 32)

        _, _, h, w = x.shape    # [64, 64, 32, 32]

        x = logits.permute((0, 2, 3, 1))    # [64, 32, 32, 128]
        x = x.view(-1, w, h, self.c, 2)     # [64, 32, 32, 64, 2]. L'ultima dimensione è 2 perché ho 2 classi: acceso o spento
        rho = self.gumbel(x)    # rho [64, 32, 32, 64]

        logits = logits.view(-1, self.c, 2, w, h)   # [64, 64, 2, 32, 32]
        rho = rho.permute((0, 3, 1, 2)) # [64, 64, 32, 32]

        return rho, logits


class MockAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_distance(self, fm_s, fm_t, rho):
        dist = (fm_s - fm_t) ** 2
        dist = dist.mean(dim=1)
        dist = dist.sum(1).mean(0)
        return dist

    def forward(self, fm_t, y, tasks_id):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)


class MimickingAttention(nn.Module):

    def __init__(self):

        super().__init__()

    def compute_distance(self, fm_s, fm_t, rho):
        M_s = (fm_s @ fm_s.permute(0, 2, 1)) / math.sqrt(fm_s.shape[2])
        M_t = (fm_t @ fm_t.permute(0, 2, 1)) / math.sqrt(fm_t.shape[2])
        dist = (M_s - M_t) ** 2
        dist = dist.sum(dim=(1, 2)).mean(0)
        return dist

    def forward(self, fm_t: torch.Tensor, y: torch.Tensor, tasks_id: torch.Tensor):
        return torch.zeros_like(fm_t), torch.zeros_like(fm_t).unsqueeze(2).repeat(1, 1, 2, 1)

