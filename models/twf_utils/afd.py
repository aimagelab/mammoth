import math
import os
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.twf_utils.conditional_bn import ConditionalBatchNorm1d
from models.twf_utils.conditional_bn import ConditionalBatchNorm2d


def get_rnd_weight(num_tasks, fin, fout=None, nonlinearity='relu'):
    results = []
    if fout is None:
        fin = fout
    for i in range(num_tasks):
        mat = torch.zeros((fout, fin))
        nn.init.kaiming_normal_(mat, mode='fan_out',
                                nonlinearity=nonlinearity)
        results.append(mat.view(-1))
    return torch.stack(results)


class ConditionalLinear(nn.Module):

    def __init__(self, fin: int, fout: int, n_tasks: int,
                 use_bn: bool = False, act_init: str = 'relu'):

        super(ConditionalLinear, self).__init__()
        self.fin, self.fout = fin, fout
        self.n_tasks = n_tasks
        self.weight = nn.Embedding(self.n_tasks, self.fin * self.fout)  # C

        self.condbn = None
        if use_bn:
            self.condbn = ConditionalBatchNorm1d(self.fout, self.n_tasks)

        self.init_parameters(act_init)

    def init_parameters(self, act_init: str):
        self.weight.weight.data.copy_(
            get_rnd_weight(self.n_tasks, fin=self.fin,
                           fout=self.fout, nonlinearity=act_init))

    def forward(self, x, task_id):
        weight = self.weight(task_id).view(-1, self.fout, self.fin)
        x = x.unsqueeze(2)  # B, fin, 1
        x = torch.bmm(weight, x)
        x = x.squeeze(2)  # B, C
        if self.condbn is not None:
            x = self.condbn(x, task_id)
        return x


class DiverseLoss(nn.Module):

    def __init__(self, lambda_loss: float, temp: float = 2.0):
        super(DiverseLoss, self).__init__()
        self.lambda_loss = lambda_loss
        self.temp = temp

    def forward(self, logits: torch.Tensor):

        c = logits.shape[1]

        if len(logits.shape) == 3:
            logits = logits.mean(1)
        elif len(logits.shape) == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).view(-1, c)

        mean = torch.mean(logits, dim=1, keepdim=True)
        std = torch.std(logits, dim=1, keepdim=True)

        normalized_logits = (logits - mean) / std

        dotlogits = torch.matmul(logits, logits.t()) / self.temp
        batch_size = normalized_logits.shape[0]

        loss = torch.logsumexp(dotlogits, dim=1).mean(0)
        loss -= 1 / self.temp
        loss -= math.log(batch_size)

        return self.lambda_loss*loss


class DiverseLossFixNormalized(nn.Module):

    def __init__(self, lambda_loss: float, temp: float = 2.0):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.temp = temp

    def forward(self, logits: torch.Tensor):

        c = logits.shape[1]

        if len(logits.shape) == 3:
            logits = logits.mean(1)
        elif len(logits.shape) == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).view(-1, c)

        mean = torch.mean(logits, dim=1, keepdim=True)
        std = torch.std(logits, dim=1, keepdim=True)

        normalized_logits = (logits - mean) / std

        dotlogits = torch.matmul(normalized_logits, normalized_logits.t()) / self.temp
        batch_size = normalized_logits.shape[0]

        loss = torch.logsumexp(dotlogits, dim=1).mean(0)
        loss -= 1 / self.temp
        loss -= math.log(batch_size)

        return self.lambda_loss*loss

class DiverseLossFixNormalizedDiagonal(nn.Module):

    def __init__(self, lambda_loss: float, temp: float = 2.0):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.temp = temp

    def forward(self, logits: torch.Tensor):

        c = logits.shape[1]

        if len(logits.shape) == 3:
            logits = logits.mean(1)
        elif len(logits.shape) == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).view(-1, c)

        mean = torch.mean(logits, dim=1, keepdim=True)
        std = torch.std(logits, dim=1, keepdim=True)

        normalized_logits = (logits - mean) / std

        dotlogits = torch.matmul(normalized_logits, normalized_logits.t()) / self.temp
        batch_size = normalized_logits.shape[0]

        dotlogits = dotlogits.masked_select(~torch.eye(batch_size, dtype=bool).to(dotlogits.device)).view(batch_size, batch_size - 1)

        loss = torch.logsumexp(dotlogits, dim=1).mean(0)
        loss -= 1 / self.temp
        loss -= math.log(batch_size)

        return self.lambda_loss*loss
class DiverseLossFixDiagonal(nn.Module):

    def __init__(self, lambda_loss: float, temp: float = 2.0):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.temp = temp

    def forward(self, logits: torch.Tensor):

        c = logits.shape[1]

        if len(logits.shape) == 3:
            logits = logits.mean(1)
        elif len(logits.shape) == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).view(-1, c)

        dotlogits = torch.matmul(logits, logits.t()) / self.temp
        batch_size = logits.shape[0]

        dotlogits = dotlogits.masked_select(~torch.eye(batch_size, dtype=bool).to(dotlogits.device)).view(batch_size, batch_size - 1)

        loss = torch.logsumexp(dotlogits, dim=1).mean(0)
        loss -= 1 / self.temp
        loss -= math.log(batch_size)

        return self.lambda_loss*loss

class ChannelDiverseLoss(nn.Module):

    def __init__(self, lambda_loss: float, temp: float = 2.0):
        super().__init__()
        self.lambda_loss = lambda_loss
        self.temp = temp

    def forward(self, logits: torch.Tensor):

        batch_size = logits.shape[0]
        c = logits.shape[1]

        if len(logits.shape) == 3:
            logits = logits.mean(1).t()
        elif len(logits.shape) == 4:
            logits = F.adaptive_avg_pool2d(logits, 1).view(-1, c)

        mean = torch.mean(logits, dim=1, keepdim=True)
        std = torch.std(logits, dim=1, keepdim=True)

        normalized_logits = (logits - mean) / std

        dotlogits = torch.matmul(logits, logits.t()) / self.temp

        loss = torch.logsumexp(dotlogits, dim=1).mean(0)
        loss -= 1 / self.temp
        loss -= math.log(batch_size)

        return self.lambda_loss*loss


class BinaryGumbelSoftmax(nn.Module):

    def __init__(self, tau: float = (2. / 3.)):

        super(BinaryGumbelSoftmax, self).__init__()
        self.tau = tau

    def forward(self, logits):

        if self.training:
            h = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)   # h --> [64, 32, 32, 64, 2]
            h = h[..., 0]   # [64, 32, 32, 64]
            return h

        h = torch.softmax(logits, -1)
        h = 1. - torch.argmax(h, -1)
        return h


class HardAttentionSoftmax(nn.Module):

    def __init__(self, fin: int, fout: int, n_tasks: int,
                 tau: float = (2./3.)):

        super(HardAttentionSoftmax, self).__init__()

        self.fin, self.fout = fin, fout
        self.n_tasks = n_tasks
        self.gumbel = BinaryGumbelSoftmax(tau)

        self.l = ConditionalLinear(self.fin, 2*self.fout, n_tasks)

    def forward(self, x, task_id, flag_stop_grad=None):
        assert len(task_id) == len(x)
        logits = self.l(x, task_id).view(-1, self.fout, 2)
        h = self.gumbel(logits)
        return h, logits


class SpatialAttn(nn.Module):

    def __init__(self, c: int, n_tasks: int, reduction_rate: int = 4):

        super(SpatialAttn, self).__init__()

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


class ChannelAttn(nn.Module):

    def __init__(self, c: int, n_tasks: int, reduction_rate: int = 1,
                 activated_with_softmax: bool = False):

        super(ChannelAttn, self).__init__()

        self.c_in = c
        self.c_out = self.c_in // reduction_rate
        self.n_tasks = n_tasks
        self.eps = 1e-6
        self.activated_with_softmax = activated_with_softmax

        self.l1 = ConditionalLinear(self.c_in, self.c_out, n_tasks,
                                    use_bn=True, act_init='tanh')
        self.l2 = ConditionalLinear(self.c_in, self.c_out, n_tasks,
                                    use_bn=True, act_init='sigmoid')
        self.lres = ConditionalLinear(self.c_in, self.c_out, n_tasks)  # C

        self.attn_act = None

        if activated_with_softmax:
            self.attn_act = HardAttentionSoftmax(self.c_out, self.c_in, n_tasks)

    def upsample(self, x, desired_shape):
        return x

    def downsample(self, x, *args, **kwargs):
        return x

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


class DoubleAttn(nn.Module):

    def __init__(self, c: int, n_tasks: int, reduction_rate: int = 4):

        super(DoubleAttn, self).__init__()

        self.c = c
        self.n_tasks = n_tasks

        self.channel_attn = ChannelAttn(c, n_tasks, reduction_rate=1,
                                        activated_with_softmax=False)
        self.spatial_attn = SpatialAttn(c, n_tasks, reduction_rate=reduction_rate)

        self.weight = nn.Embedding(self.n_tasks, self.c * (self.c * 2))
        self.gumbel = BinaryGumbelSoftmax()

    def init_parameters(self):
        self.weight.weight.data.copy_(
            get_rnd_weight(self.n_tasks, self.c, self.c * 2))

    def compute_distance(self, fm_s, fm_t, rho,
                         use_overhaul_fd):

        dist = (fm_s - fm_t) ** 2

        if use_overhaul_fd:
            mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
            dist = dist * mask

        dist = rho * dist
        dist = dist.mean(dim=(2, 3))
        dist = dist.sum(1).mean(0)

        return dist

    def upsample(self, x, desired_shape):
        _, c, h, w = x.shape
        cd, hd, wd = desired_shape
        assert cd == c and h <= hd and w <= wd
        if h == hd and w == wd:
            return x
        return F.interpolate(x, (hd, wd))

    def downsample(self, x, min_resize_threshold=16):
        _, c, h, w = x.shape
        if h < min_resize_threshold:
            return x
        return F.interpolate(x, (h // 2, w // 2))

    def forward(self, fm_t: torch.Tensor, tasks_id: torch.Tensor):

        ch_attn = self.channel_attn(fm_t, tasks_id)
        sp_attn = self.spatial_attn(fm_t, tasks_id)
        if 'ablation_type' in os.environ:
            if os.environ['ablation_type'] == 'chan_only':
                sp_attn = torch.ones_like(sp_attn)
            elif os.environ['ablation_type'] == 'space_only':
                ch_attn = torch.ones_like(ch_attn)

        ch_attn = ch_attn.unsqueeze(2).unsqueeze(3)
        # Esempio:
        # ch_attn --> (64, 64, 1, 1)
        # sp_attn --> (64, 1, 32, 32)
        x = ch_attn + sp_attn       # x --> (64, 64, 32, 32)

        weight = self.weight(tasks_id).view(-1, self.c * 2, self.c) # esempio: (64, 128, 64)
        logits = torch.einsum('bji,bixy->bjxy', weight, x)  # esempio: (64, 128, 32, 32)

        _, _, h, w = x.shape    # [64, 64, 32, 32]

        x = logits.permute((0, 2, 3, 1))    # [64, 32, 32, 128]
        x = x.view(-1, w, h, self.c, 2)     # [64, 32, 32, 64, 2]. L'ultima dimensione è 2 perché ho 2 classi: acceso o spento
        rho = self.gumbel(x)    # rho [64, 32, 32, 64]

        logits = logits.view(-1, self.c, 2, w, h)   # [64, 64, 2, 32, 32]
        rho = rho.permute((0, 3, 1, 2)) # [64, 64, 32, 32]

        return rho, logits


class TeacherTransform(nn.Module):

    def __init__(self, dims=(0, 2, 3)):
        super(TeacherTransform, self).__init__()
        self.dims = dims

    def forward(self, fm_t, targets):
        return torch.max(fm_t, self.get_margin(fm_t))

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask
        margin = masked_fm.sum(dim=self.dims, keepdim=True) / \
            (mask.sum(dim=self.dims, keepdim=True) + eps)
        return margin


class Normalize(nn.Module):

    def __init__(self, eps: float = 1e-6, dims = (2, 3)):
        super(Normalize, self).__init__()
        self.eps = eps
        self.dims = dims

    def forward(self, x):
        norm = torch.norm(x, dim=self.dims, keepdim=True)
        return torch.div(x, norm + self.eps)


class TeacherForcingLoss(nn.Module):

    def __init__(self, teacher_forcing_or: bool, lambda_forcing_loss: float):
        super(TeacherForcingLoss, self).__init__()
        self.teacher_forcing_or = teacher_forcing_or
        self.lambda_forcing_loss = lambda_forcing_loss

        self.register_buffer('index', torch.LongTensor([0]))

    def forward(self, logits, pred, target, teacher_forcing):

        logits, pred, target = logits[teacher_forcing], pred[teacher_forcing], target[teacher_forcing]
        logits = torch.index_select(logits, dim=2, index=self.index).squeeze(2)
        teacher_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        if self.teacher_forcing_or:
            mask = (1. - pred) * target
            teacher_loss = mask * teacher_loss

        teacher_loss = teacher_loss.mean() if len(teacher_loss) > 0 else .0
        teacher_loss *= self.lambda_forcing_loss
        return teacher_loss

class MultiTaskAFDAlternative(nn.Module):

    def __init__(self, chw: Tuple[int], n_tasks: int,
                 cpt: int, clear_grad: bool = False, 
                 use_overhaul_fd: bool = False,
                 lambda_diverse_loss: float = 0.0,
                 use_hard_softmax: bool = True,
                 teacher_forcing_or: bool = False, 
                 lambda_forcing_loss: float = 0.0,
                 attn_mode: str = 'ch',
                 resize_maps: bool = False,
                 min_resize_threshold: int = 16,
                 lambda_ignition_loss: float = 0.0):

        super(MultiTaskAFDAlternative, self).__init__()
        assert use_hard_softmax, 'use_hard_softmax must be True'
        assert attn_mode in ['ch', 'chsp'], 'wrong value of attn_mode'

        self.c, self.h, self.w = chw
        self.n_tasks = n_tasks
        self.cpt = cpt
        self.clear_grad = clear_grad
        self.use_overhaul_fd = use_overhaul_fd
        self.teacher_forcing_or = teacher_forcing_or
        self.lambda_forcing_loss = lambda_forcing_loss
        self.resize_maps = resize_maps
        self.min_resize_threshold = min_resize_threshold
        self.lambda_ignition_loss = lambda_ignition_loss

        self.attn_fn = None

        if attn_mode == 'ch':
            self.attn_fn = ChannelAttn(self.c, n_tasks, activated_with_softmax=True)
        elif attn_mode == 'chsp':
            self.attn_fn = DoubleAttn(self.c, self.n_tasks)
        else:
            raise ValueError

        self.teacher_forcing_loss = TeacherForcingLoss(self.teacher_forcing_or, self.lambda_forcing_loss)
        self.teacher_transform = TeacherTransform()
        self.norm = Normalize()
        self.diverse_loss = DiverseLoss(lambda_diverse_loss)

    def get_tasks_id(self, targets):
        if 'ablation_type' in os.environ and os.environ['ablation_type'] == 'non_cond':
            return torch.zeros_like(targets)
        return torch.div(targets, self.cpt, rounding_mode='floor')

    def extend_like(self, teacher_forcing, y):
        dest_shape = (-1,) + (1,) * (len(y.shape) - 1)
        return teacher_forcing.view(dest_shape).expand(y.shape)
    
    def compute_ignition_loss(self, x):
        # TODO: magari fare la media per ogni singolo canale, poi fare il log, poi fare la media su tutti i canali, poi fare la media su tutto il batch
        # x (b, n, d)
        x = torch.mean(x, dim=(1, 2, 3))
        x = -torch.log(x + 1e-8)
        x = torch.mean(x)
        return x

    def forward(self, fm_s, fm_t, targets, teacher_forcing, attention_map):

        assert len(targets) == len(fm_s) == len(fm_t) == len(teacher_forcing) == len(attention_map)

        output_rho, logits = self.attn_fn(fm_t, self.get_tasks_id(targets))

        rho = output_rho
        loss = .0
        losses = {}

        ignition_loss = self.compute_ignition_loss(output_rho)
        loss += self.lambda_ignition_loss * ignition_loss

        losses['output_rho_sum_loss'] = output_rho.sum().item()
        losses['output_rho_mean_loss'] = output_rho.mean().item()
        losses['ignition_loss'] = ignition_loss.item()

        if not self.lambda_forcing_loss > 0.0:
            if teacher_forcing.any():
                if self.resize_maps:
                    attention_map = self.attn_fn.upsample(attention_map, fm_t.shape[1:])
                p1 = torch.max(attention_map, output_rho) if self.teacher_forcing_or else attention_map
                rho = torch.where(self.extend_like(teacher_forcing, attention_map), p1, output_rho)
            else:
                rho = output_rho
        elif teacher_forcing.any():
            if 'ablation_type' not in os.environ or os.environ['ablation_type'] != 'no_mask_replay':
                if self.resize_maps:
                    attention_map = self.attn_fn.upsample(attention_map, fm_t.shape[1:])
                # loss_tf = self.teacher_forcing_loss(logits, output_rho,
                #                               attention_map, teacher_forcing)
                # losses['teacher_forcing_loss'] = loss_tf.item()
                # loss += loss_tf
        
        if self.use_overhaul_fd:
            fm_t = self.teacher_transform(fm_t, targets)    # margin relu?

        fm_t, fm_s = self.norm(fm_t), self.norm(fm_s)

        loss_dist = self.attn_fn.compute_distance(fm_s, fm_t, rho, self.use_overhaul_fd)
        losses['distillation_loss'] = loss_dist.item()
        loss += loss_dist

        if 'ablation_type' not in os.environ or os.environ['ablation_type'] != 'no_diverse':
            loss_div = self.diverse_loss(rho[~teacher_forcing])
            losses['diverse_loss'] = loss_div.item()
            loss += loss_div

        if self.resize_maps:
            output_rho = self.attn_fn.downsample(output_rho, min_resize_threshold=self.min_resize_threshold)

        return loss, output_rho, losses

class TestLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        # x is a binary map of size (B, 196, 768)
        # We take the mean of each channel
        # x = x.mean(dim=2)
        # # We take the exponential of the means
        # x = torch.exp(x)
        # # We take the mean across the batch
        # x = x.mean(dim=0)
        # return x

        # x is a binary map of size (B, 196, 768)
        # We take the mean of each channel
        # x = x.sum(dim=1)
        # # We take the exponential of the means
        # x = torch.exp(x)
        # # We take the sum
        # x = x.sum(dim=1)
        # # We take the log
        # x = torch.log(x)
        # # We take the mean across the batch
        # x = x.mean(dim=0)
        # return x

        x = x.sum(dim=1)
        x = x.mean(1)
        x = torch.log(x + 1e-7)
        x = x.mean()
        return x