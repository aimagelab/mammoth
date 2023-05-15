import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import math
from torch.nn import ParameterList

class CodaPromptSpecific(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0'):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.device = device
        self.n_prompts_per_tasks = {0: 5, 1: 13, 2: 13, 3: 13, 4: 13, 5: 13, 6: 5, 7: 5, 8: 13, 9: 5,}

        # e prompt init
        self.init_prompts(task_id=0)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = prompt_param[0]
        self.e_p_length = prompt_param[1]

        # prompt locations
        if prompt_param[2] == 0:
            self.e_layers = [0,1,2,3,4]
        # single
        elif prompt_param[2] == 1:
            self.e_layers = [0]
        elif prompt_param[2] == 2:
            self.e_layers = [1]
        elif prompt_param[2] == 3:
            self.e_layers = [2]
        elif prompt_param[2] == 4:
            self.e_layers = [3]
        elif prompt_param[2] == 5:
            self.e_layers = [4]
        # double
        elif prompt_param[2] == 6:
            self.e_layers = [0,1]
        elif prompt_param[2] == 7:
            self.e_layers = [1,2]
        elif prompt_param[2] == 8:
            self.e_layers = [2,3]
        elif prompt_param[2] == 9:
            self.e_layers = [3,4]
        # triple
        elif prompt_param[2] == 10:
            self.e_layers = [0,1,2]
        elif prompt_param[2] == 11:
            self.e_layers = [1,2,3]
        elif prompt_param[2] == 12:
            self.e_layers = [2,3,4]
        else:
            print(error)

        # location of ortho penalty
        self.ortho_mu = prompt_param[3]

        # ablations
        self.attention = True 
        self.attention_softmax = True 
        self.expand_and_freeze = True
        if prompt_param[4] > 0:
            if prompt_param[4] == 1:
                self.attention = False
                self.attention_softmax = False
            elif prompt_param[4] == 2:
                self.attention_softmax = False
            elif prompt_param[4] == 3:
                self.expand_and_freeze = False
                self.attention_softmax = False
        
    def process_frequency(self):
        self.task_count_f += 1
    
    def prepare_prompts(self, task_id):
        self.task_id = task_id
        for e in self.e_layers:
            e_l = self.e_p_length
            if task_id == 0:
                if self.ortho_mu == -1:
                    p = tensor_prompt(self.e_pool_size, e_l, self.emb_d)
                    k = tensor_prompt(self.e_pool_size, self.key_d)
                    a = tensor_prompt(self.e_pool_size, self.key_d)
                else:
                    p = tensor_prompt(self.e_pool_size, e_l, self.emb_d, ortho=True)
                    k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
                    a = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
                setattr(self, f'e_p_{e}_0', nn.Parameter(p.to(self.device))) # (pool_size, e_l, emb_d)
                setattr(self, f'e_k_{e}_0', nn.Parameter(k.to(self.device)))
                setattr(self, f'e_a_{e}_0', nn.Parameter(a.to(self.device)))
            else:
                new_pt = int(self.e_pool_size / (task_id+1))
                for t in range(task_id+1):
                    setattr(self, f'e_p_{e}_{t}', nn.Parameter(torch.zeros(new_pt, e_l, self.emb_d).to(self.device)))
                    setattr(self, f'e_k_{e}_{t}', nn.Parameter(torch.zeros(new_pt, self.key_d).to(self.device)))
                    setattr(self, f'e_a_{e}_{t}', nn.Parameter(torch.zeros(new_pt, self.key_d).to(self.device)))

    def init_prompts(self, task_id):
        self.task_id = task_id
        if task_id == 0:
            for e in self.e_layers:
                e_l = self.e_p_length
                for t in range(self.n_tasks):
                    setattr(self, f'e_p_{e}_{t}', nn.Parameter(tensor_prompt(self.n_prompts_per_tasks[t], e_l, self.emb_d, ortho=True).to(self.device)))
                    setattr(self, f'e_k_{e}_{t}', nn.Parameter(tensor_prompt(self.n_prompts_per_tasks[t], self.key_d, ortho=True).to(self.device)))
                    setattr(self, f'e_a_{e}_{t}', nn.Parameter(tensor_prompt(self.n_prompts_per_tasks[t], self.key_d, ortho=True).to(self.device)))
        else:
            return

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            if train:
                # get prompt, keys and attention vectors for current task
                cur_K = getattr(self,f'e_k_{l}_{task_id}')
                cur_A = getattr(self,f'e_a_{l}_{task_id}')
                cur_p = getattr(self,f'e_p_{l}_{task_id}')
                # get prompt, keys and attention vectors for previous tasks but freeze them
                old_K = [getattr(self,f'e_k_{l}_{i}').detach().clone() for i in range(task_id)]
                old_A = [getattr(self,f'e_a_{l}_{i}').detach().clone() for i in range(task_id)]
                old_p = [getattr(self,f'e_p_{l}_{i}').detach().clone() for i in range(task_id)]
                K = torch.cat(old_K + [cur_K], dim=0)
                A = torch.cat(old_A + [cur_A], dim=0)
                p = torch.cat(old_p + [cur_p], dim=0)

            else:
                # get prompt, keys and attention vectors for all tasks
                K = [getattr(self,f'e_k_{l}_{i}') for i in range(self.task_id+1)]
                A = [getattr(self,f'e_a_{l}_{i}') for i in range(self.task_id+1)]
                p = [getattr(self,f'e_p_{l}_{i}') for i in range(self.task_id+1)]
                K = torch.cat(K, dim=0)
                A = torch.cat(A, dim=0)
                p = torch.cat(p, dim=0)


            if self.attention:
                ##########
                # with attention and cosine sim
                ##########
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                if self.attention_softmax:
                    a_querry = torch.einsum('bd,kd->bkd', x_querry, nn.functional.softmax(A,dim=1))
                else:
                    a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,kld->bld', aq_k, p)   # prompt components moltiplicati per la similarità del coseno tra query e key
            else:
                ##########
                # cosine sim
                ##########
                # # (b x 1 x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1)
                aq_k = torch.einsum('bd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

        loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.FloatTensor(a,b)
    else:
        p = torch.FloatTensor(a,b,c)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p        