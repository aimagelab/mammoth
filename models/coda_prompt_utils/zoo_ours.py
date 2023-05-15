import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import math
from models.coda_prompt_utils.prompt import CodaPromptReservoir
from models.coda_prompt_utils.prompt_1 import CodaPromptDifferentSize
from models.coda_prompt_utils.prompt_2 import CodaPromptReservoirNew
from models.coda_prompt_utils.prompt_specific import CodaPromptSpecific

DEBUG_METRICS=True
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0

        # e prompt init
        if DEBUG_METRICS: self.metrics = {'attention':{},'keys':{}}
        for e in self.e_layers:
            e_l = self.e_p_length
            if self.ortho_mu == -1:
                p = tensor_prompt(self.e_pool_size, e_l, emb_d)
                k = tensor_prompt(self.e_pool_size, self.key_d)
                a = tensor_prompt(self.e_pool_size, self.key_d)
            else:
                p = tensor_prompt(self.e_pool_size, e_l, emb_d, ortho=True)
                k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
                a = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f'e_p_{e}',p) # (pool_size, e_l, emb_d)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

            if DEBUG_METRICS:
                self.metrics['keys'][e] = torch.zeros((self.e_pool_size,))

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

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count_f * pt)
            f = int((self.task_count_f + 1) * pt)
            
            # freeze/control past tasks
            if self.expand_and_freeze:
                
                if train:
                    if self.task_count_f > 0:
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    A = A[0:f]
                    p = p[0:f]

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
            if train and self.ortho_mu > 0:
                K = getattr(self,f'e_k_{l}')
                A = getattr(self,f'e_a_{l}')
                p = getattr(self,f'e_p_{l}')
                if self.task_count_f > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:]), dim=0)
                else:
                    K = K[s:]
                    A = A[s:]
                    p = p[s:]
                if self.ortho_mu == 1:
                    loss = ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = ortho_penalty(K)
                    loss += ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = ortho_penalty(K)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += ortho_penalty(A)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    loss = ortho_penalty(K)
                    loss += ortho_penalty(A)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
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
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p        

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if mode == 0:
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
            # classifier
            self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])

        elif self.prompt_flag == 'dual':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'reservoir':
            self.prompt = CodaPromptReservoir(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'different_size':
            self.prompt = CodaPromptDifferentSize(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'reservoir_new':
            self.prompt = CodaPromptReservoirNew(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'specific':
            self.prompt = CodaPromptSpecific(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        

    def forward(self, x, pen=False, train=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        return out, prompt_loss

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param)