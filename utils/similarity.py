import torch
import math
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt

def cos_similarity_cubed(clip_feats, target_feats, device='cuda', batch_size=10000, min_norm=1e-3):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
        target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
        
        clip_feats = clip_feats**3
        target_feats = target_feats**3
        
        clip_feats = clip_feats/torch.clip(torch.norm(clip_feats, p=2, dim=0, keepdim=True), min_norm)
        target_feats = target_feats/torch.clip(torch.norm(target_feats, p=2, dim=0, keepdim=True), min_norm)
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def cos_similarity(clip_feats, target_feats, device='cuda'):
    with torch.no_grad():
        clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        
        batch_size = 10000
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def soft_wpmi(clip_feats, target_feats, top_k=100, a=10, lam=1, device='cuda',
                        min_prob=1e-7, p_start=0.998, p_end=0.97):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        p_in_examples = p_start-(torch.arange(start=0, end=top_k)/top_k*(p_start-p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm(range(target_feats.shape[1])):
            
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            
            curr_p_d_given_e = 1+p_in_examples*(curr_clip_feats-1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
            torch.cuda.empty_cache()

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        print(prob_d_given_e.shape)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) - 
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def wpmi(clip_feats, target_feats, top_k=28, a=2, lam=0.6, device='cuda', min_prob=1e-7):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        for orig_id in tqdm(range(target_feats.shape[1])):
            torch.cuda.empty_cache()
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            curr_p_d_given_e = torch.sum(torch.log(curr_clip_feats+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) -
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))

        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def rank_reorder(clip_feats, target_feats, device="cuda", p=3, top_fraction=0.05, scale_p=0.5):
    """
    top fraction: percentage of mostly highly activating target images to use for eval. Between 0 and 1
    """
    with torch.no_grad():
        batch = 1500
        errors = []
        top_n = int(target_feats.shape[0]*top_fraction)
        target_feats, inds = torch.topk(target_feats, dim=0, k=top_n)

        for orig_id in tqdm(range(target_feats.shape[1])):
            clip_indices = clip_feats.gather(0, inds[:, orig_id:orig_id+1].expand([-1,clip_feats.shape[1]])).to(device)
            #calculate the average probability score of the top neurons for each caption
            avg_clip = torch.mean(clip_indices, dim=0, keepdim=True)
            clip_indices = torch.argsort(clip_indices, dim=0)
            clip_indices = torch.argsort(clip_indices, dim=0)
            curr_errors = []
            target = target_feats[:, orig_id:orig_id+1].to(device)
            sorted_target = torch.flip(target, dims=[0])

            baseline_diff = sorted_target - torch.cat([sorted_target[torch.randperm(len(sorted_target))] for _ in range(5)], dim=1)
            baseline_diff = torch.mean(torch.abs(baseline_diff)**p)
            torch.cuda.empty_cache()

            for i in range(math.ceil(clip_indices.shape[1]/batch)):

                clip_id = (clip_indices[:, i*batch:(i+1)*batch])
                reorg = sorted_target.expand(-1, batch).gather(dim=0, index=clip_id)
                diff = (target-reorg)
                curr_errors.append(torch.mean(torch.abs(diff)**p, dim=0, keepdim=True)/baseline_diff)
            errors.append(torch.cat(curr_errors, dim=1)/(avg_clip)**scale_p)

        errors = torch.cat(errors, dim=0)
    return -errors

