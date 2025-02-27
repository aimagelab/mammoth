import sys
sys.path.insert(0, "/home/shivank2/gokhale_user/shivanand/mammoth")
import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils.data_utils

PM_SUFFIX = {"max":"_max", "avg":""}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_save_names(clip_name, target_name, target_layer, d_probe, experience_number, model, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels, task_id in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels,task_id in tqdm(DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)):
            print("helloooooooo", images.shape)
            features = model.encode_image(images.to(device))
            
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, protocol, target_name, target_layers, d_probe, experience_num, model,
                     concept_set, batch_size, device, pool_mode, save_dir):
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    target_model, target_preprocess = data_utils.get_target_model(target_name,experience_num-1, model, device)
    #setup data
    data_c = data_utils.get_data(d_probe,protocol,experience_num, clip_preprocess)
    data_t = data_utils.get_data(d_probe,protocol,experience_num, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    #ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, experience_number=experience_num, model=model, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

    
    