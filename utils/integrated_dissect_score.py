import sys
sys.path.insert(0, "/home/shivank2/gokhale_user/shivanand/mammoth")
import time
from functools import partial
from typing import Callable, Tuple, List

import numpy as np
import torch
from math import ceil
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms


import random
import wandb
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import datetime
import json
import pandas as pd
import torch
import utils.utils_cd
import utils.similarity
from tqdm import tqdm
import math
import pandas as pd
import clip
import numpy as np
import json

class IcarlDissectandScore:
    def __init__(self, model, protocol, similarity_fn, class_order, curent_experience, current_experience_classes, device):
        self.model = model
        self.protocol = protocol
        self.similarity_fn = similarity_fn
        self.device = device
        self.experience_number = curent_experience
        self.similarity_fn = similarity_fn
        self.outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}

    def dissect(self, save_dir: str):
        utils_cd.save_activations(clip_name = "ViT-B/16", protocol=self.protocol, target_name = "Icarl", 
                        target_layers = ["classifier"], d_probe = self.experience_number, experience_num = self.experience_number, model=self.model,
                        concept_set = f"/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/new_exp{self.experience_number+1}_filtered_new.txt", batch_size = 200, 
                        device = "cuda", pool_mode="avg", 
                        save_dir = "saved_activations")
        # total_samples = 0
        target_layer= "fc"
        outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
        with open(f"/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/new_exp{self.experience_number+1}_filtered_new.txt", 'r') as f: 
            words = (f.read()).split('\n')
            # print(words)


        save_names = utils_cd.get_save_names(clip_name = "ViT-B/16", target_name = "Icarl",
                                    target_layer = "classifier", d_probe = self.experience_number,experience_number = self.experience_number, model= self.model,
                                    concept_set = f"/home/shivank2/gokhale_user/shivanand/mammoth/concept_sets/new_exp{self.experience_number+1}_filtered_new.txt", pool_mode = "avg",
                                    save_dir = "saved_activations")
        target_save_name, clip_save_name, text_save_name = save_names
        similarities = utils_cd.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, self.similarity_fn, return_target_feats=False, device="cuda"
        )
        vals, ids = torch.max(similarities, dim=1)
        # print(ids)
        del similarities
        torch.cuda.empty_cache()

        descriptions = [words[int(idx)] for idx in ids]

        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())

        df = pd.DataFrame(outputs)
        if not os.path.exists(f"/home/shivank2/gokhale_user/shivanand/mammoth/activations/results_task{self.experience_number+1}_Icarl"):
            os.mkdir(f"/home/shivank2/gokhale_user/shivanand/mammoth/activations/results_task{self.experience_number+1}_Icarl")
        # save_path = "{}/{}_{}".format("/home/shivank2/gokhale_user/shivanand/icarl-pytorch/results", "ViT-B/16", datetime.datetime.now().strftime("%y_%m_%d_%H_%M"))
        # os.mkdir(save_path)
        save_path = f"/home/shivank2/gokhale_user/shivanand/mammoth/activations/results_task{self.experience_number+1}_Icarl"
        df.to_csv(os.path.join(save_path,"descriptions.csv"), index=False)
        print("Hola! Dissected succesfully")
        self.outputs = outputs
        return outputs
    
    def get_values_from_json(self,json_file, indexes):

        with open(json_file, 'r') as file:
            data = json.load(file)

        keys = list(data.keys())

        selected_values = [data[keys[i]] for i in indexes if i < len(keys)]
        selected_values_dict = dict(zip([keys[i] for i in indexes if i < len(keys)], selected_values))

        return selected_values
    def get_clip_text_features(self,model, text, batch_size=1000):
        text_features = []
        # print("hey there, computing clip embeddings for concepts started")
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)
        # print("Tatastu, computing clip embeddings for concepts ended")
        return text_features


        return selected_values
    def get_clip_text_features(self,model, text, batch_size=1000):
        text_features = []
        # print("hey there, computing clip embeddings for concepts started")
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)
        # print("Tatastu, computing clip embeddings for concepts ended")
        return text_features

    def scoring_function(self,clip_model,filtered_concept_set, next_exp_concept_set_path, indices, device):
        concept_set = self.outputs["description"]
        similarity_scores = self.outputs['similarity']
        threshold =  np.percentile(similarity_scores,75)
        filtered_concept_set =  [desc for desc, sim in zip(self.outputs["description"], similarity_scores) if sim > threshold]
        # print(filtered_concept_set)
        # print(indices)
        next_exp_concept_set = self.get_values_from_json(next_exp_concept_set_path, indices)
        # print(next_exp_concept_set)
        text = clip.tokenize(["{}".format(word) for word in filtered_concept_set]).to(device= device)
        text_features = self.get_clip_text_features(clip_model, text, 1000)
        # print(text_features.shape)
        scores_list = []
        for i in next_exp_concept_set:
            next_exp_txt  = clip.tokenize(["{}".format(word) for word in i]).to(device= device)   
            next_exp_text_features = self.get_clip_text_features(clip_model, next_exp_txt, 1000)
            # print(next_exp_text_features.shape)

            text_features = text_features.to(device)
            next_exp_text_features = next_exp_text_features.to(device)
            normalized_text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            normalized_next_exp_text_features = torch.nn.functional.normalize(next_exp_text_features, p=2, dim=-1)
            similarity_matrix = torch.mm(normalized_text_features, normalized_next_exp_text_features.T)

            max_vector = torch.max(similarity_matrix, dim=-1).values
            # mean_vector = torch.mean(similarity_matrix, dim=-1)
            # score = torch.mean(mean_vector)
            score = torch.mean(max_vector)
            # score = torch.max(max_vector)
            scores_list.append(score)
        # print(scores_list)    
        return scores_list        

        



# if __name__ == "__main__":
        # text = clip.tokenize(["{}".format(word) for word in filtered_concept_set]).to(device= device)
        # text_features = self.get_clip_text_features(clip_model, text, 1000)
        # # print(text_features.shape)
        # scores_list = []
        # for i in next_exp_concept_set:
        #     next_exp_txt  = clip.tokenize(["{}".format(word) for word in i]).to(device= device)   
        #     next_exp_text_features = self.get_clip_text_features(clip_model, next_exp_txt, 1000)
        #     # print(next_exp_text_features.shape)

        #     text_features = text_features.to(device)
        #     next_exp_text_features = next_exp_text_features.to(device)
        #     normalized_text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        #     normalized_next_exp_text_features = torch.nn.functional.normalize(next_exp_text_features, p=2, dim=-1)
        #     similarity_matrix = torch.mm(normalized_text_features, normalized_next_exp_text_features.T)

        #     max_vector = torch.max(similarity_matrix, dim=-1).values
        #     # mean_vector = torch.mean(similarity_matrix, dim=-1)
        #     # score = torch.mean(mean_vector)
        #     score = torch.mean(max_vector)
        #     # score = torch.max(max_vector)
        #     scores_list.append(score)
        # return scores_list        

        



# if __name__ == "__main__":

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761)),  
    
#     ])

#     tmp =               [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
#                             94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
#                             84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
#                             69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
#                             17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
#                             1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
#                             38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
#                             60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
#                             40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
#                             98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
        

#     fixed_class_order = tmp
#     per_pixel_mean = get_dataset_per_pixel_mean(CIFAR100('./data/cifar100', train=True, download=True,
#                                                             transform=transform))
#     transform_prototypes = transforms.Compose([
#         icarl_cifar100_augment_data,
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         lambda img_pattern: img_pattern - per_pixel_mean,  
#     ])
#     protocol = NCProtocol(CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
#                             CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
#                             n_tasks=100//10, shuffle=True, seed=None, fixed_class_order=fixed_class_order)

#     task_info: NCProtocolIterator
#     train_dataset: Dataset
#     similarity_fn = similarity.soft_wpmi
#     current_experience= 9
#     ICaRL_Dissect= IcarlDissectandScore(model=None,protocol=protocol, similarity_fn=similarity_fn, class_order=fixed_class_order, curent_experience=current_experience, current_experience_classes= fixed_class_order[current_experience*10 : current_experience*10 - (current_experience*10 -10) ], device=device)
#     next_exp_concept_set_path = "/home/shivank2/gokhale_user/shivanand/icarl-pytorch/data/concept_sets/decider_cifar100_concepts.json"
#     indices = [94, 92, 10, 72, 49, 78, 61, 14,  8, 86]
#     print(ICaRL_Dissect.dissect(save_dir="saved_activations"))
#     scores = ICaRL_Dissect.scoring_function(clip_model,filtered_concept_set=None, next_exp_concept_set_path=next_exp_concept_set_path, indices=indices, device=device)
#     scores = [i.item() for i in scores]
#     print(scores)


