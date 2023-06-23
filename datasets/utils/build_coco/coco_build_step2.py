# %%
import torch
import numpy as np
from tqdm import tqdm

# Imgs consists of all files of NUSWIDE or MSCOCO.
# Cats consists of all label of NUSWIDE or MSCOCO.

def divide_tasks(Cats: torch.LongTensor):
    multilabel_idx = (Cats.sum(dim=1) > 1).nonzero().view(-1)
    multilabel = Cats[multilabel_idx] # Get the samples with more than one label

    singlelabel_idx = (Cats.sum(dim=1) == 1).nonzero().view(-1)
    singlelabel = Cats[singlelabel_idx] # Get the samples with only one label
    
    # # For MSCOCO: The task sequence is chosen to make the number of samples of each task no less than 1000.
    sequence = [[56, 62, 16, 72, 58, 77, 11, 75,  9, 41, 15, 60, 32, 69, 38, 37, 25, 57,  6, 51], 
                [29,  4, 68, 76, 73, 18, 79, 55,  0, 21, 22,  3, 47, 74, 67, 23, 53, 70, 17, 59], 
                [66, 63, 30, 10, 40, 65, 71, 50, 35, 46, 49, 19, 42, 43, 54, 44, 24, 36,  1, 61], 
                [ 2, 13, 26, 27, 20,  7, 28, 64, 39,  5, 12, 14, 34, 45, 52,  8, 31, 48, 33, 78]]
    
    select_all_classes = []
    train_task_idx, test_task_idx = [], []
    for i in tqdm(range(len(sequence))):
        classes = torch.LongTensor(sequence[i])
        remain_classes = torch.LongTensor([j for j in range(80) if j not in classes])
        multi_select_idx = (multilabel[:, remain_classes].sum(dim=1) == 0).nonzero().view(-1)
        single_select_idx = (singlelabel[:, remain_classes].sum(dim=1) == 0).nonzero().view(-1)
        
        multilabel_ = multilabel[multi_select_idx]
        singlelabel_ = singlelabel[single_select_idx]
        multi_class_sum = multilabel_.sum(dim=0)
        single_class_sum = singlelabel_.sum(dim=0)

        select_classes = torch.LongTensor([cls.item() for cls in classes if multi_class_sum[cls] >= 50 and single_class_sum[cls] >= 50])
        select_all_classes.append(select_classes)
        multi_select_idx_idx = (multilabel_[:, select_classes].sum(dim=1) > 0).nonzero().view(-1)
        multi_select_idx = multi_select_idx[multi_select_idx_idx]

        single_select_idx_idx = []
        for cls in select_classes:
            cls_idx = (singlelabel_[:, cls] > 0).nonzero().view(-1)
            cls_idx_idx = torch.randperm(len(cls_idx))[:50]
            cls_idx = cls_idx[cls_idx_idx]
            single_select_idx_idx.append(cls_idx)
        single_select_idx_idx = torch.cat(single_select_idx_idx)
        single_select_idx = single_select_idx[single_select_idx_idx]

        train_task_idx.append(multilabel_idx[multi_select_idx])
        test_task_idx.append(singlelabel_idx[single_select_idx])

    select_all_classes = torch.cat(select_all_classes)
    multilabel = multilabel[:, select_all_classes]
    singlelabel = singlelabel[:, select_all_classes]
    return train_task_idx, test_task_idx, select_all_classes
# %%
mh = []

import os
import torch, json
for f in tqdm(os.listdir("cocodata")):
    if f.endswith("categories_coco.json"):
        mh.append(torch.tensor(json.load(open("cocodata/"+f, "r"))))
mh = torch.cat(mh).long()
train_task, test_task, select_classes = divide_tasks(mh)
# save all
torch.save(train_task, "cocodata/train_task.pt")
torch.save(test_task, "cocodata/test_task.pt")
torch.save(select_classes, "cocodata/select_classes.pt")