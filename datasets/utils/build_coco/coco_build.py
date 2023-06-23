# %%
"""
Collection of classes for making dataset easily
download function in NUSWIDEseqMaker is based on https://github.com/nmhkahn/NUS-WIDE-downloader
"""
import os
import json
import h5py
import tqdm
import skimage.io
import numpy as np
from PIL import Image
from collections import defaultdict
from threading import Thread


def open_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.asarray(img)

    if len(img.shape) != 3:  # gray-scale img
        img = np.stack([img, img, img], axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class BaseMaker():
    def __init__(self, source_path, *args):
        self.source_path = source_path
        # with open(os.path.join(source_path, 'multihot_map.json'), 'r') as f:
            # self.multihot_map = json.load(f)  # XXX multihot_map to be a dict format
        self.multihot_map = json.loads('{"airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "chair": 21, "clock": 22, "couch": 23, "cow": 24, "dining table": 25, "dog": 26, "donut": 27, "elephant": 28, "fire hydrant": 29, "fork": 30, "frisbee": 31, "giraffe": 32, "handbag": 33, "horse": 34, "hot dog": 35, "kite": 36, "laptop": 37, "motorcycle": 38, "orange": 39, "oven": 40, "parking meter": 41, "pizza": 42, "potted plant": 43, "refrigerator": 44, "remote": 45, "sandwich": 46, "scissors": 47, "sheep": 48, "sink": 49, "skateboard": 50, "skis": 51, "snowboard": 52, "spoon": 53, "sports ball": 54, "stop sign": 55, "suitcase": 56, "surfboard": 57, "teddy bear": 58, "tennis racket": 59, "tie": 60, "toilet": 61, "toothbrush": 62, "traffic light": 63, "train": 64, "truck": 65, "tv": 66, "umbrella": 67, "vase": 68, "zebra": 69}')

    def make(self, ids, dst):
        raise NotImplementedError("make needs make func!")


class COCOseqMaker(BaseMaker):
    def __init__(self, cocosource_path, *args):
        super().__init__(cocosource_path, args)

        with open(os.path.join(self.source_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            _instances_train2014 = json.load(f)

        with open(os.path.join(self.source_path, 'annotations', 'instances_val2014.json'), 'r') as f:
            _instances_val2014 = json.load(f)

        # dict for converting catid to catname
        self.categories = {cat_info['id']: cat_info['name'] for cat_info in _instances_train2014['categories']}

        # map cocoid with image path and cat names
        cocoid_mapped_instances = dict()
        for img in _instances_train2014['images'] + _instances_val2014['images']:
            cocoid_mapped_instances[img['id']] = \
                {'img_pth': os.sep.join(img['coco_url'].split('/')[-2:])}

        for ann in _instances_train2014['annotations'] + _instances_val2014['annotations']:
            if 'cats' not in cocoid_mapped_instances[ann['image_id']]:
                cocoid_mapped_instances[ann['image_id']]['cats'] = list()

            cat = self.categories[ann['category_id']]
            if cat not in self.multihot_map.keys():
                continue

            cocoid_mapped_instances[ann['image_id']]['cats'].append(cat)

        self.cocodata = cocoid_mapped_instances

    def make(self, ids, dst):
        imgs = []
        multihot_labels = []

        desc = '_'.join(dst.split(os.sep)[-1].split('_')[:2])
        for id in tqdm.tqdm(ids, desc=desc, leave=False):
            info = self.cocodata[id]  # keys: img_path, cats
            # get img
            img = open_image(os.path.join(self.source_path, info['img_pth']))
            imgs.append(img)

            # get multihot-label
            multihot_label = np.zeros(len(self.multihot_map), dtype=np.int64)
            for cat in info['cats']:
                multihot_label[self.multihot_map[cat]] = 1
            multihot_labels.append(multihot_label.tolist())

        hf = h5py.File(dst.format(data='imgs', ext='hdf5'), 'w')
        hf.create_dataset('images', data=np.asarray(imgs))
        #hf.create_dataset('labels', data=np.asarray(multihot_labels))
        hf.close()

        with open(dst.format(data='multi_hot_categories', ext='json'), 'w') as f:
            json.dump(multihot_labels, f)


    def save_multihotdict(self, dst):
        multihot_dict_name = dst.format(dataset_name='coco')
        multihot_dict = [None for _ in range(len(self.multihot_map))]
        for cat, idx in self.multihot_map.items():
            multihot_dict[idx] = cat
        with open(multihot_dict_name, 'w') as f:
            json.dump(multihot_dict, f)


cocoroot = "/nas/softechict-nas-2/datasets/coco"
dest = "cocodata"
maker = COCOseqMaker(cocoroot)

if not os.path.exists(dest):
    os.makedirs(dest)

phase_token = ''

data_source = "."

id_fnames = [fname for fname in os.listdir(os.path.join(data_source, 'ids')) \
                if phase_token in fname]
id_fnames.sort()

for id_fname in tqdm.tqdm(id_fnames, desc='Total'):

    with open(os.path.join(data_source, 'ids', id_fname), 'r') as f:
        ids = json.load(f)

    data_fname = id_fname.replace('_id_', '_{data}_').replace('.json', '.{ext}')
    maker.make(ids, os.path.join(dest, data_fname))

maker.save_multihotdict(os.path.join(dest, 'multi_hot_dict_{dataset_name}.json'))

# %%


# %%
import torch
import numpy as np


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
    for i in range(len(sequence)):
        classes = torch.LongTensor(sequence[i])
        remain_classes = torch.LongTensor([j for j in range(81) if j not in classes])
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
