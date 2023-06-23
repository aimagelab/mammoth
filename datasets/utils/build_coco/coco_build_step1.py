# %%
"""
Collection of classes for making dataset easily
download function in NUSWIDEseqMaker is based on https://github.com/nmhkahn/NUS-WIDE-downloader
"""
import os
import json
import h5py
import tqdm
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
        self.multihot_map = json.loads("{'airplane': 0, 'apple': 1, 'backpack': 2, 'banana': 3, 'baseball bat': 4, 'baseball glove': 5, 'bear': 6, 'bed': 7, 'bench': 8, 'bicycle': 9, 'bird': 10, 'boat': 11, 'book': 12, 'bottle': 13, 'bowl': 14, 'broccoli': 15, 'bus': 16, 'cake': 17, 'car': 18, 'carrot': 19, 'cat': 20, 'cell phone': 21, 'chair': 22, 'clock': 23, 'couch': 24, 'cow': 25, 'cup': 26, 'dining table': 27, 'dog': 28, 'donut': 29, 'elephant': 30, 'fire hydrant': 31, 'fork': 32, 'frisbee': 33, 'giraffe': 34, 'hair drier': 35, 'handbag': 36, 'horse': 37, 'hot dog': 38, 'keyboard': 39, 'kite': 40, 'knife': 41, 'laptop': 42, 'microwave': 43, 'motorcycle': 44, 'mouse': 45, 'orange': 46, 'oven': 47, 'parking meter': 48, 'person': 49, 'pizza': 50, 'potted plant': 51, 'refrigerator': 52, 'remote': 53, 'sandwich': 54, 'scissors': 55, 'sheep': 56, 'sink': 57, 'skateboard': 58, 'skis': 59, 'snowboard': 60, 'spoon': 61, 'sports ball': 62, 'stop sign': 63, 'suitcase': 64, 'surfboard': 65, 'teddy bear': 66, 'tennis racket': 67, 'tie': 68, 'toaster': 69, 'toilet': 70, 'toothbrush': 71, 'traffic light': 72, 'train': 73, 'truck': 74, 'tv': 75, 'umbrella': 76, 'vase': 77, 'wine glass': 78, 'zebra': 79}")

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
        for id in tqdm(ids, desc=desc, leave=False):
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

import sys
print("Loading COCO maker...", file=sys.stderr)

maker = COCOseqMaker(cocoroot)

if not os.path.exists(dest):
    os.makedirs(dest)

phase_token = ''

data_source = "."

print("Loading ids...", file=sys.stderr)
from tqdm import tqdm

id_fnames = [fname for fname in tqdm(os.listdir(os.path.join(data_source, 'ids'))) \
                if phase_token in fname]
id_fnames.sort()

print("Loading robis...", file=sys.stderr)
for id_fname in tqdm(id_fnames, desc='Total'):

    with open(os.path.join(data_source, 'ids', id_fname), 'r') as f:
        ids = json.load(f)

    data_fname = id_fname.replace('_id_', '_{data}_').replace('.json', '.{ext}')
    maker.make(ids, os.path.join(dest, data_fname))

print("Saving dict...", file=sys.stderr)

maker.save_multihotdict(os.path.join(dest, 'multi_hot_dict_{dataset_name}.json'))

