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

