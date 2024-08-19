import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from typing import Tuple

from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils import smart_joint
from utils.conf import base_path
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


class MyCUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224
    N_CLASSES = 200

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(MyCUB200.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                print('Downloading dataset')
                download(ln, filename=smart_joint(root, 'cub_200_2011.zip'), unzip=True, unzip_path=root, clean=True)

        data_file = np.load(smart_joint(root, 'train_data.npz' if train else 'test_data.npz'), allow_pickle=True)

        self.data = data_file['data']
        self.targets = torch.from_numpy(data_file['targets']).long()
        self.classes = data_file['classes']
        self.segs = data_file['segs']
        self._return_segmask = False

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        if self._return_segmask:
            # TODO: add to the return tuple
            raise "Unsupported segmentation output in training set!"

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)


class CUB200(MyCUB200):
    """Base CUB200 dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, 'logits') else [img, target]

        if ret_segmask or self._return_segmask:
            # TODO: does not work with the current implementation
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode='L')
            seg = transforms.ToTensor()(transforms.CenterCrop((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple


class SequentialCUB200(ContinualDataset):
    """Sequential CUB200 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    TRANSFORM = transforms.Compose([
        transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(MyCUB200.IMG_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                 download=True, transform=SequentialCUB200.TRANSFORM)
        test_dataset = CUB200(base_path() + 'CUB200', train=False,
                              download=True, transform=SequentialCUB200.TEST_TRANSFORM)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
        return transform

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names


CLASS_NAMES = [
    'black footed albatross',
    'laysan albatross',
    'sooty albatross',
    'groove billed ani',
    'crested auklet',
    'least auklet',
    'parakeet auklet',
    'rhinoceros auklet',
    'brewer blackbird',
    'red winged blackbird',
    'rusty blackbird',
    'yellow headed blackbird',
    'bobolink',
    'indigo bunting',
    'lazuli bunting',
    'painted bunting',
    'cardinal',
    'spotted catbird',
    'gray catbird',
    'yellow breasted chat',
    'eastern towhee',
    'chuck will widow',
    'brandt cormorant',
    'red faced cormorant',
    'pelagic cormorant',
    'bronzed cowbird',
    'shiny cowbird',
    'brown creeper',
    'american crow',
    'fish crow',
    'black billed cuckoo',
    'mangrove cuckoo',
    'yellow billed cuckoo',
    'gray crowned rosy finch',
    'purple finch',
    'northern flicker',
    'acadian flycatcher',
    'great crested flycatcher',
    'least flycatcher',
    'olive sided flycatcher',
    'scissor tailed flycatcher',
    'vermilion flycatcher',
    'yellow bellied flycatcher',
    'frigatebird',
    'northern fulmar',
    'gadwall',
    'american goldfinch',
    'european goldfinch',
    'boat tailed grackle',
    'eared grebe',
    'horned grebe',
    'pied billed grebe',
    'western grebe',
    'blue grosbeak',
    'evening grosbeak',
    'pine grosbeak',
    'rose breasted grosbeak',
    'pigeon guillemot',
    'california gull',
    'glaucous winged gull',
    'heermann gull',
    'herring gull',
    'ivory gull',
    'ring billed gull',
    'slaty backed gull',
    'western gull',
    'anna hummingbird',
    'ruby throated hummingbird',
    'rufous hummingbird',
    'green violetear',
    'long tailed jaeger',
    'pomarine jaeger',
    'blue jay',
    'florida jay',
    'green jay',
    'dark eyed junco',
    'tropical kingbird',
    'gray kingbird',
    'belted kingfisher',
    'green kingfisher',
    'pied kingfisher',
    'ringed kingfisher',
    'white breasted kingfisher',
    'red legged kittiwake',
    'horned lark',
    'pacific loon',
    'mallard',
    'western meadowlark',
    'hooded merganser',
    'red breasted merganser',
    'mockingbird',
    'nighthawk',
    'clark nutcracker',
    'white breasted nuthatch',
    'baltimore oriole',
    'hooded oriole',
    'orchard oriole',
    'scott oriole',
    'ovenbird',
    'brown pelican',
    'white pelican',
    'western wood pewee',
    'sayornis',
    'american pipit',
    'whip poor will',
    'horned puffin',
    'common raven',
    'white necked raven',
    'american redstart',
    'geococcyx',
    'loggerhead shrike',
    'great grey shrike',
    'baird sparrow',
    'black throated sparrow',
    'brewer sparrow',
    'chipping sparrow',
    'clay colored sparrow',
    'house sparrow',
    'field sparrow',
    'fox sparrow',
    'grasshopper sparrow',
    'harris sparrow',
    'henslow sparrow',
    'le conte sparrow',
    'lincoln sparrow',
    'nelson sharp tailed sparrow',
    'savannah sparrow',
    'seaside sparrow',
    'song sparrow',
    'tree sparrow',
    'vesper sparrow',
    'white crowned sparrow',
    'white throated sparrow',
    'cape glossy starling',
    'bank swallow',
    'barn swallow',
    'cliff swallow',
    'tree swallow',
    'scarlet tanager',
    'summer tanager',
    'artic tern',
    'black tern',
    'caspian tern',
    'common tern',
    'elegant tern',
    'forsters tern',
    'least tern',
    'green tailed towhee',
    'brown thrasher',
    'sage thrasher',
    'black capped vireo',
    'blue headed vireo',
    'philadelphia vireo',
    'red eyed vireo',
    'warbling vireo',
    'white eyed vireo',
    'yellow throated vireo',
    'bay breasted warbler',
    'black and white warbler',
    'black throated blue warbler',
    'blue winged warbler',
    'canada warbler',
    'cape may warbler',
    'cerulean warbler',
    'chestnut sided warbler',
    'golden winged warbler',
    'hooded warbler',
    'kentucky warbler',
    'magnolia warbler',
    'mourning warbler',
    'myrtle warbler',
    'nashville warbler',
    'orange crowned warbler',
    'palm warbler',
    'pine warbler',
    'prairie warbler',
    'prothonotary warbler',
    'swainson warbler',
    'tennessee warbler',
    'wilson warbler',
    'worm eating warbler',
    'yellow warbler',
    'northern waterthrush',
    'louisiana waterthrush',
    'bohemian waxwing',
    'cedar waxwing',
    'american three toed woodpecker',
    'pileated woodpecker',
    'red bellied woodpecker',
    'red cockaded woodpecker',
    'red headed woodpecker',
    'downy woodpecker',
    'bewick wren',
    'cactus wren',
    'carolina wren',
    'house wren',
    'marsh wren',
    'rock wren',
    'winter wren',
    'common yellowthroat'
]
