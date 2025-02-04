import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.continual_dataset import store_masked_loaders
from datasets.bias_celeba_utils.celeba import BiasCelebA
from utils.conf import base_path


class CelebA(BiasCelebA):
    # Attributes : '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    # 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    # 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    # 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    # 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    # 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    # 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    # 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    # 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    # 'Wearing_Necktie', 'Young'

    def __init__(self, root, split="train", transform=None,
                 target_transform=None, download=False, version=None):

        super().__init__(root, split=split, transform=transform,
                         target_transform=target_transform, download=download, version=version)

        self.task_ids = self.task_number
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        imgname = self.data.iloc[index]
        img_ = Image.open(os.path.join(self.image_folder, imgname))

        original_img = img_.copy()
        original_img = self.not_aug_transform(original_img)

        targets = self.targets[index]

        if self.transform is not None:
            img = self.transform(img_)

        return img, targets, original_img


class SequentialCelebA(ContinualDataset):

    NAME = 'seq-celeba'
    SETTING = 'biased-class-il'
    N_CLASSES_PER_TASK = 1
    N_TASKS = 8
    N_CLASSES = 8
    SIZE = (224, 224)

    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])

    def __init__(self, args, split_id: int = 1):
        super().__init__(args)

        assert split_id in [1, 2], "Version not supported"
        self.split_id = split_id

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        train_dataset = CelebA(base_path(), split='train', transform=transform, download=True, version=self.split_id)
        test_dataset = CelebA(base_path(), split='test', transform=test_transform, download=True, version=self.split_id)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args('lr_scheduler')
    def get_scheduler_name(self):
        return 'cosine'

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCelebA.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone(num_clusters=None, net_type=None):
        return "resnet18_7x7_pt"

    @staticmethod
    def get_loss():
        return torch.nn.BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCelebA.MEAN, SequentialCelebA.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCelebA.MEAN, SequentialCelebA.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 25

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 64
