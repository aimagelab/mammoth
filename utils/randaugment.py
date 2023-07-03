## the code is mostly taken from autoaugment pytorch repo:
# https://github.com/DeepVoltaire/AutoAugment


import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class Cutout:

    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1], :] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        """
        Auto augment from https://arxiv.org/pdf/1805.09501.pdf
        :param fillcolor:
        """

        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10PolicyAll(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "Invert", 7, 0.2, "Contrast", 6, fillcolor),
            SubPolicy(0.7, "Rotate", 2, 0.3, "TranslateX", 9, fillcolor),
            SubPolicy(0.8, "Sharpness", 1, 0.9, "Sharpness", 3, fillcolor),
            SubPolicy(0.5, "ShearY", 8, 0.7, "TranslateY", 9, fillcolor),
            SubPolicy(0.5, "AutoContrast", 8, 0.9, "Equalize", 2, fillcolor),
            SubPolicy(0.4, "Solarize", 5, 0.9, "AutoContrast", 3, fillcolor),
            SubPolicy(0.9, "TranslateY", 9, 0.7, "TranslateY", 9, fillcolor),
            SubPolicy(0.9, "AutoContrast", 2, 0.8, "Solarize", 3, fillcolor),
            SubPolicy(0.8, "Equalize", 8, 0.1, "Invert", 3, fillcolor),
            SubPolicy(0.7, "TranslateY", 9, 0.9, "AutoContrast", 1, fillcolor),
            SubPolicy(0.4, "Solarize", 5, 0.0, "AutoContrast", 2, fillcolor),
            SubPolicy(0.7, "TranslateY", 9, 0.7, "TranslateY", 9, fillcolor),
            SubPolicy(0.9, "AutoContrast", 0, 0.4, "Solarize", 3, fillcolor),
            SubPolicy(0.7, "Equalize", 5, 0.1, "Invert", 3, fillcolor),
            SubPolicy(0.7, "TranslateY", 9, 0.7, "TranslateY", 9, fillcolor),
            SubPolicy(0.4, "Solarize", 5, 0.9, "AutoContrast", 1, fillcolor),
            SubPolicy(0.8, "TranslateY", 9, 0.9, "TranslateY", 9, fillcolor),
            SubPolicy(0.8, "AutoContrast", 0, 0.7, "TranslateY", 9, fillcolor),
            SubPolicy(0.2, "TranslateY", 7, 0.9, "Color", 6, fillcolor),
            SubPolicy(0.7, "Equalize", 6, 0.4, "Color", 9, fillcolor),
            SubPolicy(0.2, "ShearY", 7, 0.3, "Posterize", 7, fillcolor),
            SubPolicy(0.4, "Color", 3, 0.6, "Brightness", 7, fillcolor),
            SubPolicy(0.3, "Sharpness", 9, 0.7, "Brightness", 9, fillcolor),
            SubPolicy(0.6, "Equalize", 5, 0.5, "Equalize", 1, fillcolor),
            SubPolicy(0.6, "Contrast", 7, 0.6, "Sharpness", 5, fillcolor),
            SubPolicy(0.3, "Brightness", 7, 0.5, "AutoContrast", 8, fillcolor),
            SubPolicy(0.9, "AutoContrast", 4, 0.5, "AutoContrast", 6, fillcolor),
            SubPolicy(0.3, "Solarize", 5, 0.6, "Equalize", 5, fillcolor),
            SubPolicy(0.2, "TranslateY", 4, 0.3, "Sharpness", 3, fillcolor),
            SubPolicy(0.0, "Brightness", 8, 0.8, "Color", 8, fillcolor),
            SubPolicy(0.2, "Solarize", 6, 0.8, "Color", 6, fillcolor),
            SubPolicy(0.2, "Solarize", 6, 0.8, "AutoContrast", 1, fillcolor),
            SubPolicy(0.4, "Solarize", 1, 0.6, "Equalize", 5, fillcolor),
            SubPolicy(0.0, "Brightness", 0, 0.5, "Solarize", 2, fillcolor),
            SubPolicy(0.9, "AutoContrast", 5, 0.5, "Brightness", 3, fillcolor),
            SubPolicy(0.7, "Contrast", 5, 0.0, "Brightness", 2, fillcolor),
            SubPolicy(0.2, "Solarize", 8, 0.1, "Solarize", 5, fillcolor),
            SubPolicy(0.5, "Contrast", 1, 0.2, "TranslateY", 9, fillcolor),
            SubPolicy(0.6, "AutoContrast", 5, 0.0, "TranslateY", 9, fillcolor),
            SubPolicy(0.9, "AutoContrast", 4, 0.8, "Equalize", 4, fillcolor),
            SubPolicy(0.0, "Brightness", 7, 0.4, "Equalize", 7, fillcolor),
            SubPolicy(0.2, "Solarize", 5, 0.7, "Equalize", 5, fillcolor),
            SubPolicy(0.6, "Equalize", 8, 0.6, "Color", 2, fillcolor),
            SubPolicy(0.3, "Color", 7, 0.2, "Color", 4, fillcolor),
            SubPolicy(0.5, "AutoContrast", 2, 0.7, "Solarize", 2, fillcolor),
            SubPolicy(0.2, "AutoContrast", 0, 0.1, "Equalize", 0, fillcolor),
            SubPolicy(0.6, "ShearY", 5, 0.6, "Equalize", 5, fillcolor),
            SubPolicy(0.9, "Brightness", 3, 0.4, "AutoContrast", 1, fillcolor),
            SubPolicy(0.8, "Equalize", 8, 0.7, "Equalize", 7, fillcolor),
            SubPolicy(0.7, "Equalize", 7, 0.5, "Solarize", 0, fillcolor),
            SubPolicy(0.8, "Equalize", 4, 0.8, "TranslateY", 9, fillcolor),
            SubPolicy(0.8, "TranslateY", 9, 0.6, "TranslateY", 9, fillcolor),
            SubPolicy(0.9, "TranslateY", 0, 0.5, "TranslateY", 9, fillcolor),
            SubPolicy(0.5, "AutoContrast", 3, 0.3, "Solarize", 4, fillcolor),
            SubPolicy(0.5, "Solarize", 3, 0.4, "Equalize", 4, fillcolor),
            SubPolicy(0.7, "Color", 7, 0.5, "TranslateX", 8, fillcolor),
            SubPolicy(0.3, "Equalize", 7, 0.4, "AutoContrast", 8, fillcolor),
            SubPolicy(0.4, "TranslateY", 3, 0.2, "Sharpness", 6, fillcolor),
            SubPolicy(0.9, "Brightness", 6, 0.2, "Color", 8, fillcolor),
            SubPolicy(0.5, "Solarize", 2, 0.0, "Invert", 3, fillcolor),
            SubPolicy(0.1, "AutoContrast", 5, 0.0, "Brightness", 0, fillcolor),
            SubPolicy(0.2, "Cutout", 4, 0.1, "Equalize", 1, fillcolor),
            SubPolicy(0.7, "Equalize", 7, 0.6, "AutoContrast", 4, fillcolor),
            SubPolicy(0.1, "Color", 8, 0.2, "ShearY", 3, fillcolor),
            SubPolicy(0.4, "ShearY", 2, 0.7, "Rotate", 0, fillcolor),
            SubPolicy(0.1, "ShearY", 3, 0.9, "AutoContrast", 5, fillcolor),
            SubPolicy(0.3, "TranslateY", 6, 0.3, "Cutout", 3, fillcolor),
            SubPolicy(0.5, "Equalize", 0, 0.6, "Solarize", 6, fillcolor),
            SubPolicy(0.3, "AutoContrast", 5, 0.2, "Rotate", 7, fillcolor),
            SubPolicy(0.8, "Equalize", 2, 0.4, "Invert", 0, fillcolor),
            SubPolicy(0.9, "Equalize", 5, 0.7, "Color", 0, fillcolor),
            SubPolicy(0.1, "Equalize", 1, 0.1, "ShearY", 3, fillcolor),
            SubPolicy(0.7, "AutoContrast", 3, 0.7, "Equalize", 0, fillcolor),
            SubPolicy(0.5, "Brightness", 1, 0.1, "Contrast", 7, fillcolor),
            SubPolicy(0.1, "Contrast", 4, 0.6, "Solarize", 5, fillcolor),
            SubPolicy(0.2, "Solarize", 3, 0.0, "ShearX", 0, fillcolor),
            SubPolicy(0.3, "TranslateX", 0, 0.6, "TranslateX", 0, fillcolor),
            SubPolicy(0.5, "Equalize", 9, 0.6, "TranslateY", 7, fillcolor),
            SubPolicy(0.1, "ShearX", 0, 0.5, "Sharpness", 1, fillcolor),
            SubPolicy(0.8, "Equalize", 6, 0.3, "Invert", 6, fillcolor),
            SubPolicy(0.3, "AutoContrast", 9, 0.5, "Cutout", 3, fillcolor),
            SubPolicy(0.4, "ShearX", 4, 0.9, "AutoContrast", 2, fillcolor),
            SubPolicy(0.0, "ShearX", 3, 0.0, "Posterize", 3, fillcolor),
            SubPolicy(0.4, "Solarize", 3, 0.2, "Color", 4, fillcolor),
            SubPolicy(0.1, "Equalize", 4, 0.7, "Equalize", 6, fillcolor),
            SubPolicy(0.3, "Equalize", 8, 0.4, "AutoContrast", 3, fillcolor),
            SubPolicy(0.6, "Solarize", 4, 0.7, "AutoContrast", 6, fillcolor),
            SubPolicy(0.2, "AutoContrast", 9, 0.4, "Brightness", 8, fillcolor),
            SubPolicy(0.1, "Equalize", 0, 0.0, "Equalize", 6, fillcolor),
            SubPolicy(0.8, "Equalize", 4, 0.0, "Equalize", 4, fillcolor),
            SubPolicy(0.5, "Equalize", 5, 0.1, "AutoContrast", 2, fillcolor),
            SubPolicy(0.5, "Solarize", 5, 0.9, "AutoContrast", 5, fillcolor),
            SubPolicy(0.6, "AutoContrast", 1, 0.7, "AutoContrast", 8, fillcolor),
            SubPolicy(0.2, "Equalize", 0, 0.1, "AutoContrast", 2, fillcolor),
            SubPolicy(0.6, "Equalize", 9, 0.4, "Equalize", 4, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        """
        Auto augment from https://arxiv.org/pdf/1805.09501.pdf
        :param fillcolor:
        """

        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        """
        Auto augment from https://arxiv.org/pdf/1805.09501.pdf
        :param fillcolor:
        """
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),
            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(
            self,
            p1,
            operation1,
            magnitude_idx1,
            p2,
            operation2,
            magnitude_idx2,
            fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearx": np.linspace(0, 0.3, 10),
            "sheary": np.linspace(0, 0.3, 10),
            "translatex": np.linspace(0, 150 / 331, 10),
            "translatey": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "cutout": np.round(np.linspace(0, 20, 10), 0).astype(int),
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearx": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "sheary": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translatex": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translatey": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            "cutout": lambda img, magnitude: Cutout(magnitude)(img),
        }

        self.p1 = p1
        self._operation1_name = operation1
        self.operation1 = func[operation1.lower()]
        self.magnitude1 = ranges[operation1.lower()][magnitude_idx1]
        self.p2 = p2
        self._operation2_name = operation2
        self.operation2 = func[operation2.lower()]
        self.magnitude2 = ranges[operation2.lower()][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

    def __repr__(self):
        return f"{self._operation1_name} with p:{self.p1} and magnitude:{self.magnitude1} \t" \
            f"{self._operation2_name} with p:{self.p2} and magnitude:{self.magnitude2} \n"


class RandAugment:
    """
    # randaugment is adaptived from UDA tensorflow implementation:
    # https://github.com/jizongFox/uda
    """

    @classmethod
    def get_trans_list(cls):
        trans_list = [
            'Invert', 'Cutout', 'Sharpness', 'AutoContrast', 'Posterize',
            'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
            'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
        return trans_list

    @classmethod
    def get_rand_policies(cls):
        op_list = []
        for trans in cls.get_trans_list():
            for magnitude in range(1, 10):
                op_list += [(0.5, trans, magnitude)]
        policies = []
        for op_1 in op_list:
            for op_2 in op_list:
                policies += [[op_1, op_2]]
        return policies

    def __init__(self) -> None:
        super().__init__()
        self._policies = self.get_rand_policies()

    def __call__(self, img):
        randomly_chosen_policy = self._policies[random.randint(0, len(self._policies) - 1)]
        policy = SubPolicy(*randomly_chosen_policy[0], *randomly_chosen_policy[1])
        return policy(img)

    def __repr__(self):
        return "Random Augment Policy"
