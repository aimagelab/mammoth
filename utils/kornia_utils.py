from typing import List, Union
import kornia
from torch import nn
import torch
from torchvision import transforms
from kornia.augmentation.container.params import ParamItem


class KorniaMultiAug(kornia.augmentation.AugmentationSequential):
    """
    A custom augmentation class that performs multiple Kornia augmentations.

    Args:
        n_augs (int): The number of augmentations to apply.
        aug_list (List[kornia.augmentation.AugmentationBase2D]): The list of augmentations to apply.

    Methods:
        forward: Overrides the forward method to apply the transformation without gradient computation.
    """

    def __init__(self, n_augs: int, aug_list: List[kornia.augmentation.AugmentationBase2D]):
        super().__init__(*aug_list)
        self.n_augs = n_augs

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overrides the forward method to apply the transformation without gradient computation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        original_shape = x.shape
        x = super().forward(x.repeat(self.n_augs, 1, 1, 1))
        x = x.reshape(self.n_augs, *original_shape)
        return x


class KorniaAugNoGrad(kornia.augmentation.AugmentationSequential):
    """
    A custom augmentation class that applies Kornia augmentations without gradient computation.

    Inherits from `kornia.augmentation.AugmentationSequential`.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.


    Methods:
        _do_transform: Performs the transformation without gradient computation.
        forward: Overrides the forward method to apply the transformation without gradient computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _do_transform(self, *args, **kwargs) -> torch.Tensor:
        """
        Performs the transformation without gradient computation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        x = super().forward(*args, **kwargs)
        return x

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Overrides the forward method to apply the transformation without gradient computation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return self._do_transform(*args, **kwargs)


def to_kornia_transform(transform: transforms.Compose, apply: bool = True) -> Union[List[kornia.augmentation.AugmentationBase2D], KorniaAugNoGrad]:
    """
    Converts PIL transforms to Kornia transforms.

    Args:
        transform (transforms.Compose): The torchvision transform to be converted.
        apply (bool, optional): Whether to convert the processed kornia transforms list into a KorniaAugNoGrad object. Defaults to True.

    Returns:
        Union[List[kornia.augmentation.AugmentationBase2D], KorniaAugNoGrad]: The converted Kornia transforms.
    """
    if isinstance(transform, kornia.augmentation.AugmentationSequential) or \
            (isinstance(transform, nn.Sequential) and isinstance(transform[0], kornia.augmentation.AugmentationBase2D)):
        return transform
    if not isinstance(transform, list):
        if hasattr(transform, "transforms"):
            transform = list(transform.transforms)
        else:
            transform = [transform]

    ts = []

    for t in transform:
        if isinstance(t, transforms.RandomResizedCrop):
            ts.append(kornia.augmentation.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio, interpolation=t.interpolation))
        elif isinstance(t, transforms.RandomHorizontalFlip):
            ts.append(kornia.augmentation.RandomHorizontalFlip(p=t.p))
        elif isinstance(t, transforms.RandomVerticalFlip):
            ts.append(kornia.augmentation.RandomVerticalFlip(p=t.p))
        elif isinstance(t, transforms.RandomRotation):
            ts.append(kornia.augmentation.RandomRotation(degrees=t.degrees, interpolation=t.interpolation))
        elif isinstance(t, transforms.RandomGrayscale):
            ts.append(kornia.augmentation.RandomGrayscale(p=t.p))
        elif isinstance(t, transforms.RandomAffine):
            ts.append(kornia.augmentation.RandomAffine(degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear, interpolation=t.interpolation, fill=t.fill))
        elif isinstance(t, transforms.RandomPerspective):
            ts.append(kornia.augmentation.RandomPerspective(distortion_scale=t.distortion_scale, p=t.p, interpolation=t.interpolation, fill=t.fill))
        elif isinstance(t, transforms.RandomCrop):
            ts.append(kornia.augmentation.RandomCrop(size=t.size, padding=t.padding, pad_if_needed=t.pad_if_needed, fill=t.fill, padding_mode=t.padding_mode))
        elif isinstance(t, transforms.RandomErasing):
            ts.append(kornia.augmentation.RandomErasing(p=t.p, scale=t.scale, ratio=t.ratio, value=t.value, inplace=t.inplace))
        elif isinstance(t, transforms.ColorJitter):
            ts.append(kornia.augmentation.ColorJitter(brightness=t.brightness, contrast=t.contrast, saturation=t.saturation, hue=t.hue))
        elif isinstance(t, transforms.RandomApply):
            ts.append(kornia.augmentation.RandomApply(t.transforms, p=t.p))
        elif isinstance(t, transforms.RandomChoice):
            ts.append(kornia.augmentation.RandomChoice(t.transforms))
        elif isinstance(t, transforms.RandomOrder):
            ts.append(kornia.augmentation.RandomOrder(t.transforms))
        elif isinstance(t, transforms.RandomResizedCrop):
            ts.append(kornia.augmentation.RandomResizedCrop(size=t.size, scale=t.scale, ratio=t.ratio, interpolation=t.interpolation))
        elif isinstance(t, transforms.Compose):
            ts.extend(to_kornia_transform(t, apply=False))
        elif isinstance(t, transforms.ToTensor) or isinstance(t, transforms.ToPILImage):
            pass
        elif isinstance(t, transforms.Normalize):
            ts.append(kornia.augmentation.Normalize(mean=t.mean, std=t.std, p=1))
        else:
            raise NotImplementedError

    if not apply:
        return ts

    return KorniaAugNoGrad(*ts, same_on_batch=True)


class CustomKorniaRandAugment(kornia.augmentation.auto.PolicyAugmentBase):
    """
    A custom augmentation class that applies randaug as a Kornia augmentation.

    Inherits from `kornia.augmentation.auto.PolicyAugmentBase`.

    Args:
        n (int): The number of augmentations to apply.
        policy: The policy of augmentations to apply.

    Attributes:
        rand_selector (torch.distributions.Categorical): A categorical distribution for selecting augmentations randomly.
        n (int): The number of augmentations to apply.

    Methods:
        _getpolicy: Returns the Kornia augmentation operation based on the name, probability, and magnitude.
        compose_subpolicy_sequential: Composes a subpolicy of augmentations sequentially.
        get_forward_sequence: Returns the forward sequence of augmentations based on the selected indices or parameters.
        forward_parameters: Computes the forward parameters for the augmentations.
    """

    def __init__(self, n: int, policy) -> None:
        super().__init__(policy)
        selection_weights = torch.tensor([1.0 / len(self)] * len(self))
        self.rand_selector = torch.distributions.Categorical(selection_weights)
        self.n = n

    def _getpolicy(self, name, p, m):
        """
        Returns the Kornia augmentation operation based on the name, probability, and magnitude.

        Args:
            name (str): The name of the augmentation operation.
            p (float): The probability of applying the augmentation.
            m (float): The magnitude of the augmentation.

        Returns:
            kornia.augmentation.auto.operations.ops: The Kornia augmentation operation.
        """
        if 'shear' in name.lower() or 'solarize' in name.lower() or 'rotate' in name.lower() or 'translate' in name.lower() or name.lower().startswith('contrast'):
            # for some reason, some kornia ops have the probability and magnitude in the opposite order
            return getattr(kornia.augmentation.auto.operations.ops, name)(m, p)
        else:
            return getattr(kornia.augmentation.auto.operations.ops, name)(p, m)

    def compose_subpolicy_sequential(self, subpolicy):
        """
        Composes a subpolicy of augmentations sequentially.

        Args:
            subpolicy (List[Tuple[str, float, float]]): The subpolicy of augmentations.

        Returns:
            kornia.augmentation.auto.PolicySequential: The composed subpolicy of augmentations.
        """
        return kornia.augmentation.auto.PolicySequential(*[self._getpolicy(name, p, m) for (name, p, m) in subpolicy])

    def get_forward_sequence(self, params=None):
        """
        Returns the forward sequence of augmentations based on the selected indices or parameters.

        Args:
            params (List[ParamItem], optional): The parameters of the augmentations. Defaults to None.

        Returns:
            List[Tuple[str, kornia.augmentation.auto.operations.ops]]: The forward sequence of augmentations.
        """
        if params is None:
            idx = self.rand_selector.sample((self.n,))
            return self.get_children_by_indices(idx)

        return self.get_children_by_params(params)

    def forward_parameters(self, batch_shape: torch.Size):
        """
        Computes the forward parameters for the augmentations.

        Args:
            batch_shape (torch.Size): The shape of the input batch.

        Returns:
            List[ParamItem]: The forward parameters for the augmentations.
        """
        named_modules = self.get_forward_sequence()

        params = []

        for name, module in named_modules:
            mod_param = module.forward_parameters(batch_shape)
            param = ParamItem(name, [ParamItem(mname, mp)[1] for (mname, _), mp in zip(module.named_children(), mod_param)])
            params.append(param)

        return params
