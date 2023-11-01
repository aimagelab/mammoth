import kornia
from torch import nn
import torch
from torchvision import transforms
from kornia.augmentation.container.params import ParamItem


class KorniaAugNoGrad(kornia.augmentation.AugmentationSequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _do_transform(self, *args, **kwargs) -> torch.Tensor:
        x = super().forward(*args, **kwargs)
        # if len(x.shape) == 4 and x.shape[0] == 1:
        #     x = x.squeeze(0)
        return x

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._do_transform(*args, **kwargs)


def to_kornia_transform(transform: transforms.Compose, apply: bool = True):
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

    def __init__(self, n: int, policy) -> None:
        super().__init__(policy)
        selection_weights = torch.tensor([1.0 / len(self)] * len(self))
        self.rand_selector = torch.distributions.Categorical(selection_weights)
        self.n = n

    def _getpolicy(self, name, p, m):
        if 'shear' in name.lower() or 'solarize' in name.lower() or 'rotate' in name.lower() or 'translate' in name.lower() or name.lower().startswith('contrast'):
            return getattr(kornia.augmentation.auto.operations.ops, name)(m, p)  # Oh kornia, why do you have to be so inconsistent?
        else:
            return getattr(kornia.augmentation.auto.operations.ops, name)(p, m)

    def compose_subpolicy_sequential(self, subpolicy):
        return kornia.augmentation.auto.PolicySequential(*[self._getpolicy(name, p, m) for (name, p, m) in subpolicy])

    def get_forward_sequence(self, params=None):
        if params is None:
            idx = self.rand_selector.sample((self.n,))
            return self.get_children_by_indices(idx)

        return self.get_children_by_params(params)

    def forward_parameters(self, batch_shape: torch.Size):
        named_modules = self.get_forward_sequence()

        params = []

        for name, module in named_modules:
            mod_param = module.forward_parameters(batch_shape)
            # Compose it
            param = ParamItem(name, [ParamItem(mname, mp)[1] for (mname, _), mp in zip(module.named_children(), mod_param)])
            params.append(param)

        return params
