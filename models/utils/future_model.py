"""
This is the base class for all models that support future prediction, i.e., zero-shot prediction.

It extends the ContinualModel class and adds the future_forward method, which should be implemented by all models that inherit from this class.
Such a method should take an input tensor and return a tensor representing the future prediction. This method is used by the future prediction evaluation protocol.

The change_transform method is used to update the transformation applied to the input data. This is useful when the model is trained on a dataset and then evaluated on a different dataset. In this case, the transformation should be updated to match the new dataset.
"""
import torch

from datasets.utils.continual_dataset import ContinualDataset
from .continual_model import ContinualModel


class FutureModel(ContinualModel):
    def future_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Function that implements the forward pass of the model for future prediction.
        This method should be implemented by all models that inherit from this class.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the future prediction.
        """
        raise NotImplementedError

    def change_transform(self, dataset: ContinualDataset):
        """
        Change the transformation applied to the input data.
        In Zero-shot learning, the model is trained on a dataset and then evaluated on a different one.
        In this case, the transformation should be updated to match the new dataset.

        Args:
            dataset (ContinualDataset): An instance of the dataset on which the model will be evaluated on new classes.
        """
        pass
