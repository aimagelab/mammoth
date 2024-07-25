""" This is the base class for all models that support future prediction, i.e., zero-shot prediction.

    It extends the ContinualModel class and adds the future_forward method, which should be implemented by all models that inherit from this class.
    Such a method should take an input tensor and return a tensor representing the future prediction. This method is used by the future prediction evaluation protocol.

    The change_transform method is used to update the transformation applied to the input data. This is useful when the model is trained on a dataset and then evaluated on a different dataset. In this case, the transformation should be updated to match the new dataset.
"""

from .continual_model import ContinualModel


class FutureModel(ContinualModel):
    def future_forward(self, x):
        raise NotImplementedError

    def change_transform(self, dataset):
        pass
