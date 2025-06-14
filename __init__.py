import sys
import os
root = os.getcwd()
# while we do not have the `pyproject.toml` file in the root directory
while not os.path.exists(os.path.join(root, "pyproject.toml")):
    root = os.path.dirname(root)
    if root == '/':
        raise FileNotFoundError("Could not find pyproject.toml in the directory tree.")
os.chdir(root)
sys.path.insert(0, root)

from utils import globals
from models import get_model_names, register_model, ContinualModel
from datasets import get_dataset_names, register_dataset
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from backbone import get_backbone_names, register_backbone, MammothBackbone, ReturnTypes
from utils.notebooks import load_runner, get_avail_args
from utils.training import train
from utils.conf import base_path, get_device
from utils.buffer import Buffer
from utils.args import add_rehearsal_args

__all__ = [
    "get_dataset_names",
    "get_model_names",
    "get_backbone_names",
    "load_runner",
    "get_avail_args",
    "train",
    "register_model",
    "register_dataset",
    "register_backbone",
    "ContinualModel",
    "ContinualDataset",
    "MammothBackbone",
    "base_path",
    "get_device",
    "ReturnTypes",
    "Buffer",
    "add_rehearsal_args",
    "globals",
    "store_masked_loaders",
    "set_default_from_args",
]
