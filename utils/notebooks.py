import json
import os
from argparse import Namespace
from typing import Any, Optional, Tuple, TYPE_CHECKING

from main import initialize
from . import to_parsable_obj

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset


def load_runner(
    model: str, dataset: str, args: dict[str, Any]
) -> Tuple["ContinualModel", "ContinualDataset"]:
    """
    Load the model and dataset for the given experiment.
    Args:
        model_name (str): The name of the model to be used.
        dataset_name (str): The name of the dataset to be used.
        args (dict): A dictionary containing the arguments for the experiment.

    Returns:
        Tuple[ContinualModel, ContinualDataset]: A tuple containing the model and dataset.
    """
    exp_args = initalize_args(model, dataset, args)

    mammoth_model, mammoth_dataset, exp_args = initialize(exp_args)

    os.environ["MAMMOTH_ARGS"] = json.dumps(to_parsable_obj(exp_args))

    return mammoth_model, mammoth_dataset


def initalize_args(
    model_name: str, dataset_name: str, args: dict[str, Any]
) -> Namespace:
    """
    Initialize the namespace with the given model and dataset names, and additional arguments.

    Args:
        model_name (str): The name of the model to be used.
        dataset_name (str): The name of the dataset to be used.
        args (dict[str, Any]): A dictionary containing additional arguments for the experiment.

    Returns:
        Namespace: A namespace containing the model and dataset names, and additional arguments.
    """
    from main import parse_args

    exp_str = [
        f"--{k}={v}"
        for k, v in args.items()
        if k not in ["model", "dataset"] and v is not None
    ]
    exp_str += ["--dataset", dataset_name, "--model", model_name]

    return parse_args(exp_str)


def get_avail_args(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    args: dict[str, Any] = {},
) -> Tuple[dict[str, dict], dict[str, dict]]:
    """
    Get the available arguments for the Mammoth framework.
    This function returns two lists: one for required arguments and one for optional arguments.
    Each list contains dictionaries with:
    - the name of the argument as the key
    - a dictionary with the 'default' and 'description' of the argument as the value
    Note that the 'default' key is only present for optional arguments.
    """
    from main import parse_args

    exp_str = [
        f"--{k}={v}"
        for k, v in args.items()
        if k not in ["model", "dataset"] and v is not None
    ]
    if dataset is not None:
        exp_str += ["--dataset", dataset]
    if model is not None:
        exp_str += ["--model", model]

    parser = parse_args(exp_str, return_parser_only=True)

    required_args, optional_args = {}, {}
    for group in parser._action_groups:
        for action in group._group_actions:
            if action.dest not in ["help", "debug_mode"]:
                if action.required:
                    required_args[action.dest] = {"description": action.help}
                else:
                    optional_args[action.dest] = {
                        "default": action.default,
                        "description": action.help,
                    }
    return required_args, optional_args
