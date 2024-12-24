from typing import Tuple, TYPE_CHECKING
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset
    from models.utils.continual_model import ContinualModel


@torch.no_grad()
def evaluate_with_bias(model: 'ContinualModel', dataset: 'ContinualDataset', last=False, return_loss=False) -> Tuple[list, list]:
    assert not return_loss, "Loss is not supported for this dataset"

    loss_fn = dataset.get_loss()
    was_training = model.net.training
    model.net.eval()
    attribute_accuracies = []
    group_stats = {}
    tot_seen_samples = 0
    avg_loss = 0

    iterator = enumerate(dataset.test_loaders)

    with tqdm(iterator, total=len(dataset.test_loaders), desc='Evaluating', disable=model.args.non_verbose):
        for task_id, test_loader in iterator:
            if last and task_id < len(dataset.test_loaders) - 1:
                continue
            true_labels, pred_labels, bias_labels = [], [], []

            for idx, data in enumerate(test_loader):
                correct_counts = None
                total_counts = None

                inputs, labels, bias_label = data[0], data[1], data[-1]
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                outputs = F.sigmoid(outputs)
                pred = (outputs > 0.5).float()
                pred = pred[:, task_id]

                if return_loss:
                    loss = loss_fn(outputs, labels)
                    avg_loss += loss.item()
                    tot_seen_samples += len(labels)

                if labels.dim() > 1:
                    labels = labels[:, task_id]
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred.cpu().numpy())
                bias_labels.extend(bias_label.cpu().numpy())

                matches = (pred == labels).cpu().numpy()
                if correct_counts is None:
                    correct_counts = np.sum(matches, axis=0)
                    total_counts = len(matches)
                else:
                    correct_counts += np.sum(matches, axis=0)
                    total_counts += len(matches)

                # Compute group statistics based on bias and target attributes
                for attr_val in [0, 1]:
                    for alligned in [0, 1]:
                        mask = (bias_label.cpu().numpy() == alligned) & (labels.cpu().numpy() == attr_val)
                        group_key = f"Attr_{task_id}_Value_{attr_val}_Alligned_{alligned}"
                        if group_key not in group_stats:
                            group_stats[group_key] = {"correct": 0, "total": 0}
                        group_stats[group_key]["correct"] += np.sum(matches[mask])
                        group_stats[group_key]["total"] += np.sum(mask)

            attribute_accuracies.append(correct_counts / total_counts * 100)

    # Convert counts to percentages for group statistics
    for key in group_stats:
        group_stats[key] = (group_stats[key]["correct"] / group_stats[key]["total"]) * 100

    model.net.train(was_training)

    if return_loss:
        return attribute_accuracies, group_stats, avg_loss / tot_seen_samples
    return attribute_accuracies, group_stats
