"""
This module provides functions to generate noisy labels.

The following types of noise are supported:
- Symmetric noise: For each class, assign a random class as noisy target with probability `--noise_rate`.
- Asymmetric noise: Apply asymmetric noise to the supported datasets (CIFAR-10, CIFAR-100).

The noisy labels can be cached to improve reproducibility. This can be disabled by setting `--disable_noisy_labels_cache=1`.

The code is based on:
- `Symmetric Cross Entropy for Robust Learning with Noisy Labels <https://arxiv.org/abs/1908.06112>`_
- `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach <https://arxiv.org/pdf/1609.03683>`_
- `Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels` <https://arxiv.org/pdf/1805.07836>`_
"""

from argparse import Namespace
import numpy as np
from numpy.testing import assert_array_almost_equal
import pickle
import os

from utils.conf import base_path


def build_noisy_labels(targets: np.ndarray, args: Namespace) -> np.ndarray:
    """
    Generate noisy labels according to the noise type specified in the CLI arguments.

    Args:
        targets: array of true targets
        args: CLI arguments

    Returns:
        Noisy targets
    """
    if args.noise_type == 'symmetric':
        return get_symmetric_noise(targets, args)
    elif args.noise_type == 'asymmetric':
        return get_asymmetric_noise(targets, args)
    else:
        raise ValueError(f'Noise type {args.noise_type} is not supported.')


def _check_cache(args) -> np.ndarray:
    """
    Check if noisy targets are already cached. If so, load them and return them.
    """
    if args.disable_noisy_labels_cache:
        return None

    noise_rate_perc = int(args.noise_rate * 100)
    seed = args.seed if args.seed is not None else 'disabled'

    # Check if noisy targets are already cached
    cache_basepath = os.path.join(base_path(), args.cache_path_noisy_labels)
    filepath = os.path.join(cache_basepath, args.dataset, args.noise_type, str(noise_rate_perc), str(seed), 'noisy_targets')

    if os.path.exists(filepath):
        with open(filepath, 'rb') as infile:
            noisy_targets = pickle.load(infile)
        print('Noisy sym targets loaded from file!')
        return noisy_targets

    return None


def _save_cache(args: Namespace, noisy_targets: np.ndarray) -> None:
    """
    Save noisy targets to cache.
    """

    if args.disable_noisy_labels_cache:
        return

    noise_rate_perc = int(args.noise_rate * 100) if args.noise_rate < 1 else args.noise_rate
    seed = args.seed if args.seed is not None else 'disabled'

    cache_basepath = os.path.join(base_path(), args.cache_path_noisy_labels)
    filepath = os.path.join(cache_basepath, args.dataset, args.noise_type, str(noise_rate_perc), str(seed), 'noisy_targets')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as outfile:
        pickle.dump(noisy_targets, outfile)

    print(f'Cached noisy-labels for {args.dataset} with noise-rate {args.noise_rate} and seed {seed}: {filepath}')


def get_symmetric_noise(targets: np.ndarray, args: Namespace) -> np.ndarray:
    """
    For each class, assign a random class as noisy target with probability `--noise_rate`.
    Does not alter the `targets` but returns a copy of the `targets` array with noisy targets.

    The function supports caching the noisy targets to improve reproducibility. This can be disabled by setting `--disable_noisy_labels_cache=1`.

    Args:
        targets: array of true targets
        args: CLI arguments

    Returns:
        Noisy targets
    """
    noisy_targets = _check_cache(args)
    if noisy_targets is not None:
        return noisy_targets

    rng = np.random.RandomState(args.seed)
    unique_targets = np.unique(targets)
    all_idxs = np.arange(len(targets))
    noisy_targets = targets.copy()

    for tgt in unique_targets:
        remaining_targets = unique_targets[unique_targets != tgt]
        target_index = all_idxs[targets == tgt]
        n_items_to_flip = round(len(target_index) * args.noise_rate)
        rng.shuffle(target_index)
        chosen_idxs = target_index[:n_items_to_flip]
        noisy_targets[chosen_idxs] = rng.choice(remaining_targets, n_items_to_flip)

    actual_noise = (noisy_targets != targets).mean()
    assert actual_noise > 0, 'No noise was applied to the targets'

    _save_cache(args, noisy_targets)
    return noisy_targets


def get_asymmetric_noise(targets: np.ndarray, args: Namespace) -> np.ndarray:
    """
    Apply asymmetric noise to the supported datasets (CIFAR-10, CIFAR-100).

    The function supports caching the noisy targets to improve reproducibility. This can be disabled by setting `--disable_noisy_labels_cache=1`.

    Args:
        targets: array of true targets
        args: CLI arguments

    Returns:
        Noisy targets
    """
    noisy_targets = _check_cache(args)
    if noisy_targets is not None:
        return noisy_targets

    if 'cifar100' in args.dataset:
        noisy_targets = noisify_cifar100_asymmetric(targets, args)
    elif 'cifar10' in args.dataset:
        noisy_targets = noisify_cifar10_asymmetric(targets, args)
    else:
        raise NotImplementedError(f'Asymmetric noise is not supported for dataset {args.dataset}.')

    actual_noise = (noisy_targets != targets).mean()
    assert actual_noise > 0, 'No noise was applied to the targets'

    print(f'Actual noise rate: {actual_noise:.2f}')

    _save_cache(args, noisy_targets)
    return noisy_targets


def noisify_cifar10_asymmetric(targets: np.ndarray, args: Namespace) -> np.ndarray:
    """
    Apply asymmetric noisy to CIFAR-10 according to the noise rate and the following mistakes:
    - automobile <- truck
    - bird -> airplane
    - cat <-> dog
    - deer -> horse

    Reference: `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach <https://arxiv.org/pdf/1609.03683>`_
    """

    transitions = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

    noisy_targets = targets.copy()
    rng = np.random.RandomState(args.seed)
    all_idxs = np.arange(len(targets))

    for source_tgt, dest_tgt in transitions.items():
        target_index = all_idxs[targets == source_tgt]
        n_items_to_flip = round(len(target_index) * args.noise_rate)
        rng.shuffle(target_index)
        chosen_idxs = target_index[:n_items_to_flip]
        noisy_targets[chosen_idxs] = dest_tgt

    return noisy_targets


def get_cifar100_noise_matrix(size: int, noise_rate: float) -> np.ndarray:
    """
    Compute the noise matrix for CIFAR-100 by flipping each class to the "next" class with probability 'noise_rate'.

    Args:
        size: number of classes in the superclass
        noise_rate: probability of flipping to the next class

    Returns:
        Noise matrix
    """

    transition_matrix = np.eye(size) * (1.0 - noise_rate)
    for i in np.arange(size):
        next_i = (i + 1) % size
        transition_matrix[i, next_i] = noise_rate

    assert_array_almost_equal(transition_matrix.sum(axis=1), 1, 1)
    return transition_matrix


def multiclass_noisify(y, P: np.ndarray) -> np.ndarray:
    """
    Flip classes according to transition matrix `P`.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1], 'The transition matrix should be a square matrix'
    assert np.max(y) < P.shape[0], 'The max number of the class should be smaller than the size of the transition matrix'

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    y = np.array(y)
    m = np.shape(y)[0]
    new_y = y.copy()
    flipper = np.random.RandomState(0)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_cifar100_asymmetric(targets: np.ndarray, args: Namespace) -> np.ndarray:
    """
    Apply asymmetric noisy to CIFAR-100 according to the noise rate. For each superclass, the mistakes are chosen among samples of the same superclass.

    Reference: `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach <https://arxiv.org/pdf/1609.03683>`_
    """
    n_classes = 100
    transition_matrix = np.eye(n_classes)
    n_superclasses = 20
    n_subclasses = 5

    for i in np.arange(n_superclasses):
        init, end = i * n_subclasses, (i + 1) * n_subclasses
        transition_matrix[init:end, init:end] = get_cifar100_noise_matrix(n_subclasses, args.noise_rate)

    noisy_targets = multiclass_noisify(targets, P=transition_matrix)

    return noisy_targets
