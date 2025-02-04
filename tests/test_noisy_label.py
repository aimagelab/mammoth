import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'seq-cifar100',
                                     'rot-mnist', 'perm-mnist',
                                     'seq-cifar100-224-rs', 'seq-tinyimg-r', 'seq-cub200',
                                     'seq-cars196', 'seq-eurosat-rgb'])
def test_symmetric_noise(dataset):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                dataset,
                '--noise_rate',
                '0.4',
                '--noise_type',
                'symmetric',
                '--disable_noisy_labels_cache',
                '1',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-cifar100'])
@pytest.mark.parametrize('noise_type', ['asymmetric', 'symmetric'])
@pytest.mark.parametrize('disable_noisy_labels_cache', ['0', '1'])
def test_noise_caching(dataset, noise_type, disable_noisy_labels_cache):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                dataset,
                '--noise_rate',
                '0.4',
                '--noise_type',
                noise_type,
                '--disable_noisy_labels_cache',
                disable_noisy_labels_cache,
                '--cache_path_noisy_labels',
                'noisy_label_test',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()

    basepath = os.path.dirname(os.path.abspath(__file__))
    dt_dir = os.path.join(os.path.dirname(basepath), 'data')
    os.system(f'rm -rf {os.path.join(dt_dir, "noisy_label_test")}')


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-cifar100',
                                     'seq-cifar100-224', 'seq-cifar10-224'])
def test_noise_asymmetric(dataset):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                dataset,
                '--noise_rate',
                '0.4',
                '--noise_type',
                'asymmetric',
                '--disable_noisy_labels_cache',
                '0',
                '--cache_path_noisy_labels',
                'noisy_label_test',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()
