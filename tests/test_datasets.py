import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'seq-cifar100', 'seq-tinyimg',
                                     'rot-mnist', 'perm-mnist', 'mnist-360', 'seq-cifar100-224',
                                     'seq-cifar10-224', 'seq-cifar100-224-rs', 'seq-cub200-rs',
                                     'seq-cifar100-224-rs', 'seq-tinyimg-r', 'seq-cub200', 'seq-imagenet-r',
                                     'seq-cars196', 'seq-chestx', 'seq-cropdisease', 'seq-eurosat-rgb',
                                     'seq-isic', 'seq-mit67', 'seq-resisc45'])
def test_datasets(dataset):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                dataset,
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

    # clean all downloaded datasets
    dataset_paths = ['CUB200', 'CIFAR10', 'CIFAR100', 'MNIST',
                     'TINYIMG', 'imagenet-r', 'cars196', 'chestx',
                     'cropdisease', 'eurosat', 'isic', 'MIT67',
                     'NWPU-RESISC45']
    basepath = os.path.dirname(os.path.abspath(__file__))
    dt_dir = os.path.join(os.path.dirname(basepath), 'data')
    for path in dataset_paths:
        if os.path.exists(os.path.join(dt_dir, path)):
            os.system(f'rm -rf {os.path.join(dt_dir, path)}')

    main()


def test_dataset_workers():
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
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
                '--debug_mode',
                '1']

    main()


@pytest.mark.parametrize('config', ['default', 'l2p'])
def test_configs(config):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar100-224',
                '--lr',
                '1e-4',
                '--dataset_config',
                config,
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
