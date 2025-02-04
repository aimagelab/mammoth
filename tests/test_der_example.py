import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'rot-mnist', 'perm-mnist', 'mnist-360', 'seq-cifar100-224'])
def test_der(dataset):
    sys.argv = ['mammoth',
                '--model',
                'der',
                '--dataset',
                dataset,
                '--buffer_size',
                '10',
                '--lr',
                '1e-4',
                '--alpha',
                '.5',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'rot-mnist', 'perm-mnist', 'mnist-360', 'seq-cifar100-224'])
def test_derpp(dataset):
    sys.argv = ['mammoth',
                '--model',
                'derpp',
                '--dataset',
                dataset,
                '--buffer_size',
                '10',
                '--lr',
                '1e-4',
                '--alpha',
                '.5',
                '--beta',
                '.5',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()
