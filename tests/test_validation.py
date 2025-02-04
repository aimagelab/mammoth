import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('validation', ['0.2', '0', '20'])
@pytest.mark.parametrize('validation_mode', ['complete', 'current'])
def test_validation_classil(validation, validation_mode):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--validation',
                validation,
                '--validation_mode',
                validation_mode,
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()


@pytest.mark.parametrize('dataset', ['mnist-360', 'perm-mnist'])
@pytest.mark.parametrize('validation', ['0.2', '0', '20'])
@pytest.mark.parametrize('validation_mode', ['complete'])
def test_validation_domainil(dataset, validation, validation_mode):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--validation',
                validation,
                '--validation_mode',
                validation_mode,
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()
