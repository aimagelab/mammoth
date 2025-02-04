import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
def test_hal(dataset):
    sys.argv = ['mammoth',
                '--model',
                'hal',
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--buffer_size',
                '50',
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
