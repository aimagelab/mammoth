import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-eurosat-rgb'])
def test_llava(dataset):
    sys.argv = ['mammoth',
                '--model',
                'llava',
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
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


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-eurosat-rgb'])
def test_ideficts(dataset):
    sys.argv = ['mammoth',
                '--model',
                'idefics',
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
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
