import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar100-224', 'seq-cropdisease'])
def test_ita_with_best_cfg(dataset):
    sys.argv = ['mammoth',
                '--model',
                'second_order',
                '--dataset',
                dataset,
                '--model_config',
                'best',
                '--n_epochs',
                '1',
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


@pytest.mark.parametrize('dataset', ['seq-cifar100-224', 'seq-cropdisease'])
def test_iel(dataset):
    sys.argv = ['mammoth',
                '--model',
                'second_order',
                '--dataset',
                dataset,
                '--lr',
                '0.001',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--use_iel',
                'true',
                '--dataset_config',
                'second_order_ta',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()
