import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
from utils.test_utils import init_test_environ
import pytest


@init_test_environ
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_hal.{dataset}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
