import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main, parse_args
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
    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_der_example.{dataset}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_der_example.{dataset}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
