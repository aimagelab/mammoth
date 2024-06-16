import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main, parse_args
import pytest

@pytest.mark.parametrize('validation', ['0.2','0','20'])
@pytest.mark.parametrize('validation_mode', ['complete','current'])
def test_validation_classil( validation, validation_mode):
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_validation_classil.seq-cifar10.{validation}.{validation_mode}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()


@pytest.mark.parametrize('dataset', ['mnist-360','perm-mnist'])
@pytest.mark.parametrize('validation', ['0.2','0','20'])
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_validation_domainil.{dataset}.{validation}.{validation_mode}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()
