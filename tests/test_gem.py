import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


def unsupport_quadprog():
    return os.name == 'nt'


@pytest.mark.skipif(unsupport_quadprog(), reason='`quadprog` not supported on Windows.'
                    'You may have luck with `qpsolvers` instead but we will not test it here.')
@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('model', ['gem', 'agem', 'agem_r'])
def test_gem(dataset, model):
    sys.argv = ['mammoth',
                '--model',
                model,
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
