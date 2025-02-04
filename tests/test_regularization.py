import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('model', ['ewc_on'])
def test_ewc(dataset, model):
    sys.argv = ['mammoth',
                '--model',
                model,
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
                '--e_lambda',
                '0.5',
                '--gamma',
                '1',
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


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('model', ['si'])
def test_si(dataset, model):
    sys.argv = ['mammoth',
                '--model',
                model,
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
                '--c',
                '0.5',
                '--xi',
                '1',
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


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('model', ['lwf_mc', 'lwf'])
def test_lwf(dataset, model):
    sys.argv = ['mammoth',
                '--model',
                model,
                '--dataset',
                dataset,
                '--lr',
                '1e-4',
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
