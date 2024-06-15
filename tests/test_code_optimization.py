import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('code_optimization', [0, 1, 2, 3])
def test_code_optim_erace(code_optimization):
    sys.argv = ['mammoth',
                '--model',
                'er-ace',
                '--buffer_size',
                '50',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-3',
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
                '1',
                '--code_optimization',
                str(code_optimization)]

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_code_optimization.O{code_optimization}.er-ace.seq-cifar10.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()


@pytest.mark.parametrize('code_optimization', [0, 1, 2, 3])
def test_code_optimization_slca(code_optimization):
    sys.argv = ['mammoth',
                '--model',
                'slca',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-3',
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
                '1',
                '--code_optimization',
                str(code_optimization)]

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_code_optimization.{code_optimization}.slca.seq-cifar10-224.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
