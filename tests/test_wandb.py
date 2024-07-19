import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
from utils.test_utils import init_test_environ
import pytest


@init_test_environ
def test_wandb_log_erace():
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
                '--wandb_project',
                'mammoth-test',
                '--wandb_entity',
                'mammoth-test']

    os.environ['WANDB_MODE'] = 'disabled'

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_wandb_log_erace.log')

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(log_path, 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
