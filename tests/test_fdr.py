import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


def test_fdr():
    sys.argv = ['mammoth',
                '--model',
                'fdr',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--alpha',
                '0.5',
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_fdr.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
