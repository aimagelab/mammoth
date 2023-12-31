import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


def test_icarl():
    sys.argv = ['mammoth',
                '--model',
                'gdumb-lider',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--alpha_lip_lambda',
                '0.5',
                '--beta_lip_lambda',
                '0.5',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_gdumb-lider.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()


def test_icarl():
    sys.argv = ['mammoth',
                '--model',
                'icarl-lider',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--alpha_lip_lambda',
                '0.5',
                '--beta_lip_lambda',
                '0.5',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_icarl-lider.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()


def test_erace():
    sys.argv = ['mammoth',
                '--model',
                'er-ace-lider',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--alpha_lip_lambda',
                '0.5',
                '--beta_lip_lambda',
                '0.5',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_er-ace-lider.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()


def test_derpp():
    sys.argv = ['mammoth',
                '--model',
                'derpp-lider',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--alpha',
                '0.5',
                '--beta',
                '0.5',
                '--batch_size',
                '4',
                '--alpha_lip_lambda',
                '0.5',
                '--beta_lip_lambda',
                '0.5',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_derpp.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
