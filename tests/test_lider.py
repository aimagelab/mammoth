import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_gdumb_lider():
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

    main()


def test_icarl_lider():
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

    main()


def test_erace_lider():
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

    main()


def test_derpp_lider():
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

    main()
