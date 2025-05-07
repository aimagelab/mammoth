import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_derpp_star():
    sys.argv = ['mammoth',
                '--alpha',
                '0.2',
                '--beta',
                '0.1',
                '--buffer_size',
                '100',
                '--p-gamma',
                '0.1',
                '--p-lam',
                '0.005',
                '--model',
                'derpp_star',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '0.1',
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

    main()


def test_er_star():
    sys.argv = ['mammoth',
                '--buffer_size',
                '100',
                '--p-gamma',
                '0.01',
                '--p-lam',
                '0.1',
                '--model',
                'er_star',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '0.1',
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

    main()


def test_er_ace_star():
    sys.argv = ['mammoth',
                '--buffer_size',
                '100',
                '--p-gamma',
                '0.01',
                '--p-lam',
                '0.1',
                '--model',
                'er_ace_star',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '0.03',
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

    main()


def test_xder_rpc_star():
    sys.argv = ['mammoth',
                '--buffer_size',
                '100',
                '--alpha',
                '0.1',
                '--beta',
                '0.5',
                '--p-gamma',
                '0.001',
                '--p-lam',
                '0.01',
                '--model',
                'xder_rpc_star',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '0.03',
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

    main()
