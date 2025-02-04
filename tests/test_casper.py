import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_erace_casper():
    sys.argv = ['mammoth',
                '--model',
                'er_ace_casper',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '200',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '32',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()


def test_icarl_casper():
    sys.argv = ['mammoth',
                '--model',
                'icarl_casper',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '200',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '32',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()


def test_derpp_casper():
    sys.argv = ['mammoth',
                '--model',
                'derpp_casper',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '200',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--alpha',
                '0.5',
                '--beta',
                '0.5',
                '--batch_size',
                '32',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()


def test_xder_casper():
    sys.argv = ['mammoth',
                '--model',
                'xder_rpc_casper',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '200',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--alpha',
                '0.5',
                '--beta',
                '0.5',
                '--batch_size',
                '32',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    main()
