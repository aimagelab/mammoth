import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_gdumb_cutmix():
    sys.argv = ['mammoth',
                '--model',
                'gdumb',
                '--dataset',
                'seq-cifar10',
                '--fitting_epochs',
                '2',
                '--cutmix_alpha',
                '0.3',
                '--buffer_size',
                '50',
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


def test_gdumb():
    sys.argv = ['mammoth',
                '--model',
                'gdumb',
                '--dataset',
                'seq-cifar10',
                '--fitting_epochs',
                '2',
                '--buffer_size',
                '50',
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
