import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


def test_main_import_rehearsal(caplog):
    sys.argv = ['mammoth',
                '--model',
                'er-ace',
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
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    from main import main
    main()

    # check if the main file was imported multiple times
    import_print = [line for line in caplog.text.splitlines() if 'Running Mammoth!' in line]
    assert len(import_print) == 0, 'Main file imported multiple times'


def test_main_import_base(caplog):
    sys.argv = ['mammoth',
                '--model',
                'lwf_mc',
                '--dataset',
                'seq-eurosat-rgb',
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

    from main import main
    main()

    # check if the main file was imported multiple times
    import_print = [line for line in caplog.text.splitlines() if 'Running Mammoth!' in line]
    assert len(import_print) == 0, 'Main file imported multiple times'
