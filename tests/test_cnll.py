import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_cnll_base():
    sys.argv = ['mammoth',
                '--model',
                'cnll',
                '--dataset',
                'seq-cifar10',
                '--n_epochs',
                '1',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--buffer_size',
                '300',
                '--noise_type',
                'sym',
                '--noise_rate',
                '0.4',
                '--cnll_debug_mode',
                '1',
                '--debug_mode',
                '1']

    main()


def test_cnll_best():
    sys.argv = ['mammoth',
                '--model',
                'cnll',
                '--dataset',
                'seq-cifar10',
                '--model_config',
                'best',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--buffer_size',
                '500',
                '--cnll_debug_mode',
                '1',
                '--debug_mode',
                '1']

    main()
