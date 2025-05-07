import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_spr_base():
    sys.argv = ['mammoth',
                '--model',
                'spr',
                '--dataset',
                'seq-mnist',
                '--lr',
                '1e-3',
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
                '--spr_debug_mode',
                '1',
                '--debug_mode',
                '1']

    main()


def test_spr_best():
    sys.argv = ['mammoth',
                '--model',
                'spr',
                '--dataset',
                'seq-mnist',
                '--model_config',
                'best',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--buffer_size',
                '300',
                '--spr_debug_mode',
                '1',
                '--debug_mode',
                '1']

    main()
