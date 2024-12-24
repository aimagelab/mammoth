import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('split_id', [1, 2])
def test_lws(split_id):
    sys.argv = ['mammoth',
                '--model',
                'lws',
                '--dataset',
                'seq-celeba',
                '--split_id',
                str(split_id),
                '--lr',
                '1e-3',
                '--n_epochs',
                '2',
                '--batch_size',
                '2',
                '--buffer_size',
                '20',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()
