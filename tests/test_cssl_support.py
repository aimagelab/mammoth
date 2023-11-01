import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main, parse_args
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-tinyimg'])
@pytest.mark.parametrize('label_perc', ['0.1', '0.08', '0.5', '1'])
def test_cssl_support(dataset, label_perc):
    sys.argv = ['mammoth',
                '--model',
                'er',
                '--dataset',
                dataset,
                '--buffer_size',
                '10',
                '--label_perc',
                label_perc,
                '--lr',
                '1e-4',
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_cssl_support.{dataset}.{label_perc}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()
