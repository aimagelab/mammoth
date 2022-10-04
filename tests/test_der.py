import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main, parse_args
import pytest


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'seq-cifar100', 'seq-tinyimg', 'rot-mnist', 'perm-mnist', 'mnist-360'])
def test_der(dataset):
    sys.argv = ['mammoth',
                '--model',
                'der',
                '--dataset',
                dataset,
                '--buffer_size',
                '10',
                '--lr',
                '1e-4',
                '--alpha',
                '.5',
                '--n_epochs',
                '1',
                '--debug_mode',
                '1']
    a = parse_args()

    main(a)
