import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main, parse_args
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10-224', 'seq-imagenet-r'])
@pytest.mark.parametrize('code_optimization', [0, 1])
def test_codaprompt(dataset, code_optimization):
    sys.argv = ['mammoth',
                '--model',
                'coda_prompt',
                '--dataset',
                dataset,
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
                '--code_optimization',
                str(code_optimization),
                '--debug_mode',
                '1']

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_codaprompt.{dataset}.O{code_optimization}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()
