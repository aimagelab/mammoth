import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
from utils.test_utils import init_test_environ
import pytest


@init_test_environ
def test_first_and_second_stage():
    sys.argv = ['mammoth',
                '--model',
                'first_stage_starprompt',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '0.002',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '1993',
                '--debug_mode',
                '1']
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_first_stage_starprompt.log')

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(fn, 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()

    # read output file and search for the string 'Saved text-encoder keys in:'
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saved text-encoder keys in:' in line]
        assert any(ckpt_name), f'Keys not found in {fn}'

        ckpt_path = ckpt_name[0].split('Saved text-encoder keys in:')[1].strip()

        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

    # TEST CHECKPOINT LOAD
    sys.argv = ['mammoth',
                '--model',
                'second_stage_starprompt',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-4',
                '--optimizer',
                'adam',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '1993',
                '--keys_ckpt_path',
                ckpt_path,
                '--debug_mode',
                '1']

    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_second_stage_starprompt.log')

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(fn, 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # REMOVE CHECKPOINT FILE
    os.remove(ckpt_path)
