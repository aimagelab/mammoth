import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_first_and_second_stage(capsys):
    sys.argv = ['mammoth',
                '--model',
                'first_stage_starprompt',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '0.002',
                '--n_epochs',
                '2',
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

    main()

    _, err = capsys.readouterr()

    # read output file and search for the string 'Saved text-encoder keys in:'
    ckpt_name = [line for line in err.splitlines() if 'Saved text-encoder keys in:' in line]
    assert any(ckpt_name), f'Keys not found'

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
                '2',
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

    main()

    # REMOVE CHECKPOINT FILE
    os.remove(ckpt_path)


def test_full_starprompt():
    sys.argv = ['mammoth',
                '--model',
                'starprompt',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-4',
                '--optimizer',
                'adam',
                '--n_epochs',
                '2',
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '1993',
                '--debug_mode',
                '1']

    main()
