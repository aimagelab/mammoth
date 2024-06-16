import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('model', ['sgd', 'slca', 'l2p'])
def test_checkpointing_bufferfree(model):
    N_TASKS = 5  # cifar10

    # TEST CHECKPOINT SAVE
    sys.argv = ['mammoth',
                '--model',
                model,
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--savecheck',
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.{model}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.{model}.log'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved for model {model}'

        ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_{N_TASKS-1}.pt'
        ckpt_path = os.path.join('checkpoints', ckpt_name)

        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

    # TEST CHECKPOINT LOAD
    sys.argv = ['mammoth',
                '--model',
                model,
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--loadcheck',
                ckpt_path,
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_load.{model}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # REMOVE CHECKPOINT FILE
    for i in range(N_TASKS):
        c_path = ckpt_path.split(f'_{N_TASKS-1}.pt')[0] + f'_{i}.pt'
        os.remove(c_path)


def test_checkpointing_replay():
    N_TASKS = 5  # cifar10

    # TEST CHECKPOINT SAVE
    sys.argv = ['mammoth',
                '--model',
                'derpp',
                '--dataset',
                'seq-cifar10',
                '--alpha',
                '0.1',
                '--beta',
                '0.1',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--buffer_size',
                '50',
                '--savecheck',
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.derpp.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.derpp.log'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved for derpp'

        ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_{N_TASKS-1}.pt'
        ckpt_path = os.path.join('checkpoints', ckpt_name)

        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

    # TEST CHECKPOINT LOAD
    sys.argv = ['mammoth',
                '--model',
                'derpp',
                '--dataset',
                'seq-cifar10',
                '--alpha',
                '0.1',
                '--beta',
                '0.1',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--buffer_size',
                '50',
                '--loadcheck',
                ckpt_path,
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

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_load.derpp.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # REMOVE CHECKPOINT FILE
    for i in range(N_TASKS):
        c_path = ckpt_path.split(f'_{N_TASKS-1}.pt')[0] + f'_{i}.pt'
        os.remove(c_path)
