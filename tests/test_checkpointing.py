import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
from utils.test_utils import init_test_environ
import pytest


@init_test_environ
@pytest.mark.parametrize('model', ['sgd', 'slca', 'l2p'])
@pytest.mark.parametrize('savecheck', ['last', 'task'])
@pytest.mark.parametrize('joint', ['0', '1'])
def test_checkpointing_bufferfree(model, savecheck, joint):
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
                '--joint',
                joint,
                '--savecheck',
                savecheck,
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.{model}.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.{model}.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved for model {model}'

        if joint == '0':
            if savecheck == 'last':
                ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_last.pt'
            elif savecheck == 'task':
                ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_{N_TASKS-1}.pt'
        elif joint == '1':
            ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_joint.pt'

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
                '--joint',
                joint,
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_load.{model}.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # REMOVE CHECKPOINT FILE
    if joint == '0':
        if savecheck == 'task':
            for i in range(N_TASKS):
                c_path = ckpt_path.split(f'_{N_TASKS-1}.pt')[0] + f'_{i}.pt'
                os.remove(c_path)
        elif savecheck == 'last':
            os.remove(ckpt_path)
    elif joint == '1':
        os.remove(ckpt_path)


@init_test_environ
@pytest.mark.parametrize('savecheck', ['last', 'task'])
@pytest.mark.parametrize('joint', ['0', '1'])
def test_checkpointing_replay(savecheck, joint):
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
                '--joint',
                joint,
                '--n_epochs',
                '1',
                '--buffer_size',
                '50',
                '--savecheck',
                savecheck,
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.derpp.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_save.derpp.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved for derpp'

        if joint == '0':
            if savecheck == 'last':
                ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_last.pt'
            elif savecheck == 'task':
                ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_{N_TASKS-1}.pt'
        elif joint == '1':
            ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_joint.pt'

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
                '--joint',
                joint,
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_checkpoint_load.derpp.{savecheck}.{"joint" if joint=="1" else "cl"}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # REMOVE CHECKPOINT FILE
    if joint == '0':
        if savecheck == 'task':
            for i in range(N_TASKS):
                c_path = ckpt_path.split(f'_{N_TASKS-1}.pt')[0] + f'_{i}.pt'
                os.remove(c_path)
        elif savecheck == 'last':
            os.remove(ckpt_path)
    elif joint == '1':
        os.remove(ckpt_path)
