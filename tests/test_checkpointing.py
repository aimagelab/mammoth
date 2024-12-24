import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import pytest


@pytest.mark.parametrize('model', ['sgd', 'slca', 'l2p'])
@pytest.mark.parametrize('savecheck', ['last', 'task'])
@pytest.mark.parametrize('joint', ['0', '1'])
def test_checkpoint_save_and_load(model, savecheck, joint, capsys):
    N_TASKS = 5  # cifar10

    checkpoint_name = str(uuid.uuid4())

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
                '--ckpt_name',
                checkpoint_name,
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

    main()

    _, err = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    ckpt_name = [line for line in err.splitlines() if 'Saving checkpoint into' in line]
    assert any(ckpt_name), f'Checkpoint not saved for model {model}'

    if joint == '0':
        if savecheck == 'last':
            ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_last.pt'
        elif savecheck == 'task':
            ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_{N_TASKS-1}.pt'
    elif joint == '1':
        ckpt_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip() + f'_joint.pt'

    ckpt_path = os.path.join('checkpoints', ckpt_name)

    assert checkpoint_name in ckpt_path, f'Checkpoint does not contain the checkpoint name. Found {ckpt_path} but expected {checkpoint_name} in the path'
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


@pytest.mark.parametrize('savecheck', ['last', 'task'])
@pytest.mark.parametrize('joint', ['0', '1'])
def test_checkpointing_replay(savecheck, joint, capsys):
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

    main()

    _, err = capsys.readouterr()

    # read output file and search for the string 'Saving checkpoint into'
    ckpt_name = [line for line in err.splitlines() if 'Saving checkpoint into' in line]
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
