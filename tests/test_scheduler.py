import os
import sys

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_der_cifar100_defaultscheduler(capsys):
    N_TASKS = 10
    sys.argv = ['mammoth',
                '--model',
                'der',
                '--dataset',
                'seq-cifar100',
                '--buffer_size',
                '500',
                '--alpha',
                '0.3',
                '--lr',
                '0.03',
                '--n_epochs',
                '50',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1',
                '--savecheck',
                'task',
                '--seed',
                '0']

    main()

    _, err = capsys.readouterr()

    # read output file and search for the string 'Saving checkpoint into'
    ckpt_name = [line for line in err.splitlines() if 'Saving checkpoint into' in line]
    assert any(ckpt_name), f'Checkpoint not saved'

    ckpt_base_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip()
    ckpt_paths = [os.path.join('checkpoints', ckpt_base_name + f'_{i}.pt') for i in range(N_TASKS)]

    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

        ckpt = torch.load(ckpt_path, weights_only=True)
        opt, sched = ckpt['optimizer']['param_groups'][0], ckpt['scheduler']
        assert opt['initial_lr'] == 0.03, f'Learning rate not updated correctly in {ckpt_path}'
        assert opt['lr'] == opt['initial_lr'] * 0.1 * 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert list(sched['milestones'].keys()) == [35, 45], f'Milestones not updated correctly in {ckpt_path}'
        assert sched['base_lrs'] == [0.03], f'Base learning rate not updated correctly in {ckpt_path}'
        assert 'buffer' in ckpt, f'Buffer not saved in {ckpt_path}'
        assert all([k in ckpt['buffer'].keys() for k in ['examples', 'logits']]), f'Buffer not saved correctly in {ckpt_path}'
        assert len(ckpt['buffer']['examples']) == 500, f'Buffer size not saved correctly in {ckpt_path}'


def test_der_cifar100_customscheduler(capsys):
    N_TASKS = 10
    sys.argv = ['mammoth',
                '--model',
                'der',
                '--dataset',
                'seq-cifar100',
                '--buffer_size',
                '500',
                '--alpha',
                '0.3',
                '--lr',
                '0.1',
                '--n_epochs',
                '10',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--debug_mode',
                '1',
                '--savecheck',
                'task',
                '--lr_scheduler',
                'multisteplr',
                '--lr_milestones',
                '2', '4', '6', '8',
                '--seed',
                '0']
    main()

    _, err = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    ckpt_name = [line for line in err.splitlines() if 'Saving checkpoint into' in line]
    assert any(ckpt_name), f'Checkpoint not saved'

    ckpt_base_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip()
    ckpt_paths = [os.path.join('checkpoints', ckpt_base_name + f'_{i}.pt') for i in range(N_TASKS)]

    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

        ckpt = torch.load(ckpt_path, weights_only=True)
        opt, sched = ckpt['optimizer']['param_groups'][0], ckpt['scheduler']
        assert opt['initial_lr'] == 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert opt['lr'] == opt['initial_lr'] * 0.1 * 0.1 * 0.1 * 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert list(sched['milestones'].keys()) == [2, 4, 6, 8], f'Milestones not updated correctly in {ckpt_path}'
        assert sched['base_lrs'] == [0.1], f'Base learning rate not updated correctly in {ckpt_path}'
