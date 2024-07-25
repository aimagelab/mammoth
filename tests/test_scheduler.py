import os
import sys

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


def test_der_cifar100_defaultscheduler():
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

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_der_cifar100_defaultscheduler.log')
    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(log_path, 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved in {log_path}'

        ckpt_base_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip()
        ckpt_paths = [os.path.join('checkpoints', ckpt_base_name + f'_{i}.pt') for i in range(N_TASKS)]

    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

        ckpt = torch.load(ckpt_path)
        opt, sched = ckpt['optimizer']['param_groups'][0], ckpt['scheduler']
        assert opt['initial_lr'] == 0.03, f'Learning rate not updated correctly in {ckpt_path}'
        assert opt['lr'] == opt['initial_lr'] * 0.1 * 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert list(sched['milestones'].keys()) == [35, 45], f'Milestones not updated correctly in {ckpt_path}'
        assert sched['base_lrs'] == [0.03], f'Base learning rate not updated correctly in {ckpt_path}'


def test_der_cifar100_customscheduler():
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

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_der_cifar100_customscheduler.der.cifar100.log')
    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(log_path, 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()

    # read output file and search for the string 'Saving checkpoint into'
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ckpt_name = [line for line in lines if 'Saving checkpoint into' in line]
        assert any(ckpt_name), f'Checkpoint not saved in {log_path}'

        ckpt_base_name = ckpt_name[0].split('Saving checkpoint into')[-1].strip()
        ckpt_paths = [os.path.join('checkpoints', ckpt_base_name + f'_{i}.pt') for i in range(N_TASKS)]

    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), f'Checkpoint file {ckpt_path} not found'

        ckpt = torch.load(ckpt_path)
        opt, sched = ckpt['optimizer']['param_groups'][0], ckpt['scheduler']
        assert opt['initial_lr'] == 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert opt['lr'] == opt['initial_lr'] * 0.1 * 0.1 * 0.1 * 0.1, f'Learning rate not updated correctly in {ckpt_path}'
        assert list(sched['milestones'].keys()) == [2, 4, 6, 8], f'Milestones not updated correctly in {ckpt_path}'
        assert sched['base_lrs'] == [0.1], f'Base learning rate not updated correctly in {ckpt_path}'
