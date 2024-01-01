import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar100', 'seq-tinyimg'])  # , 'seq-cub200'
@pytest.mark.parametrize('resize_maps', ['0', '1'])
def test_twf_random_init(dataset, resize_maps):
    sys.argv = ['mammoth',
                '--model',
                'twf',
                '--dataset',
                dataset,
                "--buffer_size",
                "50",
                '--der_alpha',
                '0.5',
                '--der_beta',
                '0.5',
                '--lambda_fp',
                '0.01',
                '--lambda_diverse_loss',
                '0.1',
                '--lambda_fp_replay',
                '0.1',
                '--resize_maps',
                resize_maps,
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--virtual_bs_iterations',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_twf_random_init.{dataset}.resize_maps_{resize_maps}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()


@pytest.mark.parametrize(('dataset', 'loadcheck'),
                         [('seq-cifar100', 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EeWEOSls505AsMCTXAxWoLUBmeIjCiplFl40zDOCmB_lEw?e=Izv0jh'),
                          ('seq-cub200', 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EV7I5BpJvURIhMMk95r3x5YBAZKch-NPFEJ9hhPQghcWCw?e=dt8wp3'),
                         ('seq-cifar10', 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EWttSkmKfkNEpEWNiPoS3zUB6uzZydc0irOW0Xbu3jtr3Q?e=JQ6Fay')])
@pytest.mark.parametrize('resize_maps', ['0', '1'])
def test_twf_with_checkpoint(dataset, loadcheck, resize_maps):
    sys.argv = ['mammoth',
                '--model',
                'twf',
                '--dataset',
                dataset,
                "--buffer_size",
                "50",
                '--der_alpha',
                '0.5',
                '--der_beta',
                '0.5',
                '--lambda_fp',
                '0.01',
                '--lambda_diverse_loss',
                '0.3',
                '--lambda_fp_replay',
                '0.5',
                '--resize_maps',
                resize_maps,
                '--loadcheck',
                loadcheck,
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--virtual_bs_iterations',
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
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_twf_with_checkpoint.{dataset}.resize_maps_{resize_maps}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout

    main()
