import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main


def test_default(capsys):
    sys.argv = ['mammoth',
                '--model',
                'ccic',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
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

    main()

    out, _ = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    namespace_args = [line for line in out.splitlines() if line.startswith('Namespace(')]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['optimizer'] == 'adam', f'Optimizer not loaded correctly, expected adam but got {namespace_dict["optimizer"]}'


def test_default_overwrite(capsys):
    sys.argv = ['mammoth',
                '--model',
                'ccic',
                '--dataset',
                'seq-cifar10',
                '--model_config',
                'base',  # alias of `default`
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--optimizer',
                'adamw',
                '--n_epochs',
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

    main()

    out, _ = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    namespace_args = [line for line in out.splitlines() if line.startswith('Namespace(')]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['optimizer'] == 'adamw', f'Optimizer not loaded correctly, expected adamw but got {namespace_dict["optimizer"]}'


def test_best(capsys):
    sys.argv = ['mammoth',
                '--model',
                'ccic',
                '--dataset',
                'seq-cifar10',
                '--buffer_size',
                '500',
                '--dataset_config',
                'test',
                '--model_config',
                'best',
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

    out, _ = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    namespace_args = [line for line in out.splitlines() if line.startswith('Namespace(')]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['optimizer'] == 'adam', f'Otimizer not loaded correctly, expected adam but got {namespace_dict["optimizer"]}'
    assert namespace_dict['lr'] == '0.0001', f'Learning rate not loaded correctly, expected 0.0001 but got {namespace_dict["lr"]}'
    assert namespace_dict['batch_size'] == '4', f'Batch size not loaded correctly, expected 4 but got {namespace_dict["batch_size"]}'
    assert namespace_dict['n_epochs'] == '50', f'Number of epochs not loaded correctly, expected 50 but got {namespace_dict["n_epochs"]}'
