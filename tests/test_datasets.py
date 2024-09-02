import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest
import yaml


@pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'seq-cifar100', 'seq-tinyimg',
                                     'rot-mnist', 'perm-mnist', 'mnist-360', 'seq-cifar100-224',
                                     'seq-cifar10-224', 'seq-cifar100-224-rs', 'seq-cub200-rs',
                                     'seq-cifar100-224-rs', 'seq-tinyimg-r', 'seq-cub200', 'seq-imagenet-r',
                                     'seq-cars196', 'seq-chestx', 'seq-cropdisease', 'seq-eurosat-rgb',
                                     'seq-isic', 'seq-mit67', 'seq-resisc45'])
def test_datasets(dataset):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
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
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--debug_mode',
                '1']

    # clean all downloaded datasets
    dataset_paths = ['CUB200', 'CIFAR10', 'CIFAR100', 'MNIST',
                     'TINYIMG', 'imagenet-r', 'cars196', 'chestx',
                     'cropdisease', 'eurosat', 'isic', 'MIT67',
                     'NWPU-RESISC45']
    basepath = os.path.dirname(os.path.abspath(__file__))
    dt_dir = os.path.join(os.path.dirname(basepath), 'data')
    for path in dataset_paths:
        if os.path.exists(os.path.join(dt_dir, path)):
            os.system(f'rm -rf {os.path.join(dt_dir, path)}')

    main()


def test_dataset_workers():
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()


def test_configs_1(capsys):
    sys.argv = ['mammoth',
                '--model',
                'puridiver',
                '--buffer_size',
                '50',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-4',
                '--dataset_config',
                'test',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--batch_size',
                '5',
                '--noise_rate',
                '0.4',
                '--noise_type',
                'symmetric',
                '--optim_wd',
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
    assert namespace_dict['dataset_config'] == "test", f"Config 'test' not found in output"

    # load config file
    config_file = namespace_dict['dataset_config'].strip()
    # assert config_file == 'test', 'Config file not found in output'

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'configs', 'seq-cifar10', f'{config_file}.yaml')
    assert os.path.exists(config_path), f'Config file {config_file}.yaml not found in datasets/configs folder'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        assert config, 'Config file is empty'

    # check if the config file is loaded correctly:
    # - `batch_size` should be 5 (from CLI arguments, overriding puridiver's defaults)
    # - `optim_mom` should be 0.9 (from puridiver's defaults)
    # - `n_epochs` should be 1 (from puridiver's defaults, overriding config file)
    assert namespace_dict['batch_size'] == '5', f'batch_size not loaded correctly from CLI. Found {namespace_dict["batch_size"]} but expected 5'
    assert namespace_dict['optim_mom'] == '0.9', f'optim_mom not loaded correctly from puridiver defaults. Found {namespace_dict["optim_mom"]} but expected 0.9'
    assert namespace_dict['n_epochs'] == '1', f'n_epochs not loaded correctly from puridiver defaults. Found {namespace_dict["n_epochs"]} but expected 1'


def test_configs_2(capsys):
    sys.argv = ['mammoth',
                '--model',
                'ccic',
                '--buffer_size',
                '50',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-4',
                '--dataset_config',
                'test',
                '--non_verbose',
                '1',
                '--seed',
                '0',
                '--num_workers',
                '0',
                '--batch_size',
                '5',
                '--noise_rate',
                '0.4',
                '--noise_type',
                'symmetric',
                '--optim_wd',
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
    assert namespace_dict['dataset_config'] == "test", f"Config 'test' not found in output"

    # load config file
    config_file = namespace_dict['dataset_config'].strip()
    # assert config_file == 'test', 'Config file not found in output'

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'configs', 'seq-cifar10', f'{config_file}.yaml')
    assert os.path.exists(config_path), f'Config file {config_file}.yaml not found in datasets/configs folder'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        assert config, 'Config file is empty'

    # check if the config file is loaded correctly:
    # - `n_epochs` should be 1 (from dataset_config file)
    # - `batch_size` should be 5 (from CLI arguments)
    # - `optim_mom` should be 0.9 (from ccic's defaults)
    # - `optim_wd` should be 0 (from CLI arguments, overriding ccic's defaults)
    assert namespace_dict['n_epochs'] == str(config['n_epochs']), f'n_epochs not loaded correctly from config file. Found {namespace_dict["n_epochs"]} but expected {config["batch_size"]}'
    assert namespace_dict['batch_size'] == '5', f'batch_size not loaded correctly from CLI. Found {namespace_dict["batch_size"]} but expected 5'
