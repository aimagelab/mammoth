import logging
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


@pytest.mark.parametrize('backbone', ['resnet18', 'resnet34'])
def test_register_backbone_resnet(backbone, caplog):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
                '--lr',
                '1e-3',
                '--backbone',
                backbone,
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()

    namespace_args = [line.split('Namespace(')[-1] for line in caplog.text.splitlines() if 'Namespace(' in line]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['backbone'] == backbone, f'Dynamic argument `backbone` not loaded correctly, expected {backbone} but got {namespace_dict["backbone"]}'

    backbone_print = [line for line in caplog.text.splitlines() if 'using backbone: ' in line.lower()]
    assert any(backbone_print), 'Backbone not printed'

    assert f'Using backbone: {backbone}' in backbone_print[0], f'Expected backbone {backbone} not found in log'


@pytest.mark.parametrize('backbone', ['vit', 'resnet50'])
def test_register_backbone_big(backbone, caplog):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10-224',
                '--lr',
                '1e-3',
                '--backbone',
                backbone,
                '--batch_size',
                '2',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    main()

    namespace_args = [line.split('Namespace(')[-1] for line in caplog.text.splitlines() if 'Namespace(' in line]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['backbone'] == backbone, f'Dynamic argument `backbone` not loaded correctly, expected {backbone} but got {namespace_dict["backbone"]}'

    backbone_print = [line for line in caplog.text.splitlines() if 'using backbone: ' in line.lower()]
    assert any(backbone_print), 'Backbone not printed'

    assert f'Using backbone: {backbone}' in backbone_print[0], f'Expected backbone {backbone} not found in log'


def test_args_with_choices(capsys):
    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
                '--backbone',
                'efficientnet',
                '--help']
    
    with pytest.raises(SystemExit):
        main()

    lines = capsys.readouterr().out.splitlines()
    bk_line = [line for line in lines if 'efficientnet_type' in line]
    assert len(bk_line) >= 1, 'Backbone type choices not printed correctly'

def test_register_backbone(capsys):
    from backbone import register_backbone, MammothBackbone
    @register_backbone('custom-cnn')
    class CustomCNN(MammothBackbone):
        def __init__(self, num_layers=3, num_filters=64):
            self.num_layers = num_layers
            self.num_filters = num_filters

        def forward(self, x):
            # Dummy forward method
            return x

    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'seq-cifar10',
                '--backbone',
                'custom-cnn',
                '--help']
    
    with pytest.raises(SystemExit):
        main()

    lines = capsys.readouterr().out.splitlines()
    bk_line = [line.strip() for line in lines if line.strip().startswith('--backbone')]
    assert len(bk_line) >= 1, 'Backbone type choices not printed correctly'
    assert 'custom-cnn' in bk_line[0], 'Custom backbone not registered correctly'
    args_line = [line for line in lines if line.strip().startswith('--num_layers') or line.strip().startswith('--num_filters')]
    assert len(args_line) >= 2, 'Custom backbone arguments not printed correctly'

def test_register_dataset(capsys):
    from datasets import register_dataset, ContinualDataset

    @register_dataset('custom-dataset')
    class CustomDataset(ContinualDataset):
        def __init__(self, num_classes=10, custom_dataset_arg='default'):
            self.num_classes = num_classes
            self.custom_dataset_arg = custom_dataset_arg

        def get_data_loaders(self):
            pass

    sys.argv = ['mammoth',
                '--model',
                'sgd',
                '--dataset',
                'custom-dataset',
                '--backbone',
                'resnet18',
                '--help']
    
    with pytest.raises(SystemExit):
        main()

    lines = capsys.readouterr().out.splitlines()
    ds_line = [line.strip() for line in lines if line.strip().startswith('--dataset')]
    assert len(ds_line) >= 1, 'Dataset type choices not printed correctly'
    assert 'custom-dataset' in ds_line[0], 'Custom dataset not registered correctly'
    args_line = [line for line in lines if line.strip().startswith('--custom_dataset_arg')]
    assert len(args_line) >= 1, 'Custom dataset arguments not printed correctly'
