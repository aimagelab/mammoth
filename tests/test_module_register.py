import logging
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


@pytest.mark.parametrize('backbone', ['resnet18', 'resnet34'])
def test_register_backbone_resnet(backbone, capsys, caplog):
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

    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)

    main()

    out, err = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    namespace_args = [line for line in out.splitlines() if line.startswith('Namespace(')]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['backbone'] == backbone, f'Dynamic argument `backbone` not loaded correctly, expected {backbone} but got {namespace_dict["backbone"]}'

    backbone_print = [line for line in caplog.text.splitlines() if 'using backbone: ' in line.lower()]
    assert any(backbone_print), 'Backbone not printed'

    assert f'Using backbone: {backbone}' in backbone_print[0], f'Expected backbone {backbone} not found in log'


@pytest.mark.parametrize('backbone', ['vit', 'resnet50'])
def test_register_backbone_big(backbone, capsys, caplog):
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

    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)

    main()

    out, err = capsys.readouterr()
    # read output file and search for the string 'Saving checkpoint into'
    namespace_args = [line for line in out.splitlines() if line.startswith('Namespace(')]
    assert any(namespace_args), 'Arguments not found in output'

    namespace_args = namespace_args[0].replace('Namespace(', '').replace(')', '')
    namespace_args = namespace_args.split(', ')
    namespace_dict = {arg.split('=')[0]: arg.split('=')[1].replace("'", "").strip() for arg in namespace_args}

    assert namespace_dict['backbone'] == backbone, f'Dynamic argument `backbone` not loaded correctly, expected {backbone} but got {namespace_dict["backbone"]}'

    backbone_print = [line for line in caplog.text.splitlines() if 'using backbone: ' in line.lower()]
    assert any(backbone_print), 'Backbone not printed'

    assert f'Using backbone: {backbone}' in backbone_print[0], f'Expected backbone {backbone} not found in log'
