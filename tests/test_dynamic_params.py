import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main


def test_dynamic_args_backbone(capsys, caplog):
    sys.argv = ['mammoth',
                '--model',
                'si',
                '--dataset',
                'perm-mnist',
                '--lr',
                '1e-3',
                '--n_epochs',
                '10',
                '--xi',
                '0.1',
                '--c',
                '0.1',
                '--batch_size',
                '2',
                '--mlp_hidden_size',
                '2000',
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

    assert namespace_dict['mlp_hidden_size'] == '2000', f'Dynamic argument `mlp_hidden_size` not loaded correctly, expected 2000 but got {namespace_dict["mlp_hidden_size"]}'

    hidden_size_print = [line for line in caplog.text.splitlines() if 'hidden size is set to' in line.lower()]
    assert any(hidden_size_print), 'Hidden size print not found in output'

    assert 'hidden size is set to `2000` instead of the default `100`' in hidden_size_print[0], f'Hidden size print not as expected, got {hidden_size_print[0]}'