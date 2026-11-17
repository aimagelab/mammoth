import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.loggers import log_extra_metrics


@pytest.mark.parametrize('setting', ['domain-il', 'general-continual', 'biased-class-il'])
def test_log_extra_metrics_without_task_il(setting, capsys):
    # Settings without Task-IL return `None` as the masked metric (issue #75)
    log_extra_metrics(SimpleNamespace(nowand=True), -4.04, None, 'Backward Transfer', 0, setting=setting)
    err = capsys.readouterr().err
    assert 'Task-IL' not in err
    expected_label = 'Domain-IL' if setting != 'biased-class-il' else 'Class-IL'
    assert f'Backward Transfer: [{expected_label}]: -4.04' in err


def test_log_extra_metrics_class_il(capsys):
    log_extra_metrics(SimpleNamespace(nowand=True), -89.06, -1.56, 'Backward Transfer', 0)
    err = capsys.readouterr().err
    assert 'Backward Transfer: [Class-IL]: -89.06 \t [Task-IL]: -1.56' in err


@pytest.mark.parametrize('dataset', ['perm-mnist', 'seq-mnist'])
def test_enable_other_metrics(dataset):
    sys.argv = ['mammoth',
                '--model', 'sgd',
                '--dataset', dataset,
                '--lr', '0.1',
                '--batch_size', '128',
                '--n_epochs', '1',
                '--seed', '0',
                '--enable_other_metrics', '1',
                '--non_verbose', '1',
                '--debug_mode', '1']

    from main import main
    main()
