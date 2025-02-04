First steps
===============

1. Install the requirements with ``pip install -r requirements.txt``.

2. From the root directory, run ``python utils/main.py --help`` to see the available options.

See :ref:`module-utils` for more information about the most useful arguments

Results and logs - WandB
------------------------

Mammoth logs all the results and metrics under the ``data/results`` directory (by default). You can change this directory by changing the **base_path** function in :ref:`module-utils.conf`. 

- The logs are organized in the following way: `<setting>/<dataset>/<model>/logs.pyd`.

- Each line in the log file is a dictionary containing the arguments and results for a single run.

WandB
~~~~~

For advanced logging, including loss values, metrics, and hyperparameters, you can use `WandB <https://wandb.ai/>`_ by providing both ``--wandb_project`` and ``--wandb_entity`` arguments. If you don't want to use WandB, you can simply omit these arguments.

.. tip::
    By default, all arguments, loss values, and metrics are logged. Thanks to the **autolog_wandb** (:ref:`module-models`), all the variables created in the **observe** that contain *loss* or start with *_wandb_* will be logged. Thus, in order to log all the separate loss values, you can simply add ``loss = loss + loss1 + loss2`` to the **observe** function.

Metrics are logged on WandB both in a raw form, separated for each task and class. This allows further analysis (e.g., with the Mammoth :ref:`Parseval <module-parseval>`). To differentiate between raw metrics logged on WandB and other aggregated metrics that may have been logged, all the raw metrics are prefixed with **RESULTS_**. This behavior can be changed by changing the prefix in the **log_accs** function in :ref:`module-utils.loggers`.

Testing
-------

Mammoth includes a few tests to ensure that the code is working as expected for all available models and datasets. The tests are run using `pytest` and can be run using the following command:

.. code-block:: bash

    pytest

The tests are quite long, as they evaluate most of the functionality of Mammoth. The estimated runtime is about 2 hours.

.. important::

    By default, the tests will NOT delete all the downloaded datasets and re-download them (`test_datasets_with_download` is disabled). To enable this test, you can run pytest with the ``--include_dataset_reload`` argument.

    Note that this test will download all the datasets, which can take a significant amount of time and space.