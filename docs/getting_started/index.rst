First steps
===============

1. Install the requirements with ``pip install -r requirements.txt``.

2. From the root directory, run ``python utils/main.py --help`` to see the available options.

Results and logs - WandB
------------------------

Mammoth logs all the results and metrics under the ``data/results`` directory (by default). You can change this directory by changing the **base_path** function in :ref:`module-conf`. 

- The logs are organized in the following way: `<setting>/<dataset>/<model>/logs.pyd`.

- Each line in the log file is a dictionary containing the arguments and results for a single run.

WandB
~~~~~

For advanced logging, including loss values, metrics, and hyperparameters, you can use [WandB](https://wandb.ai/) by providing both ``--wandb_project`` and ``--wandb_entity`` arguments. If you don't want to use WandB, you can simply omit these arguments.

.. tip::
    By default, all arguments, loss values, and metrics are logged. Thanks to the **autolog_wandb** (:ref:`module-model`), all the variables created in the **observe** that start with *loss* or *_wandb_* will be logged. Thus, in order to loss all the separate loss values, you can simply add ``loss = loss + loss1 + loss2`` to the **observe** function.
