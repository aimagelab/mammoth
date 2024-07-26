Scripts
=======

Mammoth includes a couple of scripts (under the ``scripts`` folder) to help you with your development.

- ``scripts/prepare_grid.py``: this script contains a ``grid_combinations`` dictionary, which contains all the hyper-parameters you want to test and their possible values. It will generate a ``data/jobs/<experiment_name>.txt`` file containing all the possible combinations of hyper-parameters. You can then use this file to launch your experiments on the grid (see below).

- ``scripts/local_launcher.py``: this script will launch all the experiments in the ``data/jobs/<experiment_name>.txt`` file in paralel on your local machine. Logs for each experiment will be stored in the ``logs`` folder. It accepts the following arguments:
    - ``--file``: path to the file containing the experiments to run (default: ``data/jobs/<experiment_name>.txt``)

    - ``--cycles``: number of times each experiment should be repeated (default: 1)

    - ``--at_a_time``: number of experiments to run in parallel (default: 1)

    - ``--start_from``: index of the first experiment to run (default: 0)

    - ``--reverse``: if set, the experiments will be run in reverse order (default: False)

- ``scripts/slurm_sbatcher.py``: this script will launch all the experiments in the ``data/jobs/<experiment_name>.txt`` file on a SLURM cluster. By default, the standard output and standard error will be redirected to a ``out`` and ``err`` folder respectively. The main arguments it accepts are:
    - ``--file``: path to the file containing the experiments to run (default: ``data/jobs/<experiment_name>.txt``)

    - ``--at_a_time``: number of experiments to run in parallel for each slurm job (default: 1)

    - ``--cycles``: number of times each experiment should be repeated (default: 1)

    - ``--name``: name of the slurm job (default: ``mammoth``)

    - ``--partition``: name of the slurm partition (default: ``gpu``)

    - ``--account``: maximum time for each slurm job (default: ``1-00:00:00``)

    - ``--dry``: if set, the slurm jobs will not be submitted (default: False)

- ``scripts/wandb_sync.py``: this script is used to facilitate syncing the logs produced by WandB (useful if WandB was set to `offline`). Instead of sequentially syncing the logs for each experiment, this script will sync the logs for all the experiments in parallel. It accepts the following arguments:
    - ``--n_workers``: number of workers to use (default: 4*number of cores)

    - ``--limit``: maximum number of experiments to sync (default: None)

    - ``--reverse``: if set, the experiments will be synced in reverse order (default: False)

