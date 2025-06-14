How to Run Mammoth
==================

This section describes how to run experiments with Mammoth.

Basic Usage - Running from CLI
------------------------------

From the project root, you can run:

.. code-block:: bash

    python main.py --model <model_name> --dataset <dataset_name> [options]

or equivalently:

.. code-block:: bash

    python utils/main.py --model <model_name> --dataset <dataset_name> [options]
    # The `utils/main.py` is here to stay for backward compatibility.

You can list all available arguments by running:

.. code-block:: bash

    python main.py --help

This will tell you about the available models, datasets. For method- or dataset-specific options, you can run:

.. code-block:: bash

    python main.py --model <model_name> --dataset <dataset_name> --help

Common Options
~~~~~~~~~~~~~~

- ``--model``: name of the continual learning method
- ``--dataset``: name of the dataset
- ``--lr``: learning rate
- ``--savecheck``: save a checkpoint at the end of training (``last``) or at the end of each task (``task``)
- ``--validation``: reserve a percentage of the training data for each class for validation
- ``--wandb_entity`` and ``--wandb_project``: specify the Weights & Biases entity and project for logging

For more details on arguments, see :ref:`module-utils`.

Examples
~~~~~~~~

Run DER++ on seq-cifar100:

.. code-block:: bash

    python main.py --model derpp --dataset seq-cifar100 --buffer_size 500 --lr 0.1

Run with best hyperparameters:

.. code-block:: bash

    python main.py --model derpp --dataset seq-cifar100 --buffer_size 500 --model_config best

Running Mammoth as a Library
----------------------------

You can also use Mammoth programmatically in Python scripts or interactive sessions. Here's a simple Python example:

.. code-block:: python

    # Import Mammoth functions
    from mammoth import train, load_runner, get_avail_args

    # Inspect available arguments for a specific model and dataset
    required_args, optional_args = get_avail_args(dataset='seq-cifar10', model='sgd')
    print('Required arguments:', required_args)
    print('Optional arguments:', optional_args)

    # Load runner for a particular model and dataset
    model, dataset = load_runner(
        'sgd', 'seq-cifar10', # The model and dataset names
        {'lr': 0.1, 'n_epochs': 1, 'batch_size': 32} # Specify any additional arguments here
    )

    # Train the model
    train(model, dataset)

See the `examples/notebooks/basics.ipynb <../../examples/notebooks/basics.ipynb>`_ for a full notebook version.

.. note::

    Differently from the CLI, the Python API does not support capturing the SIGINT signal (Ctrl+C) to gracefully stop the training. 

    Sending a SIGINT signal will stop the training gracefully, allowing to keep the current state of the model and dataset. However, it will not save the checkpoint, so you will need to save it manually if needed.

See Also
--------

- :doc:`Reproducibility <getting_started/reproducibility>`
- :doc:`Checkpoints <getting_started/checkpoints>`
- :doc:`Fast Training <getting_started/fast_training>`
- :doc:`Scripts <getting_started/scripts>`
