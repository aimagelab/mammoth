.. _module-utils:

Utils
======

This module contains a collection of utility classes and functions that are used throughout the library.

.. important::
    The most important module is `main.py`, which is the entry point for the library. 
    It contains the `main` function, which is called when the library is run as a script. 
    This function is responsible for parsing the command line arguments and calling the appropriate 
    functions to perform train and validation.

Running Mammoth
---------------

To run the library, simply run the `utils/main.py` script. There are a few command line arguments that can be used to customize the execution of the library. To see the full list of arguments, run the following command:

.. code-block:: bash

  python utils/main.py --help


The most important arguments are the following:

- ``--dataset``: the name of the dataset to use. The list of available datasets can be found in the `datasets` folder (or with ``--help``).

- ``--model``: the name of the model to run. The list of available models can be found in the `models` folder (or with ``--help``). Once the model is selected, its corresponding parser is loaded (see the `parse_args` function in :ref:`module-models`) and the model-specific arguments are available and shown with ``--help``.

- ``--lr``: the learning rate to use for training.

- ``--buffer_size`` (only required for rehearsal-based methods): the size of the replay buffer.

Other arguments such as the size of the training batch and the number of epochs are automatically loaded by the selected dataset (see :ref:`module-datasets`). However, the default values can be overridden by specifying the corresponding command line arguments. For example, to run the `er` model on the `seq-cifar10` dataset with a batch size of `128` and `10` epochs (instead of the default of `32` and `50` respectively), run the following command:

.. code-block:: bash

  python utils/main.py --dataset seq-cifar10 --model der --buffer_size 500 --lr 0.03 --batch_size 128 --epochs 10

.. note::
    To ease hyper-parameter tuning, all boolean arguments follow the convention: ``--<argument>=1`` for ``True`` and ``--<argument>=0`` for ``False``. The only exceptions are ``--savecheck`` and ``--inference_only``, as they should not be included in the hyper-parameter search.

Other useful arguments
~~~~~~~~~~~~~~~~

* ``--debug_mode``: If set to ``1``, the model will run for only a few iterations per each epoch and will disable WandB logging. This is useful for debugging.

* ``--num_workers**: The number of workers to use for the data loaders. If set to ``0``, the data loaders will run in the main process. This is useful for debugging.

* ``--seed``: The seed to use for the random number generators. If this is not set, the seed will be randomly generated.

* ``--permute_classes``: If set to ``1``, the classes will be randomly permuted before splitting them into tasks.

* ``--joint``: If set to ``1``, the supplied dataset will be treated as a single task. This usually serves as a upper bound for the performance of the model.

* ``--label_perc``: The percentage of labels to use for each task. If set to ``0``, the model will be trained in a fully unsupervised manner.


Other notable modules  
---------------------

- :ref:`args <module-args>`: contains all the **global** arguments. For **model-specific** arguments, see the `parse_args` function in the corresponding model file (under `models/<MODEL NAME>`).  

- :ref:`module-buffer`: contains the `Buffer` class, which is used to store the data for the replay buffer.  

- :ref:`module-training`: contains the `train` function, which is responsible for training the model, and the `evaluate` function, which is responsible for evaluating the model. The `train` function iterates over all the tasks and supports `3` utility functions: `begin_task`, `end_task`, and `observe`:

  - `begin_task`: called at the beginning of each task. It is useful if the model needs to set its internal state before     starting the task (e.g., calculating some preliminary statistics or adding new parameters for the new task).  

  - `end_task`: called at the end of each task. This function can be used to save the model after each task or perform some last-minute operations before the task ends (for example, in the case of `gdumb` it can be used to train on the data currently stored in the buffer).  

  - `observe`: called at each training step. It should contain *all the logic to train the model on the current batch*, including updating the replay buffer and the target network (if applicable). It should also return the loss value for the current batch.  

- :ref:`module-conf`: contains some utility functions such as the default path where to download the datasets (`base_path`) and the default device to use (`get_device`). 