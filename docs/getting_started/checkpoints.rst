Load and save checkpoints
=========================

Loading and saving checkpoints is handled automatically in :ref:`module-utils.training` by supplying the ``--savecheck`` and ``--loadcheck`` arguments. 

For example, to save a checkpoint after the end of the last task, simply run the following command:

.. code-block:: python 
    
        python utils/main.py --savecheck=last --model=sgd --dataset=seq-cifar10 --lr=0.1

The available options for ``--savecheck`` are:

- ``last``: save the checkpoint after **the last task**.
- ``task``: save the checkpoint after **each task**.

.. note:: 

        The ``last`` and ``task`` options have the same effect when training with ``--joint``.

This will save the checkpoint in the ``checkpoints`` folder. To load the checkpoint, simply run the following command:
.. code-block:: python 
    
        python utils/main.py --loadcheck=<path to checkpoint>.pt --model=sgd --dataset=seq-cifar10 --lr=0.1

.. rubric:: Checkpoint save format

Mammoth saves checkpoints in the ``checkpoints`` folder, with a separate checkpoint file for each task. The checkpoint file follows the format: ``[<base_checkpoint_name>]<model>_<dataset>_<buffer_size>_<n_epochs>_<timestamp>_<task>.pt``. 

- The ``<base_checkpoint_name>`` is an extra optional argument ``--ckpt_name`` that can be supplied to the training script. 

- ``<model>`` is the name of the model, supplied by ``--model``.

- ``<dataset>`` is the name of the dataset, supplied by ``--dataset``.

- ``<buffer_size>`` is the size of the buffer. If no buffer is used, this is set to 0.

- ``<n_epochs>`` is the number of epochs trained, either set by the dataset or by ``--n_epochs``.

- ``<timestamp>`` is the timestamp of when the main script was initially run. Note that this allows all the checkpoints of different tasks to be saved under the same base name (except the ``<task>``). The timestamp follows the format ``%Y%m%d-%H%M%S``. 

- ``<task>`` is the task number, starting from 0. If ``--joint`` is supplied, this is set to ``joint``.

Inside the checkpoint file, the following information is saved:

- ``model``: the state dict of the model. This contains the weights of the backbone (in ``model.backbone``) and any other parameter that was set during the model initialization and training.

- ``optimizer``: the state dict of the optimizer.

- ``scheduler``: the state dict of the scheduler, if one was used.

- ``args``: the arguments supplied to the main script.

- ``results``: all the metrics mesured up to the current task and the state of the logger. This information is necessary in order to continue training from the last checkpoint. 

.. rubric:: Additional info on checkpoint loading

Mammoth supports loading checkpoint both from the local machine **and from a remote machine** using the ``--loadcheck`` argument. To load a checkpoint from a remote machine, simply supply the ``--loadcheck`` with the URL of the checkpoint file. 

Checkpoints can be loaded either following the mammoth format (defined above) or from a simple ``.pt`` file. In the latter case, the checkpoint file should contain all the parameters of the *backbone* of the model. The other parameters (optimizer, scheduler, etc.) will be initialized from scratch.

The loading functions are available in :ref:`module-utils.checkpoints` and should take care of loading all the parameters regardless of the presence of module parallelism (see :ref:`module-fast-training`).

Pretrained backbones
--------------------

A series of pretrained backbones are available on HuggingFaces' model hub, at the following link: `https://huggingface.co/loribonna/mammoth`.

These backbones can be loaded using the ``--loadcheck`` argument, by supplying the URL of the checkpoint file.

.. note:: 

        Mammoth does not support loading the models from the HuggingFaces model hub directly. The user should download the model and load it using the ``--loadcheck`` argument.