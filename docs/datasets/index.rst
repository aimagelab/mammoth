Datasets
========

Mammoth datasets **define a complete and separate** Continual Learning benchmark. This means that 
each dataset **must statically define** all the necessary information to run a continual learning experiment, including:

.. admonition:: Required properties

    - Name of the dataset: **NAME** attribute.

    - Incremental setting (`class-il`, `domain-il`, or `general-continual`): **SETTING** attribute. See more in section :ref:`settings`.

    - Size of the input data: **SIZE** attribute.

.. admonition:: Required properties for `class-il` and `domain-il` settings

    - Number of tasks: **TASKS** attribute.

    - Number of classes per task: **N_CLASSES_PER_TASK** attribute. This can be a list of integers (one for each task and only for `class-il` setting), or a single integer.

.. admonition:: Required methods for **all** settings

    - **get_epochs** static method: returns the number of epoch for each task. This method is optional **only** for datasets that follow the `general-continual` setting.

    - **get_batch_size** static method: returns the batch size for each task.

    - **get_data_loaders** static method: returns the data loaders for each task. See more in :ref:`Utils`.

    - **get_backbone** static method: returns the backbone model for the experiment. Backbones are defined in `backbones` folder. See more in :ref:`backbones`.

    - **get_transform** static method: returns the data-augmentation transform to apply to the data during train.

    - **get_loss** static method: returns the loss function to use during train.

    - **get_normalization_transform** static method: returns the normalization transform to apply *on torch tensors* (no `ToTensor()` required).

    - **get_denormalization_transform** static method: returns the transform to apply on the tensors to revert the normalization. You can use the `DeNormalize` function defined in `datasets/transforms/denormalization.py`.


See `datasets/utils/continual_dataset.py` for more details or `SequentialCIFAR10` in `datasets/seq_cifar10.py` for an example.


All datasets must inherit from the `ContinualDataset` class, which is defined in `datasets/utils/continual_dataset.py`. The only
exception are datasets that follow the `general-continual` setting, which inherit from the `GCLDataset` class, (defined in `datasets/utils/gcl_dataset.py`).
These classes provide some useful methods to create data loaders and store masked data loaders for continual learning experiments. See more in section :ref:`utils`.

.. note::
    Datasets are downloaded by default in the **data** folder. You can change this
    default location by setting the `base_path` function in `utils/conf.py`.

Settings
--------

There are three possible settings for a continual learning experiment:

- `class-il`: the total number of classes increases at each task, following the `N_CLASSES_PER_TASK` attribute.
    .. admonition:: On task-il and class-il
        :class: note

        Using this setting metrics will be computed both for `class-il` and `task-il`. Metrics for 
        `task-il` will be computed by masking the correct task for each sample during inference. This 
        allows to compute metrics for both settings without having to run the experiment twice.

- `domain-il`: the total number of classes is fixed, but the domain of the data changes at each task.

- `general-continual`: the distribution of the classes change gradually over time, without notion of task boundaries. In this 
setting, the `TASKS` and `N_CLASSES_PER_TASK` attributes are ignored as there is only a single long tasks that changes over time.

Steps to create a new dataset:
    
    1. ...

Required return values of train and test datasets:
    
    1. `__getitem__` method that returns a tuple of (image, label)

    2. `__len__` method that returns the length of the dataset

Utils
--------

Useful functions:

    - `get_data_loaders`

    - `store_masked_loaders`

 