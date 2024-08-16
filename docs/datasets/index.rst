.. _module-datasets:

Datasets
========

Mammoth datasets **define a complete and separate** Continual Learning benchmark. This means that 
each dataset **must statically define** all the necessary information to run a continual learning experiment, including:

.. admonition:: Required properties

    - Name of the dataset: **NAME** attribute (``str``).

    - Incremental setting (`class-il`, `domain-il`, or `general-continual`): **SETTING** attribute (``str``). See more in section :ref:`settings`.

    - Size of the input data: **SIZE** attribute (``tuple[int]``).

.. admonition:: Required properties for `class-il` and `domain-il` settings

    - Number of tasks: **TASKS** attribute (``int``).

    - Number of classes per task: **N_CLASSES_PER_TASK** attribute (``int|tuple[int]``). This can be a list of integers (one for each task and only for `class-il` setting), or a single integer.

.. admonition:: Required methods for **all** settings

    - **get_epochs** static method (``int``): returns the number of epoch for each task. This method is optional **only** for datasets that follow the `general-continual` setting.

    - **get_batch_size** static method (``int``): returns the batch size for each task.

    - **get_data_loaders** static method (``[DataLoader, DataLoader]``): returns the train and test data loaders for each task. See more in :ref:`Utils`.

    - **get_backbone** static method (``nn.Module``): returns the backbone model for the experiment. Backbones are defined in `backbones` folder. See more in :ref:`backbones`.

    - **get_transform** static method (``callable``): returns the data-augmentation transform to apply to the data during train.

    - **get_loss** static method (``callable``): returns the loss function to use during train.

    - **get_normalization_transform** static method (``callable``): returns the normalization transform to apply *on torch tensors* (no `ToTensor()` required).

    - **get_denormalization_transform** static method (``callable``): returns the transform to apply on the tensors to revert the normalization. You can use the `DeNormalize` function defined in `datasets/transforms/denormalization.py`.

    - **get_scheduler** static method (``callable``): returns the learning rate scheduler to use during train. *By default*, it also initializes the optimizer. This prevents errors due to the learning rate being continouosly reduced task after task. This behavior can be changed setting the argument ``reload_optim=False``.

.. admonition:: Optional methods to implement:
    - **get_prompt_templates** (``callable``): returns the prompt templates for the dataset. This method is expected for some methods (e.g., `clip`). *By default*, it returns the ImageNet prompt templates.

    - **get_class_names** (``callable``): returns the class names for the dataset. This method is not implemented by default, but is expected for some methods (e.g., `clip`). The method *should* populate the **class_names** attribute of the dataset to cache the result and call the ``fix_class_names_order`` method to ensure that the class names are in the correct order.

See :ref:`Continual Dataset <module-datasets.utils.continual_dataset>` for more details or **SequentialCIFAR10** in :ref:`Seq CIFAR-10 <module-datasets.seq_cifar10>` for an example.

.. note::
    Datasets are downloaded by default in the **data** folder. You can change this
    default location by setting the **base_path** function in :ref:`conf <module-utils.conf>`. 

.. _settings:
Experimental settings
---------------------

Experimental settings follow and extend the notation of `Three Scenarios for Continual Learning <https://arxiv.org/abs/1904.07734>`_, 
and are defined in the **SETTING** attribute of each dataset. The following settings are available:

- `class-il`: the total number of classes increases at each task, following the **N_CLASSES_PER_TASK** attribute.
    .. admonition:: On *task-il* and *class-il*
        :class: note

        Using this setting metrics will be computed both for `class-il` and `task-il`. Metrics for 
        `task-il` will be computed by masking the correct task for each sample during inference. This 
        allows to compute metrics for both settings without having to run the experiment twice.

- `domain-il`: the total number of classes is fixed, but the distribution of the input data changes at each task.

- `general-continual`: the distribution of the classes change gradually over time, without notion of task boundaries. In this setting, the **TASKS** and **N_CLASSES_PER_TASK** attributes are ignored as there is only a single long tasks that changes over time.

- `cssl`: this setting is the same as `class-il`, but with some of the labels missing due to limited supervision. This setting is used to simulate the case where a percentage of the labels is not available for training. For example, if ``--label_perc_by_task`` or ``--label_perc_by_class`` is set to ``0.5``, only 50% of the labels will be available for training. The remaining 50% will be masked with a label of ``-1`` and ignored during training if the currently used method does not support partial labels (check out the **COMPATIBILITY** attribute in :ref:`module-models`).

.. admonition:: Experiments on the **joint** setting
    :class: hint

    Mammoth datasets support the **joint** setting, which is a special case of the `class-il` setting where all the classes are available at each task. This is useful to compare the performance of a method on what is usually considered the *upper bound* for the `class-il` setting. To run an experiment on the **joint** setting, simply set the ``--joint`` to ``1``. This will automatically set the **N_CLASSES_PER_TASK** attribute to the total number of classes in the dataset and the **TASKS** attribute to ``1``.

    Note that the **joint** setting is available only for the `class-il` (and `task-il`) setting. If you want to run an experiment on the **joint** setting for a dataset that follows the `domain-il` setting, you can use the :ref:`Joint <module-models.joint>` **model** (with ``--model=joint``).

Evaluate on Future Tasks
~~~~~~~~~~~~~~~~~~~~~~~~

By default, the evaluation is done up to the current task. However, some models also support evaluation on future tasks (e.g., :ref:`CGIL <module-models.cgil>`). In this case, you can set the ``--eval_future`` to ``1`` to evaluate the model on future tasks. 

.. important::

    In order to be able to evaluate on future tasks, the method must extend the :ref:`FutureModel <module-models.utils.future_model>` class. Notably, this function includes the ``future_forward`` method, which performs inference on all classes, and the ``change_transform`` method, which allows to change the transform to be applied to the data during inference.

Default arguments and command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides **get_epochs** and **get_batch_size**, datasets can define default arguments that are used to set the default values for the command line arguments.
This is done with the **set_default_from_args** decorator, which takes the name of the command line argument as input. For example, the following code sets the default value for the `--label_perc_by_task` argument:

.. code-block:: python

    @set_default_from_args('--label_perc_by_task')
    def get_label_perc(self):
        return 0.5


Steps to create a new dataset
-----------------------------
    
All datasets must inherit from the **ContinualDataset** class, which is defined in :ref:`Continual Dataset <module-datasets.utils.continual_dataset>`. The only
exception are datasets that follow the `general-continual` setting, which inherit from the **GCLDataset** class, (defined in :ref:`GCL Dataset <module-datasets.utils.gcl_dataset>`).
These classes provide some useful methods to create data loaders and store masked data loaders for continual learning experiments. See more in the next section.

    1. Create a new file in the `datasets` folder, e.g. ``my_dataset.py``.

    2. Define a new class that inherits from `ContinualDataset` or `GCLDataset` and implements all the required methods and attributes.

    3. Define the **get_data_loaders** method, which returns a list of train and test data loaders for each task (see more in section :ref:`Utils <dataset-index-utils>`). 

    .. tip::
        For convenience, most datasets are initially created with all classes and then masked appropriately by the **store_masked_loaders** function. 
        For example, in :ref:`Seq CIFAR-10 <module-datasets.seq_cifar10>` the **get_data_loaders** function of **SequentialCIFAR10** dataset first inizializes the **MyCIFAR10** and **TCIFAR10** 
        datasets with train and test data for all classes respectively, and then masks the data loaders to return only the data for the current task.

    .. important::
        The train data loader **must** return both augmented and non-augmented data. This is done to allow the storage of raw data for replay-based methods 
        (for more information, check out `Rethinking Experience Replay: a Bag of Tricks for Continual Learning <https://arxiv.org/abs/2010.05595>`_).
        The signature return for the train data loader is ``(augmented_data, labels, non_augmented_data)``, while the test data loader should return ``(data, labels)``.

    4. If all goes well, your dataset should be picked up by the **get_dataset** function and you should be able to run an experiment with it.

.. _dataset-index-utils:
Utils
--------

- **get_data_loaders**: This function should take care of downloading the dataset if necessary, make sure that it contains samples and labels for 
**only** the current task (you can use the **store_masked_loaders** function), and create the data loaders.

- **store_masked_loaders**: This function is defined in :ref:`Continual Dataset <module-datasets.utils.continual_dataset>` and takes care of masking the data loaders to return only the data for the current task.
It is used by most datasets to create the data loaders for each task.

    - If the ``--permute_classes`` flag is set to ``1``, it also applies the appropriate permutation to the classes before splitting the data.

    - If the ``--label_perc_by_task/--label_perc_by_class`` argument is set to a value between ``0`` and ``1``, it also randomly masks a percentage of the labels for each task/class. 

