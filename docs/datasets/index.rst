.. _module-datasets:

Datasets
========

Mammoth datasets **define a complete** Continual Learning benchmark. This means that each dataset **defines** all the necessary information to run a continual learning experiment.

For more information on how to create a new dataset, see :ref:`Steps to create a new dataset <build_a_dataset>` or **SequentialCIFAR10** in :ref:`Seq CIFAR-10 <module-datasets.seq_cifar10>` for a practical example.

.. note::

    Datasets are downloaded by default in the **data** folder. You can change this default location by using the ``--base_path`` argument. 

.. _datasets-settings:

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

- `biased-class-il`: similar to `class-il`, but with a *biased* distribution of the classes. This setting is derived from `Learning without Shortcuts <https://iris.unimore.it/retrieve/5890cf10-47bc-4891-8138-4b4999d2eccc/2025wacv_bias.pdf>`_. Datasets that support this setting must have a `bias_label` attribute, to be used during evaluation.

.. admonition:: Experiments on the **joint** setting
    :class: hint

    Mammoth datasets support the **joint** setting, which is a special case of the `class-il` setting where all the classes are available at each task. This is useful to compare the performance of a method on what is usually considered the *upper bound* for the `class-il` setting. To run an experiment on the **joint** setting, simply set the ``--joint`` to ``1``. This will automatically set the **N_CLASSES_PER_TASK** attribute to the total number of classes in the dataset and the **TASKS** attribute to ``1``.

    Note that the **joint** setting is available only for the `class-il` (and `task-il`) setting. If you want to run an experiment on the **joint** setting for a dataset that follows the `domain-il` setting, you can use the :ref:`Joint <module-models.joint>` **model** (with ``--model=joint``).

Evaluate on Future Tasks
~~~~~~~~~~~~~~~~~~~~~~~~

By default, the evaluation is done up to the current task. However, some models also support evaluation on future tasks (e.g., :ref:`CGIL <module-models.cgil>`). In this case, you can set the ``--eval_future`` to ``1`` to evaluate the model on future tasks. 

.. important::

    In order to be able to evaluate on future tasks, the method must extend the :ref:`FutureModel <module-models.utils.future_model>` class. Notably, this function includes the ``future_forward`` method, which performs inference on all classes, and the ``change_transform`` method, which allows to change the transform to be applied to the data during inference.

.. _dataset-index-defaults:

Default arguments and command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides **get_epochs** and **get_batch_size**, datasets can define default arguments that are used to set the default values for the command line arguments.
This is done with the **set_default_from_args** decorator, which takes the name of the command line argument as input. For example, the following code sets the default value for the `--label_perc_by_task` argument:

.. code-block:: python

    @set_default_from_args('--label_perc_by_task')
    def get_label_perc(self):
        return 0.5


.. _dataset-configurations:

Dataset configurations
----------------------

To allow for a more flexible configuration of the datasets, Mammoth supports the use of configuration files that can be used to set the values of the dataset attributes. This greatly simplifies the creation of new datasets, as it allows to separate the definition of a dataset (i.e., its data) from its configuration (number of tasks, transforms, etc.).

The configuration files are stored in **datasets/configs/<dataset name>/<configuration name>.yaml** and can be selected from the command line using the ``--dataset_config`` argument. 

The configuration file may contain:
- `SETTING`: the incremental setting of the dataset. This can be one of 'class-il', 'domain-il', 'general-continual', 'cssl', or 'biased-class-il'.
- `N_CLASSES_PER_TASK`: the number of classes per task. This can be a single integer or a list of integers (one for each task).
- `N_TASKS`: the number of tasks.
- `SIZE`: the size of the input data.
- `N_CLASSES`: the total number of classes in the dataset.
- `AVAIL_SCHEDS`: the available learning rate schedulers for the dataset.
- `TRANSFORM`: the data augmentation transform to apply to the data during training.
- `TEST_TRANSFORM`: the normalization transform to apply to the data during training.
- `MEAN`, `STD`: the mean and standard deviation of the dataset, used for normalization.
- any field specified by the `set_default_from_args` decorator in the dataset class (see more in section :ref:`dataset-index-defaults`). This includes the `backbone`, `batch_size`, `n_epochs`, etc.
- `args`: special field that allows to set the values of the default values for the command line arguments

The configuration file sets the default values for the dataset attributes and all values defined by the `set_default_from_args` decorator. The priority is as follows: command line arguments > default values set by the model > configuration file.

.. toctree:: 
    :hidden:

    How to create a new dataset <build_a_dataset.rst>
    Custom schedulers <schedulers.rst>