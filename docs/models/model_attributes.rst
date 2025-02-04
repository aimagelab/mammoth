.. _model_attributes:

Utility attributes and methods
==============================

The `ContinualModel` class provides a set of utility attributes and methods that can be used to simplify the implementation of the model. The following list describes the most important attributes and methods:

    1. **net**: This attribute stores the model architecture. It is initialized in the constructor based on the provided *backbone*. The backbone is a string that defines the architecture of the model (e.g., `resnet18`, `vit`, etc.) and is defined by default by the chosen dataset. However, it can be overridden by the user by using the ``--backbone`` argument. For more information, see `backbone <module-backbones>`.

    2. **loss**: This attribute stores the loss function used to train the model. It is defined by the chosen dataset (and is usually the cross-entropy loss).

    3. **opt**: This attribute stores the optimizer used to update the model parameters. The default optimizer is the stochastic gradient descent (SGD) optimizer, but it can be overridden by the user by using the ``--optimizer`` argument. By default, *all the parameters* of the model are optimized, but this behavior can be changed by changing the ``get_parameters`` method.

    4. **scheduler**: This attribute stores the learning rate scheduler used to adjust the learning rate during training. By default, no scheduler is applied, but it can be specified using the ``--scheduler`` argument. Only the `MultiStepLR` is supported by default (with ``--scheduler=multisteplr``). More info are available in `schedulers <dataset-schedulers-docs>`. 

    5. **args**: This attribute stores the arguments passed to the model.

    6. **transform**: This attribute stores the data augmentation pipeline used during training, defined by the chosen dataset.

    7. **device**: This attribute stores the device used to run the model (i.e., `cuda` or `cpu`).

Automatic attributes
--------------------

The base class **ContinualModel** provides a few properties that are automatically set during the incremental training (see :ref:`ContinualModel <module-models.utils.continual_model>` for more details).

Task-related attributes
~~~~~~~~~~~~~~~~~~~~~~~

- **current_task**: the index of the current task (starting from 0). This attribute is automatically updated at the end of each task (*after* the **end_task**).

- **n_classes_current_task**: the number of classes in the current task.

- **n_past_classes**: the total number of classes seen so far (past).

- **n_seen_classes**: the total number of classes seen so far (past and current).

- **n_classes**: the total number of classes in the dataset (past, current, and remaining).

- **n_tasks**: the total number of tasks.

- **task_iteration**: the number of iterations performed during the current task. This attribute is automatically updated *after* each **observe** call and is reset at the beginning of each task (*before* the **begin_task**). Can be used to implement a virtual batch size (see :ref:`module-models.twf`).

- **classes_per_task** (alias **cpt**): the *raw* amount of classes for each task. This could be either an integer (i.e., the number of classes for each task is the same) or a list of integers (i.e., the number of classes for each task is different).

Transforms and dataset-related Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **transform**: the transform applied to the input data. This attribute is automatically set during the initialization of the model and is defined by the chosen **dataset** (see :ref:`module-datasets` for more details). In most cases, this is implemented as a `kornia <https://github.com/kornia/kornia>`_ transform (translated from PIL thanks to `to_kornia_transform` in :ref:`Kornia Utils <module-utils.kornia_utils>`). However, if a transform is not supported by the **to_kornia_transform**, it is implemented as `PIL <https://pillow.readthedocs.io/en/stable/>`_.

- **original_transform**: the original transform defined by the chosen **dataset**. This is implemented as a `PIL <https://pillow.readthedocs.io/en/stable/>`_ transform (and not translated into `kornia` as the **transform**).

- **normalization_transform**: the transform used to normalize the input data. As for the **weak_transform**, this is implemented as a `kornia <https://github.com/kornia/kornia>`_ transform if possible, otherwise it is implemented as `PIL <https://pillow.readthedocs.io/en/stable/>`_.

.. note::
    The automatic conversion between `PIL <https://pillow.readthedocs.io/en/stable/>`_ and `kornia <https://github.com/kornia/kornia>`_ is handeled by the **to_kornia_transform** function in :ref:`Kornia Utils <module-utils.kornia_utils>`, which converts (*most*) PIL transforms to kornia transforms. However, not all the transforms are supported, and thus this function *may not be always available*. If you want to use a custom transform, you have to extend the **to_kornia_transform** function.

Important methods
-----------------

The `ContinualModel` class defines a few important methods that can be overridden to customize the behavior of the model:

1. **begin_task**, **end_task**: These methods are called at the beginning and end of each task, respectively. They can be overridden to perform additional operations at the beginning and end of each task. They take a single argument, `dataset`, which is the current task dataset.

2. **begin_epoch**, **end_epoch**: Same as above, but for each epoch. They also take the input `epoch` as an argument.

3. **get_parameters**: This method returns the parameters that should be optimized by the optimizer. By default, it returns all the parameters of the model, but it can be overridden to optimize only a subset of the parameters.

4. **get_optimizer**: This method returns the optimizer used to update the model parameters. It calls the `get_parameters` method to get the parameters to optimize.

Mystical methods
~~~~~~~~~~~~~~~~

The `ContinualModel` class also defines the ``autolog_wandb`` method, which is used to automatically log the model parameters and metrics to `wandb` (if enabled). This method looks at all the variables defined in the `observe` method and logs them to `wandb` if they start with `_wandb_` or `loss`. This method is called automatically at the end of the `observe` method.

The idea is to simplify the logging process by automatically logging all the variables that you might be interested in. However, you can also manually log additional variables by calling `wandb.log` or return them in a dictionary (along with the loss) at the end of the `observe`.