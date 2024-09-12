Upgrading to the new Mammoth
============================

The new Mammoth is almost a complete rewrite of the old Mammoth. The new Mammoth is faster, more efficient (thanks to the ``--code_optimization``), more stable (thanks to tests), supports more validations strategies and settings, and includes more methods and datasets. 

Models
------
To upgrade your model to the new Mammoth, you need to take some care:

- The *Continual Model* already supports widely used properties such as `current_task`, `n_tasks`, `num_classes`. Check the documentation in :ref:`module-models.utils.continual_model` for more information.
- The *get_parser* has been moved **inside** the model. This is to make it easier to automatically load the arguments of a model in the case of automated parsing. This is easy to fix, just move the `get_parser` function inside the model class and make it a static method. *NOTE*: you do not need to add `add_experiment_args` and `add_management_args` to the get_parser function. These are automatically added.
- The *observe* function should follow the new signature: `def observe(inputs, labels, not_aug_inputs, epoch=None) -> dict|float`. If a `dict` is returned, it should contain at least the `loss` key. All other values will be logged in WandB (if available).

Datasets
--------
The datasets had only some minor changes. Just ensure to defined for each dataset the following properties:

- `NAME`: the name of the dataset. This will be used to load the dataset with `--dataset=<NAME>`.
- `SETTING`: the setting supported by the dataset. See :ref:`module-datasets` for more information.  
- `N_CLASSES_PER_TASK`: the number of classes per task. This can be either a single value or a list of values (one for each task).
- `N_TASKS`: the number of tasks.
- `N_CLASSES`: if missing, it will be computed from `N_CLASSES_PER_TASK` and `N_TASKS`.
- `SIZE`: the size of each input dimension (*i.e.*, height and width as a tuple for images).
- `MEAN` and `STD` for normalization.
- `TRANSFORM`: the train transform.
- `TEST_TRANSFORM`: the test transform.

Take a look at :ref:`module-datasets.seq_cifar10` for more information on how to define a dataset.

