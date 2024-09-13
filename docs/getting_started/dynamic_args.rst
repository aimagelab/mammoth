.. _module-dynamic-registration:

Registration of backbones and datasets
--------------------------------------

The registration of backbones and datasets is a new feature that allows to make them easily accessible with the new ``--backbone`` and ``--dataset`` arguments. This is done with the `register_backbone` and `register_dataset` decorators, respectively.

These decorators can be applied either to subclasses of the backbone or dataset classes, or to functions that return the backbone or dataset. The decorator takes a single argument, the name of the backbone or dataset, which will be used to select the backbone or dataset from the command line (with ``--backbone`` or ``--dataset``).

For example, the following code registers a dataset called ``MyDataset``:

.. code-block:: python

    from mammoth.utils import register_dataset

    @register_dataset("my_dataset")
    class MyDataset:
        def __init__(self, arg1, arg2="default"):
            self.arg1 = arg1
            self.arg2 = arg2

After the registration, the dataset can be accessed with the following command:

.. code-block:: bash

    python my_script.py --dataset my_dataset --arg1 value1 --arg2 value2

The arguments ``arg1`` and ``arg2`` are automatically built and added to the parser (see below).

Under the hood, the decorator registers the backbone or dataset in a global dictionary with the `register_dynamic_module_fn` function. In principle, this function allows to register any kind of module, but it is currently used only for backbones and datasets.

.. _dynamic-arguments:

Dynamic arguments
~~~~~~~~~~~~~~~~~

Each backbone or dataset may require different arguments. However, having to specify all of them would be cumbersome and would result in a big list of unreadable parameters added to the main parser. To avoid this, we now support *dynamically* creating and adding the arguments *only for the chosen* backbone or dataset.

The arguments are loaded during the registration of the backbone or dataset with the `register_backbone` or `register_dataset` decorator. The decorator will automatically try to infer the arguments from the *signature* of the function (or the `__init__` method if applied to a class). For each argument in the signature, its default value will be used as the default value during parsing. If the default is not set, the argument is *required*; otherwise, the argument is optional. The type of the argument is inferred from the default value (default is `str`) or from the type hint.

Since the arguments are dynamically added, they will be available only if the corresponding backbone or dataset is selected. This means that the pipeline to build the arguments is quite complex, as it needs also to account (in increasing order of priority) for:

- The default values set into the dataset or backbone class.
- The dynamic arguments set by the dynamic registration. Those for the dataset may be added first, but those for the backbone must be added last, as the backbone may depend on the dataset or the model configuration. Nonetheless, the default values set by the dynamic registration are overridden by the next steps.
- The dataset configuration file, which may contain additional arguments for the dataset or alter some characteristics of the dataset itself, such as the number of tasks or the transforms to apply.
- The default values set using the **set_defaults** method of ArgumentParser.
- The model configuration file, which may contain different default values for the arguments of the backbone.
