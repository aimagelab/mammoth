.. _module-models:

Models
========

A **model** is defined as a Python class that defines a few methods and attributes to be used in the continual learning framework.
To be compatible with the *auto-detection* mechanism (the **get_model** function below), a model must:

* extend the base class :ref:`ContinualModel <module-models.utils.continual_model>`, which implements most of the required methods, leaving to the user the definition of the **observe** method (see in :ref:`training and testing`). In addition, the model must define the **NAME** and **COMPATIBILITY** attributes (see below).

* be defined in a file named **<model_name>.py** and placed in the **models** folder. 

Check out the guide on :ref:`how to build a model <build_a_model>` for more details.

The model-specific hyper-parameters of the model can be set in the **get_parser** static method (see in :ref:`Model parameters <model_arguments_docs>`). 

.. note::
    The name of the file will be used to identify the model. For example, if the model is defined in a file named **my_model.py**, the name of the model will be **my_model** and will be called with the command line option **--model my_model**.

.. important::
    Each file can contain **only one** model. If you want to define multiple models, you have to create multiple files.

Training and testing
--------------------

For additional details, see :ref:`how to build a model <build_a_model>`.

The **observe** method defines the actual behavior of the model and is the only method that **must** be implemented by the user. It is called at each training iteration and it is used to update the model parameters according to the current training batch. 

Evaluation
~~~~~~~~~~

The **forward** method is used to evaluate the model on the test set. By default, it is implemented in the base class **ContinualModel** and just calls the **forward** method of the backbone model.

Attributes and utility methods
-------------------------------

The base class (:ref:`ContinualModel <module-models.utils.continual_model>`) includes the **NAME** and **COMPATIBILITY** attributes, which are used to identify the model and to check its compatibility with the chosen **setting** (see :ref:`module-datasets` for more details). The **NAME** attribute is a string that identifies the model, while the **COMPATIBILITY** attribute is a list of strings that identify the compatible settings. For example, :ref:`module-models.der` includes compatibility with ``['class-il', 'domain-il', 'task-il', 'general-continual']`` settings, and thus is compatible with all the datasets included in the framework. However, as it includes no compatibility with the ``'cssl'`` setting, it cannot take advantage of unlabeled samples (available if ``--label_perc_by_task`` or ``--label_perc_by_class`` is set to a value between ``0`` and ``1``).

For more details, see :ref:`Common attributes and methods <model_attributes>`.

Basic model class
~~~~~~~~~~~~~~~~~

The **ContinualModel** loads the backbone model (i.e., the model used to compute the output of the model, see :ref:`module-backbones`) during the initialization. By default, the backbone model is defined by the chosen **dataset** (see :ref:`module-datasets` for more details), but it can also be set with the ``--backbone`` CLI argument. Once loaded, the backbone model can be accessed through the **net** attribute.

Handling Begin and End of tasks and epochs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the **observe** and **forward** methods, the **ContinualModel** provides the **begin_task** and **end_task** methods, which are called at the beginning and at the end of each task respectively, and the **begin_epoch** and **end_epoch** methods, which are called at the beginning and at the end of each epoch respectively. These methods can be overridden to implement custom behavior. For example, the **end_task** method can be used to save the model parameters at the end of each task.

Model arguments
---------------

The **get_parser** method is used to define the model-specific hyper-parameters. It is defined as a static method (see :ref:`ContinualModel <module-models.utils.continual_model>`) that takes an existing `argparse.ArgumentParser <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_ object and returns an updated version of it with the model-specific hyper-parameters added. This method is called during the initialization of the model and it is used to parse the command line arguments. The **get_parser** method must have the following signature:

.. code-block:: python

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

        # Create the parser
        parser = argparse.ArgumentParser('MyModel parameters')

        # Add the model-specific hyper-parameters
        parser.add_argument('--my_param', type=int, default=1, help='My parameter')
        ...

        return parser

For more details, see :ref:`Defining model parameters <model_arguments_docs>`.

.. note::

    To remain backward compatible with the previous version of the framework, the `parser` parameter is *optional*. In this case, the method must create a new `argparse.ArgumentParser <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_ object and return it.

Once the model is selected with the command line option **--model**, the hyper-parameters are loaded and can be viewed with ``--help``.


Other utility methods
~~~~~~~~~~~~~~~~~~~~~

* **get_optimizer**: returns the optimizer used to train the model.

* **get_debug_iters**: used if ``--debug_mode`` is set to ``1``, it returns the number of iterations to perform during each task. By default, it returns ``5``.

* **autolog_wandb**: called after each observe, it relies on the :ref:`Magic <module-utils.magic>` module to log all the variables created in the **observe** that start with *loss* or *_wandb_*. This method can also be called manually to log custom variables by providing the ``extra`` parameter. 
    .. note::
        This method is called only if ``--debug_mode`` is set to ``0`` (i.e, it is not called during the debug mode). 

Advanced usage
---------------

The **ContinualModel** class relies on a few hooks to automatically update its internal attributes. These hooks are called before the **begin_task**, **end_task**, and **observe** methods (**meta_begin_task**, **meta_end_task**, and **meta_observe** respectively). If you want to implement a custom behavior, you can override these hooks. 

.. note::
    The **meta_observe** is responsible for removing the *unlabeled* samples (i.e., those with an associated label set to ``-1``) from the batch if the model does not support the ``--label_perc_by_class`` and ``--label_perc_by_task`` parameters.

.. toctree:: 
    :hidden:

    How to build a model <build_a_model.rst>
    Defining model parameters <model_arguments.rst>
    Common attributes and methods <model_attributes.rst>