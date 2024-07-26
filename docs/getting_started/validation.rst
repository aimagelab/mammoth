.. _module-validation:

Training, Validation, and Testing
=================================

During each task, Mammoth trains on the current data until some stopping criterion is met. 
Currently, Mammoth supports 3 types of stopping criteria, which can be chosen using the ``--fitting_mode`` command line argument. The three types are ``epochs``, ``iters``, and ``early_stopping``. The default is ``epochs``.

.. rubric:: Criterion by epochs (``--fitting_mode=epochs``)

This is the default option, for which training stops after a fixed number of **epochs**. The number of epochs can be set using the ``--n_epochs`` command line argument. Note that most datasets indicate the default number of epochs via the `set_default_from_args` decorator (see :ref:`module-datasets` for more information).

.. rubric:: Criterion by iterations (``--fitting_mode=iters``)

This option stops training after a fixed number of **iterations**. The number of iterations can be set using the ``--n_iters`` command line argument. In addition, a default value for each dataset can be set using the `set_default_from_args` decorator. For example, to set the default number of iterations to 1000 for a particular dataset you can use the following code, adding it to the dataset class definition:

.. code-block:: python

    @set_default_from_args('n_iters')
    def get_iters(self):
        return 1000

.. rubric:: Early stopping (``--fitting_mode=early_stopping``)

This option is the most flexible, as it allows training to continue until a certain stopping criterion is met. The criterion can be based either on the loss function or on the accuracy by setting the ``--early_stopping_metric`` command line argument (default is by ``loss``). The number of epochs to wait before stopping can be set using the ``--early_stopping_patience`` command line argument. The default value is 5.

.. note::

    The early stopping criterion is based on the chosen validation set (see next section).

In addition, the early stopping criterion supports the following options:

* ``early_stopping_freq`` (default is 1): the frequency at which the early stopping criterion is checked. For example, if ``early_stopping_freq`` is set to 5, the criterion is checked every 5 epochs.
* ``early_stopping_epsilon`` (default is 1e-6): the minimum improvement in the validation loss or accuracy that is considered significant. If the improvement is less than ``early_stopping_epsilon``, the training stops.

Validation
----------

During training, Mammoth uses a validation set to monitor the performance of the model. By default, the validation set is **disabled**, meaning that performance is monitored on the **test** set. This is done in line with most CL literature, which uses the test set for validation as it is not trivial to define a validation set for CL tasks. In particular, two options may be possible:

1. *The validation set includes data* **only of the current task**: this is the most straightforward option, but has the disantvantage of producing a higher degree of forgetting on past tasks as the model's objective ignores past data.
2. *The validation set includes data* **of all seen tasks**: this option should produce a more balanced result, but conflicts with the CL setting, as the model should not have access to data from past tasks. However, since most CL works only focus on maximizing the accuracy on all tasks after having seen all of them, this option is the most common in the literature.

In Mammoth, the use of a validation set can be enabled by specifiying the percentage of the training set that is used as the validation set using the ``--validation`` command line argument. For example, to use 10% of the training set as the validation set, you can use the following command:

.. code-block:: bash

    python main.py --validation 10 <other arguments>

As for the choice of strategy to build the validation set, Mammoth supports both options described above using the ``--validation_mode`` command line argument. The default is ``current``, meaning that the validation set includes only data of the current task. If you want to use a validation set that includes data of all seen tasks, you can set ``--validation_mode`` to ``complete``.
