.. _module-args:

Arguments
=========

.. rubric:: EXPERIMENT-RELATED ARGS

.. rubric:: Experiment arguments

*Arguments used to define the experiment settings.*

**\-\-lr** : float
	*Help*: Learning rate. This should either be set as default by the model (with `set_defaults <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.set_defaults>`_), by the dataset (with `set_default_from_args`, see :ref:`module-datasets.utils`), or with `--lr=<value>`.

	- *Default*: ``None``
**\-\-batch_size** : int
	*Help*: Batch size.

	- *Default*: ``None``
**\-\-label_perc** : float
	*Help*: Percentage in (0-1] of labeled examples per task.

	- *Default*: ``1``
**\-\-label_perc_by_class** : float
	*Help*: Percentage in (0-1] of labeled examples per task.

	- *Default*: ``1``
**\-\-joint** : int
	*Help*: Train model on Joint (single task)?

	- *Default*: ``0``
	- *Choices*: ``0, 1``
**\-\-eval_future** : int
	*Help*: Evaluate future tasks?

	- *Default*: ``0``
	- *Choices*: ``0, 1``

.. rubric:: Validation and fitting arguments

*Arguments used to define the validation strategy and the method used to fit the model.*

**\-\-validation** : float
	*Help*: Percentage of samples FOR EACH CLASS drawn from the training set to build the validation set.

	- *Default*: ``None``
**\-\-validation_mode** : str
	*Help*: Mode used for validation. Must be used in combination with `validation` argument. Possible values: - `current`: uses only the current task for validation (default). - `complete`: uses data from both current and past tasks for validation.

	- *Default*: ``current``
	- *Choices*: ``complete, current``
**\-\-fitting_mode** : str
	*Help*: Strategy used for fitting the model. Possible values: - `epochs`: fits the model for a fixed number of epochs (default). NOTE: this option is controlled by the `n_epochs` argument. - `iters`: fits the model for a fixed number of iterations. NOTE: this option is controlled by the `n_iters` argument. - `early_stopping`: fits the model until early stopping criteria are met. This option requires a validation set (see `validation` argument).   The early stopping criteria are: if the validation loss does not decrease for `early_stopping_patience` epochs, the training stops.

	- *Default*: ``epochs``
	- *Choices*: ``epochs, iters, time, early_stopping``
**\-\-early_stopping_patience** : int
	*Help*: Number of epochs to wait before stopping the training if the validation loss does not decrease. Used only if `fitting_mode=early_stopping`.

	- *Default*: ``5``
**\-\-early_stopping_metric** : str
	*Help*: Metric used for early stopping. Used only if `fitting_mode=early_stopping`.

	- *Default*: ``loss``
	- *Choices*: ``loss, accuracy``
**\-\-early_stopping_freq** : int
	*Help*: Frequency of validation evaluation. Used only if `fitting_mode=early_stopping`.

	- *Default*: ``1``
**\-\-early_stopping_epsilon** : float
	*Help*: Minimum improvement required to consider a new best model. Used only if `fitting_mode=early_stopping`.

	- *Default*: ``1e-06``
**\-\-n_epochs** : int
	*Help*: Number of epochs. Used only if `fitting_mode=epochs`.

	- *Default*: ``None``
**\-\-n_iters** : int
	*Help*: Number of iterations. Used only if `fitting_mode=iters`.

	- *Default*: ``None``

.. rubric:: Optimizer and learning rate scheduler arguments

*Arguments used to define the optimizer and the learning rate scheduler.*

**\-\-optimizer** : str
	*Help*: Optimizer.

	- *Default*: ``sgd``
	- *Choices*: ``sgd, adam, adamw``
**\-\-optim_wd** : float
	*Help*: optimizer weight decay.

	- *Default*: ``0.0``
**\-\-optim_mom** : float
	*Help*: optimizer momentum.

	- *Default*: ``0.0``
**\-\-optim_nesterov** : binary_to_boolean_type
	*Help*: optimizer nesterov momentum.

	- *Default*: ``0``
**\-\-lr_scheduler** : str
	*Help*: Learning rate scheduler.

	- *Default*: ``None``
**\-\-scheduler_mode** : str
	*Help*: Scheduler mode. Possible values: - `epoch`: the scheduler is called at the end of each epoch. - `iter`: the scheduler is called at the end of each iteration.

	- *Default*: ``epoch``
	- *Choices*: ``epoch, iter``
**\-\-lr_milestones** : int
	*Help*: Learning rate scheduler milestones (used if `lr_scheduler=multisteplr`).

	- *Default*: ``[]``
**\-\-sched_multistep_lr_gamma** : float
	*Help*: Learning rate scheduler gamma (used if `lr_scheduler=multisteplr`).

	- *Default*: ``0.1``

.. rubric:: Noise arguments

*Arguments used to define the noisy-label settings.*

**\-\-noise_type** : str
	*Help*: Type of noise to apply. The symmetric type is supported by all datasets, while the asymmetric must be supported explicitly by the dataset (see `datasets/utils/label_noise`).

	- *Default*: ``symmetric``
	- *Choices*: ``symmetric, asymmetric``
**\-\-noise_rate** : float
	*Help*: Noise rate in [0-1].

	- *Default*: ``0``
**\-\-disable_noisy_labels_cache** : binary_to_boolean_type
	*Help*: Disable caching the noisy label targets? NOTE: if the seed is not set, the noisy labels will be different at each run with this option disabled.

	- *Default*: ``0``
**\-\-cache_path_noisy_labels** : str
	*Help*: Path where to save the noisy labels cache. The path is relative to the `base_path`.

	- *Default*: ``noisy_labels``

.. rubric:: MANAGEMENT ARGS

.. rubric:: Management arguments

*Generic arguments to manage the experiment reproducibility, logging, debugging, etc.*

**\-\-seed** : int
	*Help*: The random seed. If not provided, a random seed will be used.

	- *Default*: ``None``
**\-\-permute_classes** : binary_to_boolean_type
	*Help*: Permute classes before splitting into tasks? This applies the seed before permuting if the `seed` argument is present.

	- *Default*: ``0``
**\-\-base_path** : str
	*Help*: The base path where to save datasets, logs, results.

	- *Default*: ``./data/``
**\-\-results_path** : str
	*Help*: The path where to save the results. NOTE: this path is relative to `base_path`.

	- *Default*: ``results/``
**\-\-device** : str
	*Help*: The device (or devices) available to use for training. More than one device can be specified by separating them with a comma. If not provided, the code will use the least used GPU available (if there are any), otherwise the CPU. MPS is supported and is automatically used if no GPU is available and MPS is supported. If more than one GPU is available, Mammoth will use the least used one if `--distributed=no`.

	- *Default*: ``None``
**\-\-notes** : str
	*Help*: Helper argument to include notes for this run. Example: distinguish between different versions of a model and allow separation of results

	- *Default*: ``None``
**\-\-eval_epochs** : int
	*Help*: Perform inference on validation every `eval_epochs` epochs. If not provided, the model is evaluated ONLY at the end of each task.

	- *Default*: ``None``
**\-\-non_verbose** : binary_to_boolean_type
	*Help*: Make progress bars non verbose

	- *Default*: ``0``
**\-\-disable_log** : binary_to_boolean_type
	*Help*: Disable logging?

	- *Default*: ``0``
**\-\-num_workers** : int
	*Help*: Number of workers for the dataloaders (default=infer from number of cpus).

	- *Default*: ``None``
**\-\-enable_other_metrics** : binary_to_boolean_type
	*Help*: Enable computing additional metrics: forward and backward transfer.

	- *Default*: ``0``
**\-\-debug_mode** : binary_to_boolean_type
	*Help*: Run only a few training steps per epoch. This also disables logging on wandb.

	- *Default*: ``0``
**\-\-inference_only** : binary_to_boolean_type
	*Help*: Perform inference only for each task (no training).

	- *Default*: ``0``
**\-\-code_optimization** : int
	*Help*: Optimization level for the code.0: no optimization.1: Use TF32, if available.2: Use BF16, if available.3: Use BF16 and `torch.compile`. BEWARE: torch.compile may break your code if you change the model after the first run! Use with caution.

	- *Default*: ``0``
	- *Choices*: ``0, 1, 2, 3``
**\-\-distributed** : str
	*Help*: Enable distributed training?

	- *Default*: ``no``
	- *Choices*: ``no, dp, ddp``
**\-\-savecheck** : str
	*Help*: Save checkpoint every `task` or at the end of the training (`last`).

	- *Default*: ``None``
	- *Choices*: ``last, task``
**\-\-loadcheck** : str
	*Help*: Path of the checkpoint to load (.pt file for the specific task)

	- *Default*: ``None``
**\-\-ckpt_name** : str
	*Help*: (optional) checkpoint save name.

	- *Default*: ``None``
**\-\-start_from** : int
	*Help*: Task to start from

	- *Default*: ``None``
**\-\-stop_after** : int
	*Help*: Task limit

	- *Default*: ``None``

.. rubric:: Wandb arguments

*Arguments to manage logging on Wandb.*

**\-\-wandb_name** : str
	*Help*: Wandb name for this run. Overrides the default name (`args.model`).

	- *Default*: ``None``
**\-\-wandb_entity** : str
	*Help*: Wandb entity

	- *Default*: ``None``
**\-\-wandb_project** : str
	*Help*: Wandb project name

	- *Default*: ``None``

.. rubric:: REEHARSAL-ONLY ARGS

**\-\-buffer_size** : int
	*Help*: The size of the memory buffer.

	- *Default*: ``None``

**\-\-minibatch_size** : int
	*Help*: The batch size of the memory buffer.

	- *Default*: ``None``

