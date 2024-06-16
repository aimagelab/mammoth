.. _module-args:

Arguments
=========

.. rubric:: EXPERIMENT-RELATED ARGS

.. rubric:: Experiment arguments

*Arguments used to define the experiment settings.*

**\-\-dataset** : <class 'str'>
            *Help*: Which dataset to perform experiments on.

            - Default: None

            - Choices: mnist-360, perm-mnist, rot-mnist, seq-cifar10, seq-cifar100, seq-cifar100-224, seq-cifar100-224-rs, seq-cifar10-224, seq-cifar10-224-rs, seq-cub200, seq-imagenet-r, seq-mnist, seq-tinyimg, seq-tinyimg-r
**\-\-model** : <function custom_str_underscore at 0x7fbde63d4dc0>
            *Help*: Model name.

            - Default: None

            - Choices: agem, agem-r, bic, ccic, coda-prompt, der, derpp, derpp-lider, dualprompt, er, er-ace, er-ace-lider, ewc-on, fdr, gdumb, gdumb-lider, gem, gss, hal, icarl, icarl-lider, joint-gcl, l2p, lucir, lwf, lwf-mc, mer, pnn, rpc, sgd, si, slca, twf, xder, xder-ce, xder-rpc
**\-\-lr** : <class 'float'>
            *Help*: Learning rate.

            - Default: None

            - Choices: 
**\-\-batch_size** : <class 'int'>
            *Help*: Batch size.

            - Default: None

            - Choices: 
**\-\-label_perc** : <class 'float'>
            *Help*: Percentage in (0-1] of labeled examples per task.

            - Default: 1

            - Choices: 
**\-\-joint** : <class 'int'>
            *Help*: Train model on Joint (single task)?

            - Default: 0

            - Choices: 0, 1

.. rubric:: Validation and fitting arguments

*Arguments used to define the validation strategy and the method used to fit the model.*

**\-\-validation** : <class 'float'>
            *Help*: Percentage of samples FOR EACH CLASS drawn from the training set to build the validation set.

            - Default: None

            - Choices: 
**\-\-validation_mode** : <class 'str'>
            *Help*: Mode used for validation. Must be used in combination with `validation` argument. Possible values: - `current`: uses only the current task for validation (default). - `complete`: uses data from both current and past tasks for validation.

            - Default: current

            - Choices: complete, current
**\-\-fitting_mode** : <class 'str'>
            *Help*: Strategy used for fitting the model. Possible values: - `epochs`: fits the model for a fixed number of epochs (default). NOTE: this option is controlled by the `n_epochs` argument. - `iters`: fits the model for a fixed number of iterations. NOTE: this option is controlled by the `n_iters` argument. - `early_stopping`: fits the model until early stopping criteria are met. This option requires a validation set (see `validation` argument).   The early stopping criteria are: if the validation loss does not decrease for `early_stopping_patience` epochs, the training stops.

            - Default: epochs

            - Choices: epochs, iters, time, early_stopping
**\-\-early_stopping_patience** : <class 'int'>
            *Help*: Number of epochs to wait before stopping the training if the validation loss does not decrease. Used only if `fitting_mode=early_stopping`.

            - Default: 5

            - Choices: 
**\-\-early_stopping_metric** : <class 'str'>
            *Help*: Metric used for early stopping. Used only if `fitting_mode=early_stopping`.

            - Default: loss

            - Choices: loss, accuracy
**\-\-early_stopping_freq** : <class 'int'>
            *Help*: Frequency of validation evaluation. Used only if `fitting_mode=early_stopping`.

            - Default: 1

            - Choices: 
**\-\-early_stopping_epsilon** : <class 'float'>
            *Help*: Minimum improvement required to consider a new best model. Used only if `fitting_mode=early_stopping`.

            - Default: 1e-06

            - Choices: 
**\-\-n_epochs** : <class 'int'>
            *Help*: Number of epochs. Used only if `fitting_mode=epochs`.

            - Default: None

            - Choices: 
**\-\-n_iters** : <class 'int'>
            *Help*: Number of iterations. Used only if `fitting_mode=iters`.

            - Default: None

            - Choices: 

.. rubric:: Optimizer and learning rate scheduler arguments

*Arguments used to define the optimizer and the learning rate scheduler.*

**\-\-optimizer** : <class 'str'>
            *Help*: Optimizer.

            - Default: sgd

            - Choices: sgd, adam, adamw
**\-\-optim_wd** : <class 'float'>
            *Help*: optimizer weight decay.

            - Default: 0.0

            - Choices: 
**\-\-optim_mom** : <class 'float'>
            *Help*: optimizer momentum.

            - Default: 0.0

            - Choices: 
**\-\-optim_nesterov** : <class 'int'>
            *Help*: optimizer nesterov momentum.

            - Default: 0

            - Choices: 
**\-\-lr_scheduler** : <class 'str'>
            *Help*: Learning rate scheduler.

            - Default: None

            - Choices: 
**\-\-lr_milestones** : <class 'int'>
            *Help*: Learning rate scheduler milestones (used if `lr_scheduler=multisteplr`).

            - Default: []

            - Choices: 
**\-\-sched_multistep_lr_gamma** : <class 'float'>
            *Help*: Learning rate scheduler gamma (used if `lr_scheduler=multisteplr`).

            - Default: 0.1

            - Choices: 

.. rubric:: MANAGEMENT ARGS

.. rubric:: Management arguments

*Generic arguments to manage the experiment reproducibility, logging, debugging, etc.*

**\-\-seed** : <class 'int'>
            *Help*: The random seed. If not provided, a random seed will be used.

            - Default: None

            - Choices: 
**\-\-permute_classes** : <class 'int'>
            *Help*: Permute classes before splitting into tasks? This applies the seed before permuting if the `seed` argument is present.

            - Default: 0

            - Choices: 0, 1
**\-\-base_path** : <class 'str'>
            *Help*: The base path where to save datasets, logs, results.

            - Default: ./data/

            - Choices: 
**\-\-notes** : <class 'str'>
            *Help*: Helper argument to include notes for this run. Example: distinguish between different versions of a model and allow separation of results

            - Default: None

            - Choices: 
**\-\-eval_epochs** : <class 'int'>
            *Help*: Perform inference on validation every `eval_epochs` epochs. If not provided, the model is evaluated ONLY at the end of each task.

            - Default: None

            - Choices: 
**\-\-non_verbose** : <class 'int'>
            *Help*: Make progress bars non verbose

            - Default: 0

            - Choices: 0, 1
**\-\-disable_log** : <class 'int'>
            *Help*: Disable logging?

            - Default: 0

            - Choices: 0, 1
**\-\-num_workers** : <class 'int'>
            *Help*: Number of workers for the dataloaders (default=infer from number of cpus).

            - Default: None

            - Choices: 
**\-\-enable_other_metrics** : <class 'int'>
            *Help*: Enable computing additional metrics: forward and backward transfer.

            - Default: 0

            - Choices: 0, 1
**\-\-debug_mode** : <class 'int'>
            *Help*: Run only a few training steps per epoch. This also disables logging on wandb.

            - Default: 0

            - Choices: 0, 1
**\-\-inference_only** : <class 'int'>
            *Help*: Perform inference only for each task (no training).

            - Default: 0

            - Choices: 0, 1
**\-\-code_optimization** : <class 'int'>
            *Help*: Optimization level for the code.0: no optimization.1: Use TF32, if available.2: Use BF16, if available.3: Use BF16 and `torch.compile`. BEWARE: torch.compile may break your code if you change the model after the first run! Use with caution.

            - Default: 0

            - Choices: 0, 1, 2, 3
**\-\-distributed** : <class 'str'>
            *Help*: Enable distributed training?

            - Default: no

            - Choices: no, dp, ddp
**\-\-savecheck** : <class 'int'>
            *Help*: Save checkpoint?

            - Default: 0

            - Choices: 0, 1
**\-\-loadcheck** : <class 'str'>
            *Help*: Path of the checkpoint to load (.pt file for the specific task)

            - Default: None

            - Choices: 
**\-\-ckpt_name** : <class 'str'>
            *Help*: (optional) checkpoint save name.

            - Default: None

            - Choices: 
**\-\-start_from** : <class 'int'>
            *Help*: Task to start from

            - Default: None

            - Choices: 
**\-\-stop_after** : <class 'int'>
            *Help*: Task limit

            - Default: None

            - Choices: 

.. rubric:: Wandb arguments

*Arguments to manage logging on Wandb.*

**\-\-wandb_name** : <class 'str'>
            *Help*: Wandb name for this run. Overrides the default name (`args.model`).

            - Default: None

            - Choices: 
**\-\-wandb_entity** : <class 'str'>
            *Help*: Wandb entity

            - Default: None

            - Choices: 
**\-\-wandb_project** : <class 'str'>
            *Help*: Wandb project name

            - Default: mammoth

            - Choices: 

.. rubric:: REEHARSAL-ONLY ARGS

**\-\-buffer_size** : <class 'int'>
            *Help*: The size of the memory buffer.

            - Default: None

            - Choices: 

**\-\-minibatch_size** : <class 'int'>
            *Help*: The batch size of the memory buffer.

            - Default: None

            - Choices: 

