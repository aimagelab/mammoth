.. _module-args:

Arguments
=========

.. rubric:: EXPERIMENT-RELATED ARGS

**\-\-dataset** : <class 'str'>
            *Help*: Which dataset to perform experiments on.

            - Default: None

            - Choices: seq-tinyimg, seq-tinyimg-r, perm-mnist, seq-cifar10, seq-cifar100-224, seq-cub200, rot-mnist, seq-cifar100, seq-cifar100-224-rs, seq-mnist, mnist-360

**\-\-model** : <function custom_str_underscore at 0x7fe9f47a42c0>
            *Help*: Model name.

            - Default: None

            - Choices: agem, agem-r, ewc-on, derpp-lider, gdumb-lider, slca, dualprompt, si, bic, er-ace, fdr, gdumb, gem, gss, joint-gcl, lwf, mer, rpc, twf, ccic, der, derpp, er, hal, icarl, l2p, lucir, lwf-mc, sgd, xder, xder-ce, xder-rpc, pnn, er-ace-lider, icarl-lider, coda-prompt

**\-\-lr** : <class 'float'>
            *Help*: Learning rate.

            - Default: None

            - Choices: 

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

**\-\-n_epochs** : <class 'int'>
            *Help*: Number of epochs.

            - Default: None

            - Choices: 

**\-\-batch_size** : <class 'int'>
            *Help*: Batch size.

            - Default: None

            - Choices: 

**\-\-distributed** : <class 'str'>
            *Help*: Enable distributed training?

            - Default: no

            - Choices: no, dp, ddp

**\-\-savecheck** : None
            *Help*: Save checkpoint?

            - Default: False

            - Choices: 

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

**\-\-joint** : <class 'int'>
            *Help*: Train model on Joint (single task)?

            - Default: 0

            - Choices: 0, 1

**\-\-label_perc** : <class 'float'>
            *Help*: Percentage in (0-1] of labeled examples per task.

            - Default: 1

            - Choices: 

.. rubric:: MANAGEMENT ARGS

**\-\-seed** : <class 'int'>
            *Help*: The random seed.

            - Default: None

            - Choices: 

**\-\-permute_classes** : <class 'int'>
            *Help*: Permute classes before splitting tasks (applies seed before permute if seed is present)?

            - Default: 0

            - Choices: 0, 1

**\-\-base_path** : <class 'str'>
            *Help*: The base path where to save datasets, logs, results.

            - Default: ./data/

            - Choices: 

**\-\-notes** : <class 'str'>
            *Help*: Notes for this run.

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

**\-\-validation** : <class 'int'>
            *Help*: Percentage of validation set drawn from the training set.

            - Default: None

            - Choices: 

**\-\-enable_other_metrics** : <class 'int'>
            *Help*: Enable computing additional metrics: forward and backward transfer.

            - Default: 0

            - Choices: 0, 1

**\-\-debug_mode** : <class 'int'>
            *Help*: Run only a few forward steps per epoch

            - Default: 0

            - Choices: 0, 1

**\-\-wandb_entity** : <class 'str'>
            *Help*: Wandb entity

            - Default: None

            - Choices: 

**\-\-wandb_project** : <class 'str'>
            *Help*: Wandb project name

            - Default: mammoth

            - Choices: 

**\-\-eval_epochs** : <class 'int'>
            *Help*: Perform inference intra-task at every `eval_epochs`.

            - Default: None

            - Choices: 

**\-\-inference_only** : None
            *Help*: Perform inference only for each task (no training).

            - Default: False

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

