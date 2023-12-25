.. _module-args:

Arguments
=========

.. rubric:: EXPERIMENT-RELATED ARGS

**\-\-dataset** : <class 'str'>
            *Help*: Which dataset to perform experiments on.

            - Default: None

            - Choices: mnist-360, perm-mnist, rot-mnist, seq-cifar10, seq-cifar100, seq-cifar100-224, seq-cifar100-224-rs, seq-cub200, seq-mnist, seq-tinyimg, seq-tinyimg-r

**\-\-model** : <class 'str'>
            *Help*: Model name.

            - Default: None

            - Choices: agem, agem_r, bic, ccic, der, derpp, er, er_ace, ewc_on, fdr, gdumb, gem, gss, hal, icarl, joint_gcl, l2p, lucir, lwf, lwf_mc, mer, pnn, rpc, sgd, si, twf, xder, xder_ce, xder_rpc

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

**\-\-n_epochs** : <class 'int'>
            *Help*: Number of epochs.

            - Default: None

            - Choices: 

**\-\-batch_size** : <class 'int'>
            *Help*: Batch size.

            - Default: None

            - Choices: 

**\-\-distributed** : <class 'str'>
            *Help*: Distributed training?

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
            *Help*: Test on the validation set

            - Default: 0

            - Choices: 0, 1

**\-\-enable_other_metrics** : <class 'int'>
            *Help*: Enable additional metrics

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

