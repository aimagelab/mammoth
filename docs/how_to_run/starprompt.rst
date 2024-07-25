How to run STAR-Prompt
======================

.. important::

    You can find the complete paper at this `link <https://arxiv.org/abs/2403.06870>`_. The hyperparameters reported in Tab. D and E are the ones used to obtain the results in the paper. Here we report the best hyperparameters we found for each dataset after a more thorough search. The results are very similar to the ones reported in the paper.

STAR-Prompt
-----------

The most important hyperparameters for STAR-Prompt are a combination of those of the first and second stage (detailed below). The most important ones are:

- ``lambda_ortho_first_stage``: the weight of the orthogonality loss for the first stage.
- ``lambda_ortho_second_stage``: the weight of the orthogonality loss for the second stage.
- ``learning_rate_gr_first_stage``: the learning rate of the Generative Replay for the first stage.
- ``learning_rate_gr_second_stage``: the learning rate of the Generative Replay for the second stage.
- ``num_epochs_gr_first_stage``: the number of epochs for the Generative Replay for the first stage.
- ``num_epochs_gr_second_stage``: the number of epochs for the Generative Replay for the second stage.

The best configurations can be found in the tables below by merging the tables of the first and second stage.

.. note::

  In the paper we report the results with 3 different choices of random seeds: ``1993``, ``1996``, and ``1997``. We to not report the seed in the commands below for brevity but the seed can be set with the ``--seed`` argument.
  
First stage only
~~~~~~~~~~~~~~~~


In the following we report the commands to run the *first stage* of STAR-Prompt on the different datasets.

The most important Hyperparameters are:

* ``lambda_ortho_first_stage``: the weight of the orthogonality loss. :math:`\lambda` in the main paper (Alg 1, Tab D, E).
* ``learning_rate_gr``: the learning rate of the Generative Replay. :math:`lr` in the main paper (Alg 1, Tab D, E).
* ``num_epochs_gr``: the number of epochs for the Generative Replay. :math:`E_1` in the main paper (Alg 1, Tab D, E).

Other hyperparameters such as ``gr_mog_n_iters`` and ``num_monte_carlo_gr`` have a much smaller impact. Here are reported the best configurations, but the default ones already give pretty much the same results.

.. list-table:: Hyperparameter table
   :header-rows: 1

   * - Dataset
     - Command
   * - EuroSAT-RGB
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --gr_mog_n_iters=200 --lambda_ortho_first_stage=30 --dataset=seq-eurosat-rgb``
   * - CropDisease
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --lambda_ortho_first_stage=30 --learning_rate_gr=0.01 --dataset=seq-cropdisease``
   * - Resisc45
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_first_stage=10 --dataset=seq-resisc45``
   * - CIFAR-100
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --lambda_ortho_first_stage=10 --num_monte_carlo_gr=1 --dataset=seq-cifar100-224``
   * - Imagenet-R
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --gr_mog_n_iters=200 --lambda_ortho_first_stage=30 --dataset=seq-imagenet-r``
   * - ISIC
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_first_stage=5 --num_epochs_gr=50 --learning_rate_gr=0.01 --dataset=seq-isic``
   * - ChestX
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=10 --lambda_ortho_first_stage=30 --dataset=seq-chestx``
   * - CUB-200
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_first_stage=30 --num_epochs_gr=50 --num_monte_carlo_gr=5 --dataset=seq-cub200``
   * - Cars-196
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_first_stage=30 --learning_rate_gr=0.01 --dataset=seq-cars196``

Second stage only
~~~~~~~~~~~~~~~~~

The *second stage* of STAR-Prompt can take either the class-specific embeddings learned during the first stage or the pre-existing templates of CLIP. This is controlled by the ``--keys_ckpt_path`` argument. If supplied (see :ref:`module-second_stage_starprompt`), it will load the pre-trained embeddings from the first stage. If not supplied, it will use the pre-existing templates of CLIP. The most important Hyperparameters are:

* ``lambda_ortho_second_stage``: the weight of the orthogonality loss. :math:`\lambda` in the main paper (Alg 1, Tab D, E).
* ``learning_rate_gr``: the learning rate of the Generative Replay. :math:`lr` in the main paper (Alg 1, Tab D, E).
* ``num_epochs_gr``: the number of epochs for the Generative Replay. :math:`E_2` in the main paper (Alg 1, Tab D, E).

.. list-table:: Hyperparameter table
   :header-rows: 1

   * - Dataset
     - Command
   * - ISIC
     - ``--model=second_stage_starprompt --lr=0.001 --optimizer=adam --n_epochs=30 --num_epochs_gr=50 --num_monte_carlo_gr=5 --learning_rate_gr=0.01 --dataset=seq-isic --lambda_ortho_second_stage=50 --keys_ckpt_path=<path_to_keys_checkpoint>``
   * - CUB-200
     - ``--model=second_stage_starprompt --dataset=seq-cub200 --n_epochs=50 --batch_size=64 --virtual_bs_n=2 --lr=0.001 --optimizer=adam --lambda_ortho_second_stage=30 --learning_rate_gr=0.01 --num_monte_carlo_gr=5``