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

The best configurations can be found in the tables below by merging the tables of the first and second stage. The only difference is that the number of epochs for the first stage is set as ``--first_stage_epochs`` (by default, is set as ``--n_epochs``).

.. note::

  In the paper we report the results with 3 different choices of random seeds: ``1993``, ``1996``, and ``1997``. We to not report the seed in the commands below for brevity but it can be set with ``--seed=<seed>``. We also set ``--permute_classes=1`` to shuffle the classes before splitting them into tasks. For example, to run STAR-Prompt on the CIFAR-100 with a seed of ``1993``, run the following command
  
  .. code-block:: bash

    python utils/main.py --model starprompt --dataset seq-cifar100-224 --seed 1993 --permute_classes=1
  
First stage only
~~~~~~~~~~~~~~~~

In the following we report the commands to run the *first stage* of STAR-Prompt on the different datasets.

The most important Hyperparameters are:

* ``lambda_ortho_first_stage``: the weight of the orthogonality loss. :math:`\lambda` in the main paper (Alg 1, Tab D, E).
* ``learning_rate_gr_first_stage``: the learning rate of the Generative Replay. :math:`lr` in the main paper (Alg 1, Tab D, E).
* ``num_epochs_gr_first_stage``: the number of epochs for the Generative Replay. :math:`E_1` in the main paper (Alg 1, Tab D, E).

Other hyperparameters such as ``gr_mog_n_iters`` and ``num_monte_carlo_gr`` have a much smaller impact. Here are reported the best configurations, but the default ones already give pretty much the same results.

.. list-table:: Hyperparameter table
   :header-rows: 1

   * - Dataset
     - Command
   * - EuroSAT-RGB
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --gr_mog_n_iters_first_stage=200 --lambda_ortho_first_stage=30 --dataset=seq-eurosat-rgb``
   * - CropDisease
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --lambda_ortho_first_stage=30 --learning_rate_gr_first_stage=0.01 --dataset=seq-cropdisease``
   * - Resisc45
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_first_stage=10 --dataset=seq-resisc45``
   * - CIFAR-100
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --lambda_ortho_first_stage=10 --num_monte_carlo_gr_first_stage=1 --dataset=seq-cifar100-224``
   * - Imagenet-R
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --gr_mog_n_iters_first_stage=200 --lambda_ortho_first_stage=30 --dataset=seq-imagenet-r``
   * - ISIC
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_first_stage=5 --num_epochs_gr_first_stage=50 --learning_rate_gr_first_stage=0.01 --dataset=seq-isic``
   * - ChestX
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=10 --lambda_ortho_first_stage=30 --dataset=seq-chestx``
   * - CUB-200
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_first_stage=30 --num_epochs_gr_first_stage=50 --num_monte_carlo_gr_first_stage=5 --dataset=seq-cub200``
   * - Cars-196
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_first_stage=30 --learning_rate_gr_first_stage=0.01 --dataset=seq-cars196``

Second stage only
~~~~~~~~~~~~~~~~~

The *second stage* of STAR-Prompt can take either the class-specific embeddings learned during the first stage or the pre-existing templates of CLIP. This is controlled by the ``--keys_ckpt_path`` argument. If supplied (see :ref:`module-models.second_stage_starprompt`), it will load the pre-trained embeddings from the first stage. If not supplied, it will use the pre-existing templates of CLIP. The most important Hyperparameters are:

* ``lambda_ortho_second_stage``: the weight of the orthogonality loss. :math:`\lambda` in the main paper (Alg 1, Tab D, E).
* ``learning_rate_gr_first_stage``: the learning rate of the Generative Replay. :math:`lr` in the main paper (Alg 1, Tab D, E).
* ``num_epochs_gr_second_stage``: the number of epochs for the Generative Replay. :math:`E_2` in the main paper (Alg 1, Tab D, E).

.. important::

  Remember to set the ``--keys_ckpt_path`` argument to the path of the checkpoint of the first stage. Otherwise, the second stage will not be able to load the class-specific embeddings and will use the pre-existing templates of CLIP.

.. list-table:: Hyperparameter table
   :header-rows: 1

   * - Dataset
     - Command
   * - ISIC
     - ``--model=second_stage_starprompt --lr=0.001 --optimizer=adam --n_epochs=30 --num_epochs_gr_second_stage=50 --num_monte_carlo_gr_second_stage=5 --learning_rate_gr_second_stage=0.01 --dataset=seq-isic --lambda_ortho_second_stage=50 --keys_ckpt_path=<path_to_keys_checkpoint>``
   * - CUB-200
     - ``--model=second_stage_starprompt --dataset=seq-cub200 --n_epochs=50 --lr=0.001 --optimizer=adam --lambda_ortho_second_stage=30 --learning_rate_gr_second_stage=0.01 --num_monte_carlo_gr_second_stage=5``
   * - Imagenet-R 
     - ``--model=second_stage_starprompt --optimizer=adam --dataset=seq-imagenet-r --batch_size=16 --n_epochs=5 --lr=0.001 --lambda_ortho_second_stage=10 --learning_rate_gr_second_stage=0.001``
   * - CIFAR-100
     - ``--model=second_stage_starprompt --dataset=seq-cifar100-224 --n_epochs=20 --lr=0.001 --optimizer=adam --lambda_ortho_second_stage=2 --learning_rate_gr_second_stage=0.001``
   * - ChestX
     - ``--model=second_stage_starprompt --dataset=seq-chestx --n_epochs=30 --lr=0.001 --optimizer=adam --lambda_ortho_second_stage=5 --learning_rate_gr_second_stage=0.05 --num_monte_carlo_gr_second_stage=1``
   * - CropDisease
     - ``--model=second_stage_starprompt --optimizer=adam --dataset=seq-cropdisease --lr=0.001 --lambda_ortho_second_stage=5 --learning_rate_gr_second_stage=0.001 --num_monte_carlo_gr_second_stage=5 --num_epochs_gr_second_stage=10``
   * - Cars-196
     - ``--model=second_stage_starprompt --dataset=seq-cars196 --n_epochs=50 --lr=0.001 --optimizer=adam --lambda_ortho_second_stage=10 --learning_rate_gr_second_stage=0.01``
   * - Resisc45
     - ``--model=second_stage_starprompt --lr=0.001 --optimizer=adam --dataset=seq-resisc45 --n_epochs=30 --lambda_ortho_second_stage=5 --learning_rate_gr_second_stage=0.01 --num_monte_carlo_gr_second_stage=1 --num_epochs_gr_second_stage=50``
   * - Cars-196
     - ``--model=second_stage_starprompt --num_monte_carlo_gr_second_stage=2 --optimizer=adam --dataset=seq-eurosat-rgb --lr=0.001 --lambda_ortho_second_stage=5.0 --learning_rate_gr_second_stage=0.1``