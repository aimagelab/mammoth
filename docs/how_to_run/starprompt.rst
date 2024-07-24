How to run STAR-Prompt
======================

.. important::

    You can find the complete paper at this `link <https://arxiv.org/abs/2403.06870>`_.

First stage only
----------------

In the following we report the commands to run the *first stage* of STAR-Prompt on the different datasets.

The most important Hyperparameters are:

* ``lambda_ortho_coop``: the weight of the orthogonality loss. :math:`\lambda` in the main paper (Alg 1, Tab D, E).
* ``learning_rate_gr``: the learning rate of the Generative Replay. :math:`lr` in the main paper (Alg 1, Tab D, E).
* ``num_epochs_gr``: the number of epochs for the Generative Replay. :math:`E_1` in the main paper (Alg 1, Tab D, E).

Other hyperparameters such as ``gr_mog_n_iters`` and ``num_monte_carlo_gr`` have a much smaller impact. Here are reported the best configurations, but the default ones already give pretty much the same results.

.. list-table:: Hyperparameter table
   :header-rows: 1

   * - Dataset
     - Command
   * - EuroSAT-RGB
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --gr_mog_n_iters=200 --lambda_ortho_coop=30 --dataset=seq-eurosat-rgb``
   * - CropDisease
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=5 --lambda_ortho_coop=30 --learning_rate_gr=0.01 --dataset=seq-cropdisease``
   * - Resisc45
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_coop=10 --dataset=seq-resisc45``
   * - CIFAR-100
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --lambda_ortho_coop=10 --num_monte_carlo_gr=1 --dataset=seq-cifar100-224``
   * - Imagenet-R
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=20 --gr_mog_n_iters=200 --lambda_ortho_coop=30 --dataset=seq-imagenet-r``
   * - ISIC
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=30 --lambda_ortho_coop=5 --num_epochs_gr=50 --learning_rate_gr=0.01 --dataset=seq-isic``
   * - ChestX
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=10 --lambda_ortho_coop=30 --dataset=seq-chestx``
   * - CUB-200
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_coop=30 --num_epochs_gr=50 --num_monte_carlo_gr=5 --dataset=seq-cub200``
   * - Cars-196
     - ``--model=first_stage_starprompt --lr=0.002 --n_epochs=50 --lambda_ortho_coop=30 --learning_rate_gr=0.01 --dataset=seq-cars196``

Second stage only
-----------------

The *second stage* of STAR-Prompt can take either the class-specific embeddings learned during the first stage or the pre-existing templates of CLIP. This is controlled by the ``--keys_ckpt_path`` argument. If supplied (see :ref:`module-second_stage_starprompt`), it will load the pre-trained embeddings from the first stage. If not supplied, it will use the pre-existing templates of CLIP. The most important Hyperparameters are:
