.. _module-fast-training:

Fast training \& optimizations
==============================

.. important::
    The optimizations described in this section require an NVIDIA GPU with the `Ampere architecture <https://www.nvidia.com/en-gb/data-center/ampere-architecture/>`_ (RTX 30xx series or newer) and the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed. If you do not have an Ampere GPU, you can still use Mammoth without these optimizations.

Mammoth provides a number of optimizations to speed up training. These are **disabled** by default (mainly to improve ease of debugging), but can be enabled by passing the `--code_optimization` (or `-O`) flag to ``utils/main.py``. The available optimizations are:

* **0**: No optimization (default)
* **1**: Use the ``TF32`` data type for training IF IT IS AVAILABLE (*i.e.*, sets the `torch.set_float32_matmul_precision` to `high`). **This will fall back to FP32 if TF32 is not available**.
* **2**: Use the ``BF16`` data type for training (*i.e.*, sets the `torch.set_bf16_cvt_precision` to `medium`). **This will throw an error if the GPU does not support BF16**.
* **3**: Same as *2*, but also includes ``torch.compile``. This option has some caveats:
    - It is only available on Linux (check `this issue <https://github.com/pytorch/pytorch/issues/90768>`_ for updates).
    - It does not work if the model *changes* during training. This includes increasing the number of classifiers, prompts, etc.
    - It may not give a significant speedup for small models.

Distributed training
--------------------

Mammoth supports distributed training via `DataParallel <https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed>`_. To use it, simply pass the `--distributed=dp` argument to ``utils/main.py``. This will automatically use all available GPUs on the machine using the **make_dp** function in :ref:`module-utils.distributed`.

DataParallel training **splits the batch** across GPUs and performs the forward and backward passes on each GPU. The gradients are then **averaged** across GPUs and the model parameters are updated. This is the simplest form of distributed training supported by PyTorch and is the only one supported by Mammoth as of now.

.. important::
    As of now, Mammoth only supports DataParallel training. This is due to the difficulty of synchronizing the memory buffer across multiple GPUs after each batch. However, experimental support for `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ training in a `slurm <https://slurm.schedmd.com/documentation.html>`_ cluster is available in the :ref:`module-utils.distributed` module via the **make_ddp** function. 