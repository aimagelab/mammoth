Welcome to Mammoth's documentation!
===================================
.. image:: _static/logo.png
    :alt: logo
    :align: center
    :height: 230px
    :width: 230px

Mammoth - An Extendible (General) Continual Learning Framework for Pytorch
==========================================================================

Official repository of:

- `Class-Incremental Continual Learning into the eXtended DER-verse <https://arxiv.org/abs/2201.00766>`_
- `Dark Experience for General Continual Learning: a Strong, Simple Baseline <https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html>`_
- `Semantic Residual Prompts for Continual Learning <https://arxiv.org/abs/2403.06870>`_
- `CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning <https://arxiv.org/abs/2407.15793>`_

Mammoth is a framework for continual learning research. With **more than 40 methods and 20 datasets**, it includes the most complete list competitors and benchmarks for research purposes.

The core idea of Mammoth is that it is designed to be modular, easy to extend, and - most importantly - *easy to debug*.
Ideally, all the code necessary to run the experiments is included *in the repository*, without needing to check out other repositories or install additional packages.

With Mammoth, nothing is set in stone. You can easily add new models, datasets, training strategies, or functionalities.

.. important::

    **All the models included in mammoth are verified against the original papers (or subsequent relevant papers) to reproduce their original results.**

.. list-table::
   :widths: 15 15 15 15 15 15
   :class: centered
   :stub-columns: 0

   * - .. image:: _static/seq_mnist.gif
         :alt: Sequential MNIST
         :height: 112px
         :width: 112px

     - .. image:: _static/seq_cifar10.gif
         :alt: Sequential CIFAR-10
         :height: 112px
         :width: 112px

     - .. image:: _static/seq_tinyimg.gif
         :alt: Sequential TinyImagenet
         :height: 112px
         :width: 112px

     - .. image:: _static/perm_mnist.gif
         :alt: Permuted MNIST
         :height: 112px
         :width: 112px

     - .. image:: _static/rot_mnist.gif
         :alt: Rotated MNIST
         :height: 112px
         :width: 112px

     - .. image:: _static/mnist360.gif
         :alt: MNIST-360
         :height: 112px
         :width: 112px

Setup
-----

- Install with ``pip install -r requirements.txt``.
- Use ``./utils/main.py`` to run experiments.
- New models can be added to the ``models/`` folder.
- New datasets can be added to the ``datasets/`` folder.

.. note::
    **Pytorch version >=2.1.0 is required for scaled_dot_product_attention** (see: https://github.com/Lightning-AI/litgpt/issues/763). If you cannot support this version, the slower base version (see `backbone/vit.py`).

Models
------

Mammoth currently supports **more than 40 models**, with new releases covering the main competitors in literature.

Datasets
--------

**NOTE**: Datasets are automatically downloaded in ``data/``.
- This can be changed by changing the ``base_path`` function in ``utils/conf.py`` or using the ``--base_path`` argument.
- The ``data/`` folder should not be tracked by git and is craeted automatically if missing.

Mammoth includes **21** datasets, covering *toy classification problems* (different versions of MNIST), *standard domains* (CIFAR, Imagenet-R, TinyImagenet, MIT-67), *fine-grained classification domains* (Cars-196, CUB-200), *aerial domains* (EuroSAT-RGB, Resisc45), *medical domains* (CropDisease, ISIC, ChestX).

Work in progress
----------------

All the code is under active development. Here are some of the features we are working on:

- **Configurations for datasets**: Currently, each dataset represents a *specific configuration* (e.g., number of tasks, data augmentations, backbone, etc.). This makes adding a new *setting* a bit cumbersome. We are working on a more flexible way to define configurations, while leaving the current system as a default for retro-compatibility.
- **New models**: We are working on adding new models to the repository.
- **New training modalities**: We will introduce new CL training regimes, such as training with *noisy labels*, *regression*, *segmentation*, *detection*, etc.
- **Openly accessible result dashboard**: We are working on a dashboard to visualize the results of all the models in both their respective settings (to prove their reproducibility) and in a general setting (to compare them). 

All the new additions will try to preserve the current structure of the repository, making it easy to add new functionalities with a simple merge.

Pretrained backbones
--------------------

- `ResNet18 on cifar100 <https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII>`_
- `ResNet18 on TinyImagenet resized (seq-tinyimg-r) <https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok>`_
- `ResNet50 on ImageNet (pytorch version) <https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M>`_
- `ResNet18 on SVHN <https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/ETdCpRoA891KsAAuibMKWYwBX_3lfw3dMbE4DFEkhOm96A?e=NjdzLN>`_