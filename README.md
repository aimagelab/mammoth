# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

## NEWS

**STAY TUNED:** The additional code for the MiniImagenet and NTU-60 experiments will be uploaded soon due to complications in the distribution of datasets.

-----------------------------


Official repository of [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766) and [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

<p align="center">
  <img width="112" height="112" src="seq_mnist.gif" alt="Sequential MNIST">
  <img width="112" height="112" src="seq_cifar10.gif" alt="Sequential CIFAR-10">
  <img width="112" height="112" src="seq_tinyimg.gif" alt="Sequential TinyImagenet">
  <img width="112" height="112" src="perm_mnist.gif" alt="Permuted MNIST">
  <img width="112" height="112" src="rot_mnist.gif" alt="Rotated MNIST">
  <img width="112" height="112" src="mnist360.gif" alt="MNIST-360">
</p>

## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Models

+ eXtended-DER (X-DER)

+ Dark Experience Replay (DER)
+ Dark Experience Replay++ (DER++)

+ Learning a Unified Classifier Incrementally via Rebalancing (LUCIR)
+ Greedy Sampler and Dumb Learner (GDumb)
+ Bias Correction (BiC)
+ Regular Polytope Classifier (RPC)

+ Gradient Episodic Memory (GEM)
+ A-GEM
+ A-GEM with Reservoir (A-GEM-R)
+ Experience Replay (ER)
+ Meta-Experience Replay (MER)
+ Function Distance Regularization (FDR)
+ Greedy gradient-based Sample Selection (GSS)
+ Hindsight Anchor Learning (HAL)
+ Incremental Classifier and Representation Learning (iCaRL)
+ online Elastic Weight Consolidation (oEWC)
+ Synaptic Intelligence
+ Learning without Forgetting
+ Progressive Neural Networks

## Datasets

**Class-Il / Task-IL settings**

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential Tiny ImageNet
+ Sequential CIFAR-100

**Domain-IL settings**

+ Permuted MNIST
+ Rotated MNIST

**General Continual Learning setting**

+ MNIST-360

## Citing these works

```
@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@inproceedings{buzzega2020dark,
 author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {15920--15930},
 publisher = {Curran Associates, Inc.},
 title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
 volume = {33},
 year = {2020}
}
```

## Previous versions

If you're interested in a version of this repo that only includes the code for [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), please use our [neurips2020 tag](https://github.com/aimagelab/mammoth/releases/tag/neurips2020).
