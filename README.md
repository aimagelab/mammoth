<p align="center">
  <img width="230" height="230" src="logo.png" alt="logo">
</p>

# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

Official repository of [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766) and [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

Mammoth is a framework for continual learning research. It is designed to be modular, easy to extend, and - most importantly - _easy to debug_.
Idelly, all the code necessary to run the experiments is included _in the repository_, without needing to check out other repositories or install additional packages.

With Mammoth, nothing is set in stone. You can easily add new models, datasets, training strategies, or functionalities.

## **NEW**: WIKI

We have created a [WIKI](https://aimagelab.github.io/mammoth/)! Check it out for more information on how to use Mammoth.

<p align="center">
  <img width="112" height="112" src="seq_mnist.gif" alt="Sequential MNIST">
  <img width="112" height="112" src="seq_cifar10.gif" alt="Sequential CIFAR-10">
  <img width="112" height="112" src="seq_tinyimg.gif" alt="Sequential TinyImagenet">
  <img width="112" height="112" src="perm_mnist.gif" alt="Permuted MNIST">
  <img width="112" height="112" src="rot_mnist.gif" alt="Rotated MNIST">
  <img width="112" height="112" src="mnist360.gif" alt="MNIST-360">
</p>

## Setup

- Use `./utils/main.py` to run experiments.
- Use argument `--load_best_args` to use the best hyperparameters from the paper.
- New models can be added to the `models/` folder.
- New datasets can be added to the `datasets/` folder.

## Models

- LiDER (on DER++, iCaRL, GDumb, and ER-ACE): `derpp_lider`, `icarl_lider`, `gdumb_lider`, `er_ace_lider`.
- eXtended-DER (X-DER): `xder` (full version), `xder_ce` (X-DER with CE), `xder_rpc` (X-DER with RPC).
- Dark Experience Replay (DER): `der`.
- Dark Experience Replay++ (DER++): `derpp`.
- Learning a Unified Classifier Incrementally via Rebalancing (LUCIR): `lucir`.
- Greedy Sampler and Dumb Learner (GDumb): `gdumb`.
- Bias Correction (BiC): `bic`.
- Regular Polytope Classifier (RPC): `rpc`.
- Gradient Episodic Memory (GEM) - _Unavailable on windows_: `gem`.
- A-GEM: `agem`.
- A-GEM with Reservoir (A-GEM-R): `agem_r`.
- Experience Replay (ER): `er`.
- Meta-Experience Replay (MER): `mer`.
- Function Distance Regularization (FDR): `fdr`.
- Greedy gradient-based Sample Selection (GSS): `gss`.
- Hindsight Anchor Learning (HAL): `hal`.
- Incremental Classifier and Representation Learning (iCaRL): `icarl`.
- online Elastic Weight Consolidation (oEWC): `ewc_on`.
- Synaptic Intelligence (SI): `si`.
- Learning without Forgetting (LwF): `lwf`.
- Progressive Neural Networks (PNN): `pnn`.
- Learning to Prompt (L2P) - _Requires_ `pip install timm==0.9.8`: `l2p`.
- DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning (DualPrompt) - _Requires_ ``pip install timm==0.9.8``: `dualprompt`.
- CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning (CODA-Prompt) - _Requires_ ``pip install timm==0.9.8``: `coda-prompt`.
- SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model (SLCA) - _Requires_ ``pip install timm==0.9.8``: `slca`.
- Transfer without Forgetting (TwF): `twf`.
- Continual Contrastive Interpolation Consistency (CCIC) - _Requires_ `pip install kornia`: `ccic`.
- JointGCL: `joint_gcl` (only for General Continual).

## Datasets

**NOTE**: Datasets are automatically downloaded in the `data/`.

- This can be changes by changing the `base_path` function in `utils/conf.py`.
- The `data/` folder is not tracked by git and is craeted automatically if missing.

- Sequential MNIST (_Class-Il / Task-IL_): `seq-mnist`.
- Sequential CIFAR-10 (_Class-Il / Task-IL_): `seq-cifar10`.
- Sequential Tiny ImageNet (_Class-Il / Task-IL_): `seq-tinyimg`.
- Sequential Tiny ImageNet resized 32x32 (_Class-Il / Task-IL_): `seq-tinyimg-r`.
- Sequential CIFAR-100 (_Class-Il / Task-IL_): `seq-cifar100`.
- Sequential CIFAR-100 resized 224x224 (ViT version) (_Class-Il / Task-IL_): `seq-cifar100-224`.
- Sequential CIFAR-100 resized 224x224 (ResNet50 version) (_Class-Il / Task-IL_): `seq-cifar100-224-rs`.
- Permuted MNIST (_Domain-IL_): `perm-mnist`.
- Rotated MNIST (_Domain-IL_): `rot-mnist`.
- MNIST-360 (_General Continual Learning_): `mnist-360`.
- Sequential CUB-200 (_Class-Il / Task-IL_): `seq-cub200`.

## Pretrained backbones

- [ResNet18 on cifar100](https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII)
- [ResNet18 on TinyImagenet resized (seq-tinyimg-r)](https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok)
- [ResNet50 on ImageNet (pytorch version)](https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M)
- [ResNet18 on SVHN](https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/ETdCpRoA891KsAAuibMKWYwBX_3lfw3dMbE4DFEkhOm96A?e=NjdzLN)

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

## Awesome Papers using Mammoth

### Our Papers

- Dark Experience for General Continual Learning: a Strong, Simple Baseline (**NeurIPS 2020**) [[paper](https://arxiv.org/abs/2004.07211)]
- Rethinking Experience Replay: a Bag of Tricks for Continual Learning (**ICPR 2020**) [[paper](https://arxiv.org/abs/2010.05595)] [[code](https://github.com/hastings24/rethinking_er)]
- Class-Incremental Continual Learning into the eXtended DER-verse (**TPAMI 2022**) [[paper](https://arxiv.org/abs/2201.00766)]
- Effects of Auxiliary Knowledge on Continual Learning (**ICPR 2022**) [[paper](https://arxiv.org/abs/2206.02577)]
- Transfer without Forgetting (**ECCV 2022**) [[paper](https://arxiv.org/abs/2206.00388)][[code](https://github.com/mbosc/twf)]
- Continual semi-supervised learning through contrastive interpolation consistency (**PRL 2022**) [[paper](https://arxiv.org/abs/2108.06552)][[code](https://github.com/aimagelab/CSSL)]
- On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning (**NeurIPS 2022**) [[paper](https://arxiv.org/abs/2210.06443)] [[code](https://github.com/aimagelab/lider)]

### Other Awesome CL works using Mammoth

- New Insights on Reducing Abrupt Representation Change in Online Continual Learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=N8MaByOzUfb)] [[code](https://github.com/pclucas14/AML)]
- Learning fast, learning slow: A general continual learning method based on complementary learning system (**ICLR2022**) [[paper](https://openreview.net/pdf?id=uxxFrDwrE7Y)] [[code](https://github.com/NeurAI-Lab/CLS-ER)]
- Self-supervised models are continual learners (**CVPR2022**) [[paper](https://arxiv.org/abs/2112.04215)] [[code](https://github.com/DonkeyShot21/cassle)]
- Representational continuity for unsupervised continual learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=9Hrka5PA7LW)] [[code](https://github.com/divyam3897/UCL)]
- Continual Learning by Modeling Intra-Class Variation (**TMLR 2023**) [[paper](https://arxiv.org/abs/2210.05398)] [[code](https://github.com/yulonghui/MOCA)]
- Consistency is the key to further Mitigating Catastrophic Forgetting in Continual Learning (**CoLLAs2022**) [[paper](https://arxiv.org/pdf/2207.04998.pdf)] [[code](https://github.com/NeurAI-Lab/ConsistencyCL)]
- Continual Normalization: Rethinking Batch Normalization for Online Continual Learning (**ICLR2022**) [[paper](https://arxiv.org/abs/2203.16102)] [[code](https://github.com/phquang/Continual-Normalization)]
- NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks (**ICML2022**) [[paper](https://arxiv.org/abs/2206.09117)]
- Learning from Students: Online Contrastive Distillation Network for General Continual Learning (**IJCAI2022**) [[paper](https://www.ijcai.org/proceedings/2022/0446.pdf)] [[code](https://github.com/lijincm/OCD-Net)]

## Update Roadmap

In the near future, we plan to incorporate the following improvements into this master repository:

- ER+Tricks (_Rethinking Experience Replay: a Bag of Tricks for Continual Learning_)
- CCIC & CSSL Baselines (_Continual semi-supervised learning through contrastive interpolation consistency_)
- LiDER (_On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning_)
- Additional X-DER datasets (_Class-Incremental Continual Learning into the eXtended DER-verse_)

Pull requests welcome! [Get in touch](mailto:matteo.boschini@unimore.it)

### Contributing

Please use `autopep8` with parameters:

- `--aggressive`
- `--max-line-length=200`
- `--ignore=E402`

## Previous versions

If you're interested in a version of this repo that only includes the original code for [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) or [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766>), please use the following tags:

- [neurips2020](https://github.com/aimagelab/mammoth/releases/tag/neurips2020) for DER (NeurIPS 2020).
- [tpami2023](https://github.com/aimagelab/mammoth/releases/tag/tpami2023) for X-DER (TPAMI 2022).
