<p align="center">
  <img width="230" height="230" src="logo.png" alt="logo">
</p>

# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

Official repository of [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766) and [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

Mammoth is a framework for continual learning research. It is designed to be modular, easy to extend, and - most importantly - _easy to debug_.
Idelly, all the code necessary to run the experiments is included _in the repository_, without needing to check out other repositories or install additional packages.

With Mammoth, nothing is set in stone. You can easily add new models, datasets, training strategies, or functionalities.

Join our Discord Server for all your Mammoth-related questions → ![Discord Shield](https://discordapp.com/api/guilds/1164956257392799860/widget.png?style=shield)

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

- Efficient Lifelong Learning with A-GEM (A-GEM, A-GEM-R - A-GEM with reservoir buffer): `agem`, `agem_r`
- Bias Correction (BiC): `bic`.
- Continual Contrastive Interpolation Consistency (CCIC) - _Requires_ `pip install kornia`: `ccic`.
- CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning (CODA-Prompt) - _Requires_ `pip install timm==0.9.8`: `coda-prompt`.
- Dark Experience Replay (DER): `der`.
- Dark Experience Replay++ (DER++): `derpp`.
- DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning (DualPrompt) - _Requires_ `pip install timm==0.9.8`: `dualprompt`.
- Experience Replay (ER): `er`.
- online Elastic Weight Consolidation (oEWC): `ewc_on`.
- Function Distance Regularization (FDR): `fdr`.
- Greedy Sampler and Dumb Learner (GDumb): `gdumb`.
- Gradient Episodic Memory (GEM) - _Unavailable on windows_: `gem`.
- Greedy gradient-based Sample Selection (GSS): `gss`.
- Hindsight Anchor Learning (HAL): `hal`.
- Incremental Classifier and Representation Learning (iCaRL): `icarl`.
- JointGCL: `joint_gcl` (only for General Continual).
- Learning to Prompt (L2P) - _Requires_ `pip install timm==0.9.8`: `l2p`.
- LiDER (on DER++, iCaRL, GDumb, and ER-ACE): `derpp_lider`, `icarl_lider`, `gdumb_lider`, `er_ace_lider`.
- Learning a Unified Classifier Incrementally via Rebalancing (LUCIR): `lucir`.
- Learning without Forgetting (LwF): `lwf`.
- Meta-Experience Replay (MER): `mer`.
- Progressive Neural Networks (PNN): `pnn`.
- Regular Polytope Classifier (RPC): `rpc`.
- Synaptic Intelligence (SI): `si`.
- SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model (SLCA) - _Requires_ `pip install timm==0.9.8`: `slca`.
- Transfer without Forgetting (TwF): `twf`.
- eXtended-DER (X-DER): `xder` (full version), `xder_ce` (X-DER with CE), `xder_rpc` (X-DER with RPC).

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
- Sequential ImageNet-R (_Class-Il / Task-IL_): `seq-imagenet-r`.

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

***Get in touch if we missed your awesome work!***

- A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm (**NeurIPS2023**) [[paper](https://arxiv.org/pdf/2310.12244.pdf)] [[code](https://github.com/Wang-ML-Lab/unified-continual-learning)]
- TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion (**NeurIPS2023**) [[paper](https://arxiv.org/pdf/2310.08217.pdf)] [[code](https://github.com/NeurAI-Lab/TriRE)]
- Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation (**NeurIPS2023**) [[paper](https://arxiv.org/pdf/2310.08855.pdf)] [[code](https://github.com/lvyilin/AdaB2N)]
- Decoupling Learning and Remembering: a Bilevel Memory Framework with Knowledge Projection for Task-Incremental Learning (**CVPR2023**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Decoupling_Learning_and_Remembering_A_Bilevel_Memory_Framework_With_Knowledge_CVPR_2023_paper.pdf)] [[code](https://github.com/SunWenJu123/BMKP)]
- Regularizing Second-Order Influences for Continual Learning (**CVPR2023**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Regularizing_Second-Order_Influences_for_Continual_Learning_CVPR_2023_paper.pdf)] [[code](https://github.com/feifeiobama/InfluenceCL)]
- Sparse Coding in a Dual Memory System for Lifelong Learning (**CVPR2023**) [[paper](https://arxiv.org/pdf/2301.05058.pdf)] [[code](https://github.com/NeurAI-Lab/SCoMMER)]
- A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm (**CVPR2023**) [[paper](https://arxiv.org/pdf/2310.12244.pdf)] [[code](https://github.com/Wang-ML-Lab/unified-continual-learning)]
- A Multi-Head Model for Continual Learning via Out-of-Distribution Replay (**CVPR2023**) [[paper](https://arxiv.org/pdf/2208.09734.pdf)] [[code](https://github.com/k-gyuhak/MORE)]
- Preserving Linear Separability in Continual Learning by Backward Feature Projection (**CVPR2023**) [[paper](https://arxiv.org/pdf/2303.14595.pdf)] [[code](https://github.com/rvl-lab-utoronto/BFP)]
- Complementary Calibration: Boosting General Continual Learning With Collaborative Distillation and Self-Supervision (**TIP2023**) [[paper](https://ieeexplore.ieee.org/document/10002397)] [[code](https://github.com/lijincm/CoCa)]
- Continual Learning by Modeling Intra-Class Variation (**TMLR2023**) [[paper](https://arxiv.org/abs/2210.05398)] [[code](https://github.com/yulonghui/MOCA)]
- ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis (**ICCV2023**) [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_ConSlide_Asynchronous_Hierarchical_Interaction_Transformer_with_Breakup-Reorganize_Rehearsal_for_Continual_ICCV_2023_paper.pdf)] [[code](https://github.com/HKU-MedAI/ConSlide)]
- CBA: Improving Online Continual Learning via Continual Bias Adaptor (**ICCV2023**) [[paper](https://arxiv.org/pdf/2308.06925.pdf)] [[code](https://github.com/wqza/CBA-online-CL)]
- Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal (**ICML2023**) [[paper](https://arxiv.org/pdf/2302.01242.pdf)] [[code](https://github.com/ema-marconato/NeSy-CL)]
- Learnability and Algorithm for Continual Learning (**ICML2023**) [[paper](https://arxiv.org/pdf/2306.12646.pdf)] [[code](https://github.com/k-gyuhak/CLOOD)]
- Pretrained Language Model in Continual Learning: a Comparative Study (**ICLR2022**) [[paper](https://openreview.net/pdf?id=figzpGMrdD)] [[code](https://github.com/wutong8023/PLM4CL)]
- Representational continuity for unsupervised continual learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=9Hrka5PA7LW)] [[code](https://github.com/divyam3897/UCL)]
- Continual Normalization: Rethinking Batch Normalization for Online Continual Learning (**ICLR2022**) [[paper](https://arxiv.org/abs/2203.16102)] [[code](https://github.com/phquang/Continual-Normalization)]
- Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System (**ICLR2022**) [[paper](https://arxiv.org/pdf/2201.12604.pdf)] [[code](https://github.com/NeurAI-Lab/CLS-ER)]
- New Insights on Reducing Abrupt Representation Change in Online Continual Learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=N8MaByOzUfb)] [[code](https://github.com/pclucas14/AML)]
- Looking Back on Learned Experiences for Class/Task Incremental Learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=RxplU3vmBx)] [[code](https://github.com/MozhganPourKeshavarz/Cost-Free-Incremental-Learning)]
- Task Agnostic Representation Consolidation: a Self-supervised based Continual Learning Approach (**CoLLAs2022**) [[paper](https://arxiv.org/pdf/2207.06267.pdf)] [[code](https://github.com/NeurAI-Lab/TARC)]
- Consistency is the key to further Mitigating Catastrophic Forgetting in Continual Learning (**CoLLAs2022**) [[paper](https://arxiv.org/pdf/2207.04998.pdf)] [[code](https://github.com/NeurAI-Lab/ConsistencyCL)]
- Self-supervised models are continual learners (**CVPR2022**) [[paper](https://arxiv.org/abs/2112.04215)] [[code](https://github.com/DonkeyShot21/cassle)]
- Learning from Students: Online Contrastive Distillation Network for General Continual Learning (**IJCAI2022**) [[paper](https://www.ijcai.org/proceedings/2022/0446.pdf)] [[code](https://github.com/lijincm/OCD-Net)]


### Contributing

Pull requests welcome!

Please use `autopep8` with parameters:

- `--aggressive`
- `--max-line-length=200`
- `--ignore=E402`

## Previous versions

If you're interested in a version of this repo that only includes the original code for [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) or [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766>), please use the following tags:

- [neurips2020](https://github.com/aimagelab/mammoth/releases/tag/neurips2020) for DER (NeurIPS 2020).
- [tpami2023](https://github.com/aimagelab/mammoth/releases/tag/tpami2023) for X-DER (TPAMI 2022).
