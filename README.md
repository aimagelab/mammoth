<p align="center">
  <img width="230" height="230" src="logo.png" alt="logo">
</p>

# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch


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
+ Synaptic Intelligence (SI)
+ Learning without Forgetting (LwF)
+ Progressive Neural Networks (PNN)

## Datasets

+ Sequential MNIST (*Class-Il / Task-IL*)
+ Sequential CIFAR-10 (*Class-Il / Task-IL*)
+ Sequential Tiny ImageNet (*Class-Il / Task-IL*)
+ Sequential CIFAR-100 (*Class-Il / Task-IL*)
+ Permuted MNIST (*Domain-IL*)
+ Rotated MNIST (*Domain-IL*)
+ MNIST-360 (*General Continual Learning*)

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

+ Dark Experience for General Continual Learning: a Strong, Simple Baseline (**NeurIPS 2020**) [[paper](https://arxiv.org/abs/2004.07211)]
+ Rethinking Experience Replay: a Bag of Tricks for Continual Learning (**ICPR 2020**) [[paper](https://arxiv.org/abs/2010.05595)] [[code](https://github.com/hastings24/rethinking_er)]
+ Class-Incremental Continual Learning into the eXtended DER-verse (**TPAMI 2022**) [[paper](https://arxiv.org/abs/2201.00766)]
+ Effects of Auxiliary Knowledge on Continual Learning (**ICPR 2022**) [[paper](https://arxiv.org/abs/2206.02577)]
+ Transfer without Forgetting  (**ECCV 2022**) [[paper](https://arxiv.org/abs/2206.00388)][[code](https://github.com/mbosc/twf)]
+ Continual semi-supervised learning through contrastive interpolation consistency (**PRL 2022**) [[paper](https://arxiv.org/abs/2108.06552)][[code](https://github.com/aimagelab/CSSL)]
+ On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning (**NeurIPS 2022**) [[paper](https://arxiv.org/abs/2210.06443)] [[code](https://github.com/aimagelab/lider)]

### Other Awesome CL works using Mammoth

+ New Insights on Reducing Abrupt Representation Change in Online Continual Learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=N8MaByOzUfb)] [[code](https://github.com/pclucas14/AML)]
+ Learning fast, learning slow: A general continual learning method based on complementary learning system (**ICLR2022**) [[paper](https://openreview.net/pdf?id=uxxFrDwrE7Y)] [[code](https://github.com/NeurAI-Lab/CLS-ER)]
+ Self-supervised models are continual learners (**CVPR2022**) [[paper](https://arxiv.org/abs/2112.04215)] [[code](https://github.com/DonkeyShot21/cassle)]
+ Representational continuity for unsupervised continual learning (**ICLR2022**) [[paper](https://openreview.net/pdf?id=9Hrka5PA7LW)] [[code](https://github.com/divyam3897/UCL)]
+ Continual Learning by Modeling Intra-Class Variation (**TMLR 2023**) [[paper](https://arxiv.org/abs/2210.05398)] [[code](https://github.com/yulonghui/MOCA)]
+ Consistency is the key to further Mitigating Catastrophic Forgetting in Continual Learning (**CoLLAs2022**) [[paper](https://arxiv.org/pdf/2207.04998.pdf)] [[code](https://github.com/NeurAI-Lab/ConsistencyCL)]
+ Continual Normalization: Rethinking Batch Normalization for Online Continual Learning (**ICLR2022**) [[paper](https://arxiv.org/abs/2203.16102)] [[code](https://github.com/phquang/Continual-Normalization)]
+ NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks (**ICML2022**) [[paper](https://arxiv.org/abs/2206.09117)]
+ Learning from Students: Online Contrastive Distillation Network for General Continual Learning (**IJCAI2022**) [[paper](https://www.ijcai.org/proceedings/2022/0446.pdf)] [[code](https://github.com/lijincm/OCD-Net)]

## Update Roadmap

In the near future, we plan to incorporate the following improvements into this master repository:

+ ER+Tricks (*Rethinking Experience Replay: a Bag of Tricks for Continual Learning*)
+ TwF & Pretraining Baselines (*Transfer without Forgetting*)
+ CCIC & CSSL Baselines (*Continual semi-supervised learning through contrastive interpolation consistency*)
+ LiDER (*On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning*)
+ Additional X-DER datasets (*Class-Incremental Continual Learning into the eXtended DER-verse*)

Pull requests welcome! [Get in touch](mailto:matteo.boschini@unimore.it)


## Previous versions

If you're interested in a version of this repo that only includes the code for [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), please use our [neurips2020 tag](https://github.com/aimagelab/mammoth/releases/tag/neurips2020).
