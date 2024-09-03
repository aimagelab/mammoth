<p align="center">
  <img width="230" height="230" src="docs/_static/logo.png" alt="logo">
</p>

<p align="center">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/aimagelab/mammoth">
  <a href="https://aimagelab.github.io/mammoth/index.html"><img alt="Static Badge" src="https://img.shields.io/badge/wiki-gray?style=flat&logo=readthedocs&link=https%3A%2F%2Faimagelab.github.io%2Fmammoth%2Findex.html"></a>
  <img alt="Discord" src="https://img.shields.io/discord/1164956257392799860">
</p>

# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

Official repository of:
- [Class-Incremental Continual Learning into the eXtended DER-verse](https://arxiv.org/abs/2201.00766)
- [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
- [Semantic Residual Prompts for Continual Learning](https://arxiv.org/abs/2403.06870)
- [CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning](https://arxiv.org/abs/2407.15793)

Mammoth is a framework for continual learning research. With **more than 40 methods and 20 datasets**, it includes the most complete list competitors and benchmarks for research purposes.

The core idea of Mammoth is that it is designed to be modular, easy to extend, and - most importantly - _easy to debug_.
Ideally, all the code necessary to run the experiments is included _in the repository_, without needing to check out other repositories or install additional packages.

With Mammoth, nothing is set in stone. You can easily add new models, datasets, training strategies, or functionalities.

> ***All the models included in mammoth are verified against the original papers (or subsequent relevant papers) to reproduce their original results.***

## Documentation

### Check out the official [DOCUMENTATION](https://aimagelab.github.io/mammoth/) for more information on how to use Mammoth!

<p align="center">
  <img width="112" height="112" src="docs/_static/seq_mnist.gif" alt="Sequential MNIST">
  <img width="112" height="112" src="docs/_static/seq_cifar10.gif" alt="Sequential CIFAR-10">
  <img width="112" height="112" src="docs/_static/seq_tinyimg.gif" alt="Sequential TinyImagenet">
  <img width="112" height="112" src="docs/_static/perm_mnist.gif" alt="Permuted MNIST">
  <img width="112" height="112" src="docs/_static/rot_mnist.gif" alt="Rotated MNIST">
  <img width="112" height="112" src="docs/_static/mnist360.gif" alt="MNIST-360">
</p>

## Setup

- Install with `pip install -r requirements.txt`. NOTE: Pytorch version >= 2.1.0 is required for scaled_dot_product_attention (see: https://github.com/Lightning-AI/litgpt/issues/763). If you cannot support this requirement, uncomment the lines 136-139 under `scaled_dot_product_attention` in `backbone/vit.py`.
- Use `./utils/main.py` to run experiments.
- New models can be added to the `models/` folder.
- New datasets can be added to the `datasets/` folder.

## Update roadmap

All the code is under active development. Here are some of the features we are working on:

- **Configurations for datasets**: Currently, each dataset represents a *specific configuration* (e.g., number of tasks, data augmentations, backbone, etc.). This makes adding a new *setting* a bit cumbersome. We are working on a more flexible way to define configurations, while leaving the current system as a default for retro-compatibility.
- **New models**: We are working on adding new models to the repository.
- **New training modalities**: We will introduce new CL training regimes, such as training with *noisy labels*, *regression*, *segmentation*, *detection*, etc.
- **Openly accessible result dashboard**: We are working on a dashboard to visualize the results of all the models in both their respective settings (to prove their reproducibility) and in a general setting (to compare them). *This may take some time, since compute is not free.*

All the new additions will try to preserve the current structure of the repository, making it easy to add new functionalities with a simple merge.

## Models

Mammoth currently supports **more than 40** models, with new releases covering the main competitors in literature.

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
- AttriCLIP: `attriclip`.
- Slow Learner with Classifier Alignment (SLCA): `slca`.
- Continual Generative training for Incremental prompt-Learning (CGIL): `cgil`
- Semantic Two-level Additive Residual Prompt (STAR-Prompt): `starprompt`. Also includes the first-stage only (`first_stage_starprompt`) and second-stage only (`second_stage_starprompt`) versions.

## Datasets

**NOTE**: Datasets are automatically downloaded in `data/`.
- This can be changed by changing the `base_path` function in `utils/conf.py` or using the `--base_path` argument.
- The `data/` folder should not be tracked by git and is created automatically if missing.

Mammoth currently includes **21** datasets, covering *toy classification problems* (different versions of MNIST), *standard domains* (CIFAR, Imagenet-R, TinyImagenet, MIT-67), *fine-grained classification domains* (Cars-196, CUB-200), *aerial domains* (EuroSAT-RGB, Resisc45), *medical domains* (CropDisease, ISIC, ChestX).

- Sequential MNIST (_Class-Il / Task-IL_): `seq-mnist`.
- Permuted MNIST (_Domain-IL_): `perm-mnist`.
- Rotated MNIST (_Domain-IL_): `rot-mnist`.
- MNIST-360 (_General Continual Learning_): `mnist-360`.
- Sequential CIFAR-10 (_Class-Il / Task-IL_): `seq-cifar10`.
- Sequential CIFAR-10 resized 224x224 (ViT version) (_Class-Il / Task-IL_): `seq-cifar10-224`.
- Sequential CIFAR-10 resized 224x224 (ResNet50 version) (_Class-Il / Task-IL_): `seq-cifar10-224-rs`.
- Sequential Tiny ImageNet (_Class-Il / Task-IL_): `seq-tinyimg`.
- Sequential Tiny ImageNet resized 32x32 (_Class-Il / Task-IL_): `seq-tinyimg-r`.
- Sequential CIFAR-100 (_Class-Il / Task-IL_): `seq-cifar100`.
- Sequential CIFAR-100 resized 224x224 (ViT version) (_Class-Il / Task-IL_): `seq-cifar100-224`.
- Sequential CIFAR-100 resized 224x224 (ResNet50 version) (_Class-Il / Task-IL_): `seq-cifar100-224-rs`.
- Sequential CUB-200 (_Class-Il / Task-IL_): `seq-cub200`.
- Sequential ImageNet-R (_Class-Il / Task-IL_): `seq-imagenet-r`.
- Sequential Cars-196 (_Class-Il / Task-IL_): `seq-cars196`.
- Sequential RESISC45 (_Class-Il / Task-IL_): `seq-resisc45`.
- Sequential EuroSAT-RGB (_Class-Il / Task-IL_): `seq-eurosat-rgb`.
- Sequential ISIC (_Class-Il / Task-IL_): `seq-isic`.
- Sequential ChestX (_Class-Il / Task-IL_): `seq-chestx`.
- Sequential MIT-67 (_Class-Il / Task-IL_): `seq-mit67`.
- Sequential CropDisease (_Class-Il / Task-IL_): `seq-cropdisease`.

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

Expand to see the BibTex!

<ul>
<li><details><summary>CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning (<b>BMVC 2024</b>) <a href=https://arxiv.org/abs/2407.15793>paper</a></summary>

<pre><code>@inproceedings{heng2022enhancing,
  title={CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning},
  author={Frascaroli, Emanuele and Panariello, Aniello and Buzzega, Pietro and Bonicelli, Lorenzo and Porrello, Angelo and Calderara, Simone},
  booktitle={35th British Machine Vision Conference},
  year={2024}
}</code></pre>

</li>

<li><details><summary>Semantic Residual Prompts for Continual Learning (<b>ECCV 2024</b>) <a href=https://arxiv.org/abs/2403.06870>paper</a></summary>

<pre><code>@inproceedings{menabue2024semantic,
  title={Semantic Residual Prompts for Continual Learning},
  author={Menabue, Martin and Frascaroli, Emanuele and Boschini, Matteo and Sangineto, Enver and Bonicelli, Lorenzo and Porrello, Angelo and Calderara, Simone},
  booktitle={18th European Conference on Computer Vision},
  year={202},
  organization={Springer}
}</code></pre>

</li>

<li><details><summary>Mask and Compress: Efficient Skeleton-based Action Recognition in Continual Learning (<b>ICPR 2024</b>) <a href=https://arxiv.org/pdf/2407.01397>paper</a> <a href=https://github.com/Sperimental3/CHARON>code</a></summary>

<pre><code>@inproceedings{mosconi2024mask,
  title={Mask and Compress: Efficient Skeleton-based Action Recognition in Continual Learning},
  author={Mosconi, Matteo and Sorokin, Andriy and Panariello, Aniello and Porrello, Angelo and Bonato, Jacopo and Cotogni, Marco and Sabetta, Luigi and Calderara, Simone and Cucchiara, Rita},
  booktitle={International Conference on Pattern Recognition},
  year={2024}
}</code></pre>

</li>

<li><details><summary>On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning (<b>NeurIPS 2022</b>) <a href=https://arxiv.org/abs/2210.06443>paper</a> <a href=https://github.com/aimagelab/lider>code</a> (Also available here)</summary>

<pre><code>@article{bonicelli2022effectiveness,
  title={On the effectiveness of lipschitz-driven rehearsal in continual learning},
  author={Bonicelli, Lorenzo and Boschini, Matteo and Porrello, Angelo and Spampinato, Concetto and Calderara, Simone},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31886--31901},
  year={2022}
}</code></pre>

</li>

<li><details><summary>Continual semi-supervised learning through contrastive interpolation consistency (<b>PRL 2022</b>) <a href=https://arxiv.org/abs/2108.06552>paper</a> <a href=https://github.com/aimagelab/CSSL>code</a> (Also available here)</summary>

<pre><code>@article{boschini2022continual,
  title={Continual semi-supervised learning through contrastive interpolation consistency},
  author={Boschini, Matteo and Buzzega, Pietro and Bonicelli, Lorenzo and Porrello, Angelo and Calderara, Simone},
  journal={Pattern Recognition Letters},
  volume={162},
  pages={9--14},
  year={2022},
  publisher={Elsevier}
}</code></pre>

</li>

<li><details><summary>Transfer without Forgetting (<b>ECCV 2022</b>) <a href=https://arxiv.org/abs/2206.00388>paper</a> <a href=https://github.com/mbosc/twf>code</a> (Also available here)</summary>

<pre><code>@inproceedings{boschini2022transfer,
  title={Transfer without forgetting},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Porrello, Angelo and Bellitto, Giovanni and Pennisi, Matteo and Palazzo, Simone and Spampinato, Concetto and Calderara, Simone},
  booktitle={17th European Conference on Computer Vision},
  pages={692--709},
  year={2022},
  organization={Springer}
}</code></pre>

</li>

<li><details><summary>Effects of Auxiliary Knowledge on Continual Learning (<b>ICPR 2022</b>) <a href=https://arxiv.org/abs/2206.02577>paper</a></summary>

<pre><code>@inproceedings{bellitto2022effects,
  title={Effects of auxiliary knowledge on continual learning},
  author={Bellitto, Giovanni and Pennisi, Matteo and Palazzo, Simone and Bonicelli, Lorenzo and Boschini, Matteo and Calderara, Simone},
  booktitle={26th International Conference on Pattern Recognition},
  pages={1357--1363},
  year={2022},
  organization={IEEE}
}</code></pre>

</li>

<li><details><summary>Class-Incremental Continual Learning into the eXtended DER-verse (<b>TPAMI 2022</b>) <a href=https://arxiv.org/abs/2201.00766>paper</a></summary>

<pre><code>@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}</code></pre>

</li>

<li><details><summary>Rethinking Experience Replay: a Bag of Tricks for Continual Learning (<b>ICPR 2020</b>) <a href=https://arxiv.org/abs/2010.05595>paper</a> <a href=https://github.com/hastings24/rethinking_er>code</a></summary>

<pre><code>@inproceedings{buzzega2021rethinking,
  title={Rethinking experience replay: a bag of tricks for continual learning},
  author={Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Calderara, Simone},
  booktitle={25th International Conference on Pattern Recognition},
  pages={2180--2187},
  year={2021},
  organization={IEEE}
}</code></pre>

</li>

<li><details><summary>Dark Experience for General Continual Learning: a Strong, Simple Baseline (<b>NeurIPS 2020</b>) <a href=https://arxiv.org/abs/2004.07211>paper</a></summary>

<pre><code>@inproceedings{buzzega2020dark,
 author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {15920--15930},
 publisher = {Curran Associates, Inc.},
 title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
 volume = {33},
 year = {2020}
}</code></pre>

</details>
</li>
</ul>

### Other Awesome CL works using Mammoth

**_Get in touch if we missed your awesome work!_**

- Gradual Divergence for Seamless Adaptation: A Novel Domain Incremental Learning Method (**ICML 2024**) [[paper](https://arxiv.org/abs/2305.04769)] [[code](https://github.com/NeurAI-Lab/DARE)]
- AGILE - Mitigating Interference in Incremental Learning through Attention-Guided Rehearsal (**CoLLAs 2024**) [[paper](https://arxiv.org/abs/2405.13978)] [[code](https://github.com/NeurAI-Lab/AGILE)]
- Interactive Continual Learning (ICL) (**CVPR 2024**) [[paper](https://arxiv.org/abs/2403.02628)] [[code](https://github.com/Biqing-Qi/Interactive-continual-Learning-Fast-and-Slow-Thinking)]
- Prediction Error-based Classification for Class-Incremental Learning (**ICLR 2024**) [[paper](https://arxiv.org/abs/2305.18806)] [[code](https://github.com/michalzajac-ml/pec)]
- TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion (**NeurIPS 2023**) [[paper](https://arxiv.org/abs/2310.08217)] [[code](https://github.com/NeurAI-Lab/TriRE)]
- Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation (**NeurIPS 2023**) [[paper](https://arxiv.org/abs/2310.08855)] [[code](https://github.com/lvyilin/AdaB2N)]
- A Unified and General Framework for Continual Learning (**ICLR 2024**) [[paper](https://arxiv.org/abs/2403.13249)] [[code](https://github.com/joey-wang123/CL-refresh-learning)]
- Decoupling Learning and Remembering: a Bilevel Memory Framework with Knowledge Projection for Task-Incremental Learning (**CVPR 2023**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Decoupling_Learning_and_Remembering_A_Bilevel_Memory_Framework_With_Knowledge_CVPR_2023_paper.pdf)] [[code](https://github.com/SunWenJu123/BMKP)]
- Regularizing Second-Order Influences for Continual Learning (**CVPR 2023**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Regularizing_Second-Order_Influences_for_Continual_Learning_CVPR_2023_paper.pdf)] [[code](https://github.com/feifeiobama/InfluenceCL)]
- Sparse Coding in a Dual Memory System for Lifelong Learning (**CVPR 2023**) [[paper](https://arxiv.org/abs/2301.05058)] [[code](https://github.com/NeurAI-Lab/SCoMMER)]
- A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm (**CVPR 2023**) [[paper](https://arxiv.org/abs/2310.12244)] [[code](https://github.com/Wang-ML-Lab/unified-continual-learning)]
- A Multi-Head Model for Continual Learning via Out-of-Distribution Replay (**CVPR 2023**) [[paper](https://arxiv.org/abs/2208.09734)] [[code](https://github.com/k-gyuhak/MORE)]
- Preserving Linear Separability in Continual Learning by Backward Feature Projection (**CVPR 2023**) [[paper](https://arxiv.org/abs/2303.14595)] [[code](https://github.com/rvl-lab-utoronto/BFP)]
- Complementary Calibration: Boosting General Continual Learning With Collaborative Distillation and Self-Supervision (**TIP 2023**) [[paper](https://ieeexplore.ieee.org/document/10002397)] [[code](https://github.com/lijincm/CoCa)]
- Continual Learning by Modeling Intra-Class Variation (**TMLR 2023**) [[paper](https://arxiv.org/abs/2210.05398)] [[code](https://github.com/yulonghui/MOCA)]
- ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis (**ICCV 2023**) [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_ConSlide_Asynchronous_Hierarchical_Interaction_Transformer_with_Breakup-Reorganize_Rehearsal_for_Continual_ICCV_2023_paper.pdf)] [[code](https://github.com/HKU-MedAI/ConSlide)]
- CBA: Improving Online Continual Learning via Continual Bias Adaptor (**ICCV 2023**) [[paper](https://arxiv.org/abs/2308.06925)] [[code](https://github.com/wqza/CBA-online-CL)]
- Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal (**ICML 2023**) [[paper](https://arxiv.org/abs/2302.01242)] [[code](https://github.com/ema-marconato/NeSy-CL)]
- Learnability and Algorithm for Continual Learning (**ICML 2023**) [[paper](https://arxiv.org/abs/2306.12646)] [[code](https://github.com/k-gyuhak/CLOOD)]
- Pretrained Language Model in Continual Learning: a Comparative Study (**ICLR 2022**) [[paper](https://openreview.net/pdf?id=figzpGMrdD)] [[code](https://github.com/wutong8023/PLM4CL)]
- Representational continuity for unsupervised continual learning (**ICLR 2022**) [[paper](https://openreview.net/pdf?id=9Hrka5PA7LW)] [[code](https://github.com/divyam3897/UCL)]
- Continual Normalization: Rethinking Batch Normalization for Online Continual Learning (**ICLR 2022**) [[paper](https://arxiv.org/abs/2203.16102)] [[code](https://github.com/phquang/Continual-Normalization)]
- Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System (**ICLR 2022**) [[paper](https://arxiv.org/abs/2201.12604)] [[code](https://github.com/NeurAI-Lab/CLS-ER)]
- New Insights on Reducing Abrupt Representation Change in Online Continual Learning (**ICLR 2022**) [[paper](https://openreview.net/pdf?id=N8MaByOzUfb)] [[code](https://github.com/pclucas14/AML)]
- Looking Back on Learned Experiences for Class/Task Incremental Learning (**ICLR 2022**) [[paper](https://openreview.net/pdf?id=RxplU3vmBx)] [[code](https://github.com/MozhganPourKeshavarz/Cost-Free-Incremental-Learning)]
- Task Agnostic Representation Consolidation: a Self-supervised based Continual Learning Approach (**CoLLAs 2022**) [[paper](https://arxiv.org/abs/2207.06267)] [[code](https://github.com/NeurAI-Lab/TARC)]
- Consistency is the key to further Mitigating Catastrophic Forgetting in Continual Learning (**CoLLAs 2022**) [[paper](https://arxiv.org/abs/2207.04998)] [[code](https://github.com/NeurAI-Lab/ConsistencyCL)]
- Self-supervised models are continual learners (**CVPR 2022**) [[paper](https://arxiv.org/abs/2112.04215)] [[code](https://github.com/DonkeyShot21/cassle)]
- Learning from Students: Online Contrastive Distillation Network for General Continual Learning (**IJCAI 2022**) [[paper](https://www.ijcai.org/proceedings/2022/0446)] [[code](https://github.com/lijincm/OCD-Net)]

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
