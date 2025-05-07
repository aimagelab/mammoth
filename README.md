<p align="center">
  <!-- <img width="230" height="230" src="docs/_static/logo.png" alt="logo"> -->
  <img width="1000" height="200" src="docs/_static/mammoth_banner.svg" alt="logo">
</p>

<p align="center">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/aimagelab/mammoth">
  <a href="https://aimagelab.github.io/mammoth/index.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-mammoth-blue?style=flat&logo=readthedocs"></a>
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/aimagelab/mammoth?style=social">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white">
</p>

# 🦣 Mammoth - A PyTorch Framework for Benchmarking Continual Learning

_Mammoth_ is built to streamline the development and benchmark of continual learning research. With **more than 60 methods and 20 datasets**, it includes the most complete list competitors and benchmarks for research purposes.

The core idea of Mammoth is that it is designed to be modular, easy to extend, and - most importantly - _easy to debug_.

With Mammoth, nothing is set in stone. You can easily add new models, datasets, training strategies, or functionalities.

## 📚 Documentation

<p align="center">
  <a href="https://aimagelab.github.io/mammoth/">
    <em style="display: inline-block; margin-top: 8px; font-size: 16px; color: #4B73C9; background-color: #f8f9fa; padding: 8px 16px; border-radius: 0 0 8px 8px; border: 1px solid #4B73C9; border-top: none; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">Check out our guides on using Mammoth for continual learning research</em>
    <br/>
    <img src="https://img.shields.io/badge/Documentation-📚-4B73C9?style=for-the-badge&logo=gitbook&logoColor=white" alt="Documentation" height="40">
  </a>
</p>

## ⚙️ Setup

- 📥 Install with `pip install -r requirements.txt` or run it directly with `uv run python main.py ...`
  > **Note**: PyTorch version >= 2.1.0 is required for scaled_dot_product_attention. If you cannot support this requirement, uncomment the lines 136-139 under `scaled_dot_product_attention` in `backbone/vit.py`.
- 🚀 Use `main.py` or `./utils/main.py` to run experiments.
- 🧩 New models can be added to the `models/` folder.
- 📊 New datasets can be added to the `datasets/` folder.

## 🧪 Examples

### Run a model

The following command will run the model `derpp` on the dataset `seq-cifar100` with a buffer of 500 samples the some random hyperparameters for _lr_, _alpha_, and _beta_:
```bash
python main.py --model derpp --dataset seq-cifar100 --alpha 0.5 --beta 0.5 --lr 0.001 --buffer_size 500
```

To run the model with the best hyperparameters, use the `--model_config=best` argument:
```bash
python main.py --model derpp --dataset seq-cifar100 --model_config best
```

 > NOTE: the `--model_config` argument will look for a file `<model_name>.yaml` in the `models/configs/` folder. This file should contain the hyperparameters for the best configuration of the model. You can find more information in [the documentation](https://aimagelab.github.io/mammoth/models/model_arguments.html#model-configurations-and-best-arguments).

### Build a new model

See the [documentation](https://aimagelab.github.io/mammoth/models/build_a_model.html) for a detailed guide on how to create a new model.

### Build a new dataset

See the [documentation](https://aimagelab.github.io/mammoth/datasets/build_a_dataset.html) for a detailed guide on how to create a new dataset.


## 🗺️ Update Roadmap

All the code is under active development. Here are some of the features we are working on:

- 🧠 **New models**: We are continuously working on adding new models to the repository.
- 🔄 **New training modalities**: New training regimes, such a *regression*, *segmentation*, *detection*, etc.
- 📊 **Openly accessible result dashboard**: The ideal would be a dashboard to visualize the results of all the models in both their respective settings (to prove their reproducibility) and in a general setting (to compare them). *This may take some time, since compute is not free.*

All the new additions will try to preserve the current structure of the repository, making it easy to add new functionalities with a simple merge.

## 🧠 Models

Mammoth currently supports **more than 60** models, with new releases covering the main competitors in literature.

<details>
<summary><b>Click to expand model list</b></summary>

- AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning (AttriCLIP): `attriclip`.
- Bias Correction (BiC): `bic`.
- CaSpeR-IL (on DER++, X-DER with RPC, iCaRL, and ER-ACE): `derpp_casper`, `xder_rpc_casper`, `icarl_casper`, `er_ace_casper`.
- CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning (CODA-Prompt) - _Requires_ `pip install timm==0.9.8`: `coda-prompt`.
- Continual Contrastive Interpolation Consistency (CCIC) - _Requires_ `pip install kornia`: `ccic`.
- Continual Generative training for Incremental prompt-Learning (CGIL): `cgil`
- Contrastive Language-Image Pre-Training (CLIP): `clip` (*static* method with no learning).
- CSCCT (on DER++, X-DER with RPC, iCaRL, and ER-ACE): `derpp_cscct`, `xder_rpc_cscct`, `icarl_cscct`, `er_ace_cscct`.
- Dark Experience for General Continual Learning: a Strong, Simple Baseline (DER & DER++): `der` and `derpp`.
- DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning (DualPrompt) - _Requires_ `pip install timm==0.9.8`: `dualprompt`.
- Efficient Lifelong Learning with A-GEM (A-GEM, A-GEM-R - A-GEM with reservoir buffer): `agem`, `agem_r`.
- Experience Replay (ER): `er`.
- Experience Replay with Asymmetric Cross-Entropy (ER-ACE): `er_ace`.
- eXtended-DER (X-DER): `xder` (full version), `xder_ce` (X-DER with CE), `xder_rpc` (X-DER with RPC).
- Function Distance Regularization (FDR): `fdr`.
- Generating Instance-level Prompts for Rehearsal-free Continual Learning (DAP): `dap`.
- Gradient Episodic Memory (GEM) - _Unavailable on windows_: `gem`.
- Greedy gradient-based Sample Selection (GSS): `gss`.
- Greedy Sampler and Dumb Learner (GDumb): `gdumb`.
- Hindsight Anchor Learning (HAL): `hal`.
- Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS (IDEFICS): `idefics` (*static* method with no learning).
- Incremental Classifier and Representation Learning (iCaRL): `icarl`.
- Joint training for the General Continual setting: `joint_gcl` (_only for General Continual_).
- Large Language and Vision Assistant (LLAVA): `llava` (*static* method with no learning).
- Learning a Unified Classifier Incrementally via Rebalancing (LUCIR): `lucir`.
- Learning to Prompt (L2P) - _Requires_ `pip install timm==0.9.8`: `l2p`.
- Learning without Forgetting (LwF): `lwf`.
- Learning without Forgetting adapted for Multi-Class classification (LwF.MC): `lwf_mc` (from the iCaRL paper).
- Learning without Shortcuts (LwS): `lws`.
- LiDER (on DER++, iCaRL, GDumb, and ER-ACE): `derpp_lider`, `icarl_lider`, `gdumb_lider`, `er_ace_lider`.
- May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels (AER & ABS): `er_ace_aer_abs`.
- Meta-Experience Replay (MER): `mer`.
- Mixture-of-Experts Adapters (MoE Adapters): `moe_adapters`.
- Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries (PuriDivER): `puridiver`.
- online Elastic Weight Consolidation (oEWC): `ewc_on`.
- Progressive Neural Networks (PNN): `pnn`.
- Random Projections and Pre-trained Models for Continual Learning (RanPAC): `ranpac`.
- Regular Polytope Classifier (RPC): `rpc`.
- Rethinking Experience Replay: a Bag of Tricks for Continual Learning (ER-ACE with tricks): `er_ace_tricks`.
- Semantic Two-level Additive Residual Prompt (STAR-Prompt): `starprompt`. Also includes the first-stage only (`first_stage_starprompt`) and second-stage only (`second_stage_starprompt`) versions.
- SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model (SLCA) - _Requires_ `pip install timm==0.9.8`: `slca`.
- Slow Learner with Classifier Alignment (SLCA): `slca`.
- Synaptic Intelligence (SI): `si`.
- Transfer without Forgetting (TwF): `twf`.
- ZSCL: Zero-Shot Continual Learning: `zscl`.
</details>

## 📊 Datasets

**NOTE**: Datasets are automatically downloaded in `data/`.  
- This can be changed by changing the `base_path` function in `utils/conf.py` or using the `--base_path` argument.  
- The `data/` folder should not be tracked by _git_ and is created automatically if missing.

<details>
<summary><b>Click to expand dataset list</b></summary>

Mammoth currently includes **23** datasets, covering *toy classification problems* (different versions of MNIST), *standard natural-image domains* (CIFAR, Imagenet-R, TinyImagenet, MIT-67), *fine-grained classification domains* (Cars-196, CUB-200), *aerial domains* (EuroSAT-RGB, Resisc45), *medical domains* (CropDisease, ISIC, ChestX).

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
- Sequential CelebA (_Biased-Class-Il_): `seq-celeba`. *This dataset is multi-label (i.e., trains with binary cross-entropy)*
</details>

## 📝 Citing the library

```bibtex
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

## 🔬 On the reproducibility of Mammoth
We take great pride and care in the reproducibility of the models in Mammoth and we are commited to provide the community with the most accurate results possible. To this end, we provide a _REPRODUCIBILITY.md_ file in the repository that contains the results of the models in Mammoth.

The performance of each model is evaluated on the same dataset used in the paper and we report in _REPRODUCIBILITY.md_ the list of models that have been verified. We also provide the exact command used to train the model (most times, it follows `python main.py --model <model-name> --dataset <dataset-name> --model_config best`).

We encourage the community to report any issues with the reproducibility of the models in Mammoth. If you find any issues, please open an issue in the GitHub repository or contact us directly.

**Disclaimer**: Since there are many models in Mammoth (and some of them predate PyTorch), the process of filling the _REPRODUCIBILITY.md_ file is ongoing. We are working hard to fill the file with the results of all models in Mammoth. If you need the results of a specific model, please open an issue in the GitHub repository or contact us directly.

> Does this mean that the models that are not in the _REPRODUCIBILITY.md_ file do not reproduce?

No! It means that we have not yet found the appropriate dataset and hyperparameters to fill the file with the results of that model. We are working hard to fill the file with the results of all models in Mammoth. If you need the results of a specific model, please open an issue in the GitHub repository or contact us directly.

## 🤝 Contributing
Pull requests are welcome!

<a href="https://github.com/aimagelab/mammoth/graphs/contributors"> <img src="https://contrib.rocks/image?repo=aimagelab/mammoth" /> </a>

Please use autopep8 with parameters:

```
--aggressive
--max-line-length=200
--ignore=E402
```

## Previous versions

If you're interested in a version of this repo that only includes the original code for _"Dark Experience for General Continual Learning: a Strong, Simple Baseline"_ or _"Class-Incremental Continual Learning into the eXtended DER-verse"_, please use the following tags:

`neurips2020` for DER (NeurIPS 2020).  
`tpami2023` for X-DER (TPAMI 2022).