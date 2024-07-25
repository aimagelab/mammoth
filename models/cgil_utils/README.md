# How to Run
`cd` into the root directory of mammoth.

It may be required to add the root directory to the python path. if you are using bash, you can do this by running the following command:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/mammoth
```

In the [[paper](https://arxiv.org/abs/2407.15793)] we employ the following hyperparameters for the different datasets accross the seeds `1992`, `1996` and `1997` (we report only the seed `1992` for brevity). The hyperparameters are the same for the other seeds:

- **Imagenet-R**

```bash
python utils/main.py --dataset=seq-imagenet-r --model=cgil --backbone=ViT-L/14 --lr=0.01 --g_models=vae --optim_alignment=adamw --learning_rate_alignment=0.05 --eval_future=1 --seed=1992 --combo_context=1 --gr_vae_n_iters=500 --num_epochs_alignment=60
```

- **Cars-196**

```bash
python utils/main.py --dataset=seq-cars196 --model=cgil --backbone=ViT-L/14 --lr=0.01 --g_models=vae --optim_alignment=adamw --learning_rate_alignment=0.03 --eval_future=1 --seed=1992 --combo_context=1 --gr_vae_n_iters=500 --num_epochs_alignment=60
```

- **CUB-200**

```bash
python utils/main.py --dataset=seq-cub200 --model=cgil --backbone=ViT-L/14 --lr=0.01 --g_models=vae --optim_alignment=adamw --learning_rate_alignment=0.01 --eval_future=1 --seed=1992 --combo_context=1 --gr_vae_n_iters=500 --num_epochs_alignment=60
```

- **EuroSAT-RGB**

```bash
python utils/main.py --dataset=seq-eurosat-rgb --model=cgil --backbone=ViT-L/14 --lr=0.01 --g_models=vae --optim_alignment=adamw --learning_rate_alignment=0.03 --eval_future=1 --seed=1992 --combo_context=1 --gr_vae_n_iters=500 --num_epochs_alignment=150
```

- **ISIC**

```bash
python utils/main.py --dataset=seq-isic --model=cgil --backbone=ViT-L/14 --lr=0.01 --g_models=vae --optim_alignment=adamw --learning_rate_alignment=0.05 --eval_future=1 --seed=1992 --combo_context=1 --gr_vae_n_iters=750 --num_epochs_alignment=150
```
