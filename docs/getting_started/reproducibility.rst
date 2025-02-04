.. _reproduce_mammoth:

On the reproducibility of Mammoth
=================================

We take great pride and care in the reproducibility of the models in Mammoth. To ensure reproducibility, we follow the following guidelines:

1. We follow (if available) the official implementation of the models. If the official implementation is not available, we implement the model from scratch and compare the results with the official implementation.

2. We store the best hyperparameters for each model in the `models/configs` directory. The best hyperparameters are stored in a separate file named `<model-name>.yaml`. The best hyperparameters are determined either by the authors of the model or by us based on the performance on a designed dataset.

3. The performance of each model is evaluated on the same dataset used in the paper and we report in `REPRODUCIBILITY.md` our results compared to the results reported in the paper. We also provide the exact command used to train the model (most times, it follows `python main.py --model <model-name> --dataset <dataset-name> --model_config best`).

We encourage the community to report any issues with the reproducibility of the models in Mammoth. If you find any issues, please open an issue in the GitHub repository or contact us directly.

.. note::

    Since there are many models in Mammoth (and some of them predate PyTorch), the process of filling the `REPRODUCIBILITY.md` file is ongoing. We are working hard to fill the file with the results of all models in Mammoth. 

.. important::

    *Does this mean that the models that are not in the `REPRODUCIBILITY.md` file do not reproduce?*
    
    No! It means that we have not yet found the appropriate dataset and hyperparameters to fill the file with the results of that model. We are working hard to fill the file with the results of all models in Mammoth. If you need the results of a specific model, please open an issue in the GitHub repository or contact us directly.
