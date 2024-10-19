.. _dataset-schedulers-docs:

Schedulers for datasets
=======================

Mammoth supports the use of learning rate schedulers to adjust the learning rate during training. The learning rate scheduler can be specified using the ``--scheduler`` argument in the training script. 

The use of schedulers depends on the chosen dataset and only the `MultiStepLR` scheduler is supported by default. Additional schedulers can be added by:

1. Extending the `AVAIL_SCHEDS` variable in `datasets/utils/continual_dataset.py` with the desired scheduler name.

2. Adding the scheduler-specific arguments to `utils/args.py`.

If you follow this convention, the scheduler will be automatically loaded and handled by the training script. However, you can always define your own scheduler in the model class. Just make sure to reload the optimizer and scheduler at the end of each task, or you might loose many hours in debugging (*as I did...)*.

