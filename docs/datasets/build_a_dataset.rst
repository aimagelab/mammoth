.. _build_a_dataset:

How to build a dataset in Mammoth
===============================

In Mammoth, a dataset is a class that inherits from :ref:`Continual Dataset <module-datasets.utils.continual_dataset>` and defines all the hyper-parameters that define a complete Continual Learning scenario. The only exception are datasets that follow the `general-continual` setting, which inherit from the **GCLDataset** class, (defined in :ref:`GCL Dataset <module-datasets.utils.gcl_dataset>`). 

To build a dataset, you need to define the following methods:

    - **get_data_loaders** (returns ``[DataLoader, DataLoader]``): this function is tasked with creating and returning the train and test data loaders for the task at hand. The function should return a list of two data loaders, the first one for the training data and the second one for the test data. To split the data for the current task, you can use the ``store_masked_loaders`` function. For more information, see :ref:`Splitting the data <split_datasets>`.

    .. important::

        The train dataset **must** return both augmented and non-augmented data. This is done to allow the storage of raw data for replay-based methods. The signature return for the train data loader is ``(augmented_data, labels, non_augmented_data)``, while the test data loader should return ``(data, labels)``.

    - **get_backbone** (returns ``str``): returns the *name* of backbone model to be used for the experiment. Backbones are defined in `backbones` folder and can be registered with the `register_backbone` decorator. See more in :ref:`backbones`.

    - **get_transform** (returns ``callable``): a static method that returns the data-augmentation transform to apply to the data during train.

    - **get_loss** (returns ``callable``): a static method that returns the loss function to use during train.

    - **get_normalization_transform** (returns ``callable``): a static method that returns the normalization transform to apply *on torch tensors* (no `ToTensor()` required).

    - **get_denormalization_transform** (returns ``callable``): a static method that returns the transform to apply on the tensors to revert the normalization. You can use the `DeNormalize` function defined in `datasets/transforms/denormalization.py`.

Some additional methods can be defined as a quality of life improvement. They are not strictly necessary, but can be useful if you want to avoid specifying them in the CLI all the time:

    - **get_batch_size** (returns ``int``): returns the batch size for each task.

    - **get_epochs** (returns ``int``): returns the number of epoch for each task. This method should not be changed **only** for datasets that follow the `general-continual` setting (as it is single-epoch by definition).

.. _split_datasets:

Splitting the data
~~~~~~~~~~~~~~~~~~~

Before returning the data loaders, you need to split the data for the current task. You can do this in two ways:

- **store_masked_loaders**: this function is the easiest way to split the data. It takes care of masking the data loaders to return only the data for the current task and is used by most datasets to create the data loaders for each task. Besides splitting the data, this function also applies the following transformations:

    - If the ``--permute_classes`` flag is set to ``1``, it also applies the appropriate permutation to the classes before splitting the data.

    - If the ``--label_perc_by_task/--label_perc_by_class`` argument is set to a value between ``0`` and ``1``, it also randomly masks a percentage of the labels for each task/class (**Continual Semi-Supervised Learning** scenario)

    - If the ``--noise_rate`` argument is set to a value between ``0`` and ``1``, it also adds noise to the labels for each task/class (**Continual Learning under Noisy Labels** scenario). This option is available only if the task is a multi-class single-label classification task (*i.e.*, the `get_loss()` method returns `F.cross_entropy`).

- *custom*: you can split data manually in the ``get_data_loaders`` method. This may be used for more complex scenarios (e.g., scenarios with blurry task boundaries) or if your scenario is composed of multiple datasets that need to be combined in a specific way.

Required attributes
~~~~~~~~~~~~~~~~~~~

To define a dataset, you need to define the following attributes:

    - Name of the dataset: **NAME** attribute (``str``). This will be used to select the dataset from the command line with the ``--dataset`` argument.

    - Incremental setting (`class-il`, `domain-il`, or `general-continual`): **SETTING** attribute (``str``). See more in section :ref:`datasets-settings`.

    - Size of the input data: **SIZE** attribute (``tuple[int]``).

Some additional attributes are necessary for the `class-il` and `domain-il` settings:

    - Number of tasks: **TASKS** attribute (``int``).

    - Number of classes per task: **N_CLASSES_PER_TASK** attribute (``int|tuple[int]``). This can be a list of integers (one for each task and only for `class-il` setting), or a single integer.

Putting it all together with an example
----------------------------------------

Let's see an example of how to define a dataset. In this case, we will define a dataset that follows the `class-il` setting with 5 tasks and 2 classes per task. The dataset will be called `SequentialCIFAR10` and will have a size of 32x32. The dataset will use the CIFAR-10 dataset and will apply only the standard normalization.

.. code-block:: python

    # ... imports

    class SequentialCIFAR10(ContinualDataset):
        NAME = 'seq-cifar10' # Name of the dataset
        SETTING = 'class-il' # Class-Incremental setting
        N_CLASSES_PER_TASK = 2 # 2 classes per task
        N_TASKS = 5 # 5 tasks in total
        N_CLASSES = N_CLASSES_PER_TASK * N_TASKS # Total number of classes in the dataset. This is automatically calculated by the framework if not provided.
        SIZE = (32, 32) # Size of the input data
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615) # Mean and standard deviation of the dataset
        TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]) # Data augmentation transform

        def get_data_loaders(self): 
            
            # TrainCIFAR10 is a custom dataset that returns both augmented and non-augmented data 
            train_dataset = TrainCIFAR10(base_path() + 'CIFAR10', train=True, 
                                    download=True, transform=self.TRANSFORM)
            # for the test dataset, we use the standard CIFAR-10 dataset, as we don't need to non-augmented data
            test_dataset = CIFAR10(base_path() + 'CIFAR10', train=False,
                                    download=True, transform=self.TEST_TRANSFORM)
            
            # Split the data for the current task and return the data loaders
            return store_masked_loaders(train_dataset, test_dataset, self)
            
        @staticmethod
        def get_transform(): # this should include the ToPILImage() transform, as it will be applied on tensors
            return transforms.Compose([transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
            
        @set_default_from_args("backbone")
        def get_backbone(): # the name of the backbone model to use
            return "resnet18"

        @staticmethod
        def get_loss(): # the loss function 
            return F.cross_entropy

        @staticmethod
        def get_normalization_transform():
            return transforms.Normalize(SequentialCIFAR10.MEAN, SequentialCIFAR10.STD)

        @staticmethod
        def get_denormalization_transform():
            return DeNormalize(SequentialCIFAR10.MEAN, SequentialCIFAR10.STD)

In the code above, we define a dataset called `SequentialCIFAR10` that follows the `class-il` setting. The most important method is the `get_data_loaders` method, which returns the train and test data loaders for the current task. Since the train data loader must return both augmented and non-augmented data, we define a custom dataset called `TrainCIFAR10` that returns both augmented and non-augmented data. We can define it as follows:

.. code-block:: python

    from PIL import Image
    from torchvision.datasets import CIFAR10

    class TrainCIFAR10(CIFAR10):
        def __init__(self, root, train=True, transform=None, download=False):
            super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())
            
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img, mode='RGB')
            return self.transform(img), target, transforms.ToTensor()(img)

The test data loader uses the standard CIFAR-10 dataset, as we don't need non-augmented data for the test data loader.

.. _dataset-naming-convention:

Naming convention and automatical dataset registration
------------------------------------------------------

The following steps are required to create a dataset following the *legacy* naming convention. A new and more flexible way to define datasets is available with the **register_dataset** decorator. See more in :ref:`module-dynamic-registration`.


These classes provide some useful methods to create data loaders and store masked data loaders for continual learning experiments. See more in the next section.

    1. Create a new file in the `datasets` folder, e.g. ``my_dataset.py``.

    2. Define a *SINGLE* new class that inherits from `ContinualDataset` or `GCLDataset` and implements all the required methods and attributes.

    3. Define the **get_data_loaders** method, which returns a list of train and test data loaders for each task. 

    .. tip::
        For convenience, most datasets are initially created with all classes and then masked appropriately by the **store_masked_loaders** function. 
        For example, in :ref:`Seq CIFAR-10 <module-datasets.seq_cifar10>` the **get_data_loaders** function of **SequentialCIFAR10** dataset first inizializes the **MyCIFAR10** and **TCIFAR10** 
        datasets with train and test data for all classes respectively, and then masks the data loaders to return only the data for the current task.

    .. important::
        The train data loader **must** return both augmented and non-augmented data. This is done to allow the storage of raw data for replay-based methods 
        (for more information, check out `Rethinking Experience Replay: a Bag of Tricks for Continual Learning <https://arxiv.org/abs/2010.05595>`_).
        The signature return for the train data loader is ``(augmented_data, labels, non_augmented_data)``, while the test data loader should return ``(data, labels)``.

    4. If all goes well, your dataset should be picked up by the **get_dataset** function and you should be able to run an experiment with it.

Additional methods for the dataset
----------------------------------

Some additional methods can be defined for the dataset to provide additional functionalities. These methods are necessary for some methods (e.g., `clip`) and can be useful for others:

    - **get_prompt_templates** (``callable``): returns the prompt templates for the dataset. This method is expected for some methods (e.g., `clip`). *By default*, it returns the ImageNet prompt templates.

    - **get_class_names** (``callable``): returns the class names for the dataset. This method is not implemented by default, but is expected for some methods (e.g., `clip`). The method *should* populate the **class_names** attribute of the dataset to cache the result and call the ``fix_class_names_order`` method to ensure that the class names are in the correct order.
