.. _model_arguments_docs:

Define and handle the model arguments
======================================

The `sgd` baseline we defined in the previous chapter is quite simple and does not require any additional arguments. However, most models require additional arguments to be defined. In this chapter, we will show how to define and handle the model arguments in Mammoth.

The model arguments can be defined by adding the static method `get_parser` to the model class. This method takes the main instance of the `argparse.ArgumentParser` class as input and returns the instance with the desired arguments. The following example shows how to define some arguments for the `sgd` model:

.. code-block:: python

    from models.utils.continual_model import ContinualModel

    class Sgd(ContinualModel):
        NAME = 'sgd'
        COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

        @staticmethod
        def get_parser(parser): # Define the get_parser method
            # Add the desired arguments to the parser
            parser.add_argument('--gradient_clipping_norm', type=float, default=0.0, help='The norm value for gradient clipping (0.0 to disable)')
            return parser

        def __init__(self, backbone, loss, args, transform, dataset=None):  # Define the constructor of the model
            super(Sgd, self).__init__(backbone, loss, args, transform, dataset=dataset)

        def observe(self, inputs, labels, not_aug_inputs, epoch=None): # Define the observe method
                        self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()

            # Perform gradient clipping if the norm value is greater than 0.0
            if self.args.gradient_clipping_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.gradient_clipping_norm)
            
            self.opt.step()
            return loss.item()

In the example above, we added the `gradient_clipping_norm` argument to the `sgd` model. This argument is used to perform gradient clipping during training. 

External arguments and default values
--------------------------------------

Mammoth already defines some common arguments that can be used by all models, such as `lr`, `batch_size`, and `n_epochs`. These arguments are defined in the `utils/args.py` file. 

The default values of the arguments can be defined in the `get_parser` method by using the `set_defaults` method of the `argparse.ArgumentParser` class. The following example shows how to define the default values for the `lr` and `batch_size` arguments:

.. code-block:: python

    # ... (code)

    @staticmethod
    def get_parser(parser): # Define the get_parser method
        # Add the desired arguments to the parser
        parser.set_defaults(lr=0.1, batch_size=128) # Define the default values for the lr and batch_size arguments
        parser.add_argument('--gradient_clipping_norm', type=float, default=0.0, help='The norm value for gradient clipping (0.0 to disable)')
        return parser

    # ... (code)



.. _model-configurations:

Model configurations and best arguments
----------------------------------------

In order to facilitate the use and reproducibility of the models, we introduced the concept of *configurations* for the models. The configurations are stored as a separate file named `<model-name>.yaml` in the `models/configs` directory. 

The configuration file store a **default** configuration (indepenent of the dataset) and a **best** configuration. The best configuration depends on the dataset and, if available, the buffer size:

- **default**: the default configuration for the model, which does *NOT* depend on the dataset or buffer size. This configuration is used if the ``--model_config`` argument is set to ``default`` (or ``base``). This is the default behaviour. A similar effect of setting the ``--model_config`` argument to ``default`` can be achieved by setting the default values in the **get_parser** method, using the **set_defaults** method of the `argparse.ArgumentParser <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_ object. The default values set in the **get_parser** method are always loaded, but can be overridden by the CLI arguments.

- **best**: the best configuration for the model for a particular dataset (and buffer size, if applicable). This configuration is used if the ``--model_config`` argument is set to ``best``.

Each configuration is defined in a file named **<model_name>.yaml** and placed in the **models/configs** folder. The configuration file is a `YAML <https://yaml.org/>`_ file that defines the hyper-parameters of the model. The hyper-parameters are defined as a dictionary with the hyper-parameter name as the key and the hyper-parameter value as the value. All hyper-parameters defined under the key ``default`` are loaded with the ``default`` configuration, while only the hyper-parameters defined at under the dataset name (and buffer size, if applicable) are loaded with the ``best`` configuration. For example, the following configuration file for **my_model** defines a default `optimizer` for the model, a `learning_rate` when trained on the **seq-cifar100** dataset, and a `optim_wd` when the buffer size is **100**:

.. code-block:: yaml

    default:
        optimizer: adam # this optimizer is set to 'adam' by default (i.e., is ALWAYS loaded)
    seq-cifar100: # all the hyper-parameters defined under 'seq-cifar100' are loaded only if the dataset is 'seq-cifar100'
        learning_rate: 0.001
        100: # all the hyper-parameters defined under '100' are loaded only if the buffer size is '100'
            optim_wd: 1e-5

Once defined, to load the best configuration for a model, you can use the `--model_config=best` argument in the training script. For example, to train the `derpp` (DER++) model with the best configuration on the `seq-cifar100` dataset and a buffer size of ``500`` elements, you can run the following command:

.. code-block:: bash

    python main.py --model derpp --dataset seq-cifar100 --buffer_size 500 --model_config best

This will load the best configuration to reproduce the results reported in the paper. This includes loading the exact configuration of the dataset, epochs, learning rate, and other hyperparameters.

`You can check out this page <reproduce_mammoth>` more information on the reproducibility of the models in Mammoth.