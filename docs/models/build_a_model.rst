.. _build_a_model:

How to build a model in Mammoth
===============================

In Mammoth, a model is defined as a class that inherits from the base class :ref:`ContinualModel <module-models.utils.continual_model>`. This class defines a few special methods that can be implemented:

1. **observe** (*mandatory*): This method is called at each training iteration and is used to update the model parameters according to the current training batch. The method must have the following signature:

    .. code-block:: python

        def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                    not_aug_inputs: torch.Tensor, epoch: int = None) -> float|dict:

            # Update the model parameters according to the current batch
            ...

            # Return the current loss value (as a float value)
            # or a dictionary of elements with at least a 'loss' key, all the other keys will be logged with wandb (if enabled)
            return loss.item()

    The method receives as input the current training batch (i.e., **inputs** and **labels**), the original batch (i.e., **not_aug_inputs**) and (*optionally*) the current training epoch (i.e., **epoch**). Additional arguments can be defined and will be passed to the method if supported by the training dataset. For example, methods that support the *noisy label* setting can receive the true labels as an additional argument (for logging and debug). This can be done by defining the method signature as follows:

    .. code-block:: python

        def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                    not_aug_inputs: torch.Tensor, epoch: int = None, true_labels: torch.Tensor = None) -> float|dict:
    
    The observe method can return either the current loss value (as a float value) or a dictionary of elements with at least a 'loss' key. All the other keys will be logged with wandb (if enabled).

2. **forward** (*optional*): This method is used to evaluate the model on the test set. By default, it is implemented in the base class:
        
        .. code-block:: python
    
            def forward(self, x: torch.Tensor) -> torch.Tensor:
    
                # Compute the output of the model
                ...
    
                # Return the output of the model
                return output
    
        The method receives as input the current input batch (i.e., **x**) and must return the output of the model.


Basic example - the `sgd` model
--------------------------------

The following exaplme shows how to build the simple `sgd` baseline model (also referred to as `finetuning` in some papers). This model does not perform any continual learning strategy and is used as a baseline for comparison. A full example of the `sgd` model can be found in the `models/sgd.py` file.

.. code-block:: python

    from models.utils.continual_model import ContinualModel # Import the base class

    class Sgd(ContinualModel): # Define the model class
        NAME = 'sgd' # Define the name of the model
        COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual'] # Define the compatibility of the model with the different continual learning scenarios

        def __init__(self, backbone, loss, args, transform, dataset=None):  # Define the constructor of the model
            super(Sgd, self).__init__(backbone, loss, args, transform, dataset=dataset)

        def observe(self, inputs, labels, not_aug_inputs, epoch=None): # Define the observe method
            
            # Update the model parameters according to the current batch
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()

            # Return the current loss value for logging
            return loss.item()

The class defined above is quite simple and only implements the `observe` method. The `forward` method is not implemented because it is already defined in the base class. The `observe` method updates the model parameters according to the current training batch and returns the current loss value for logging. 

In order to be picked up by the framework, the model class must be defined in the `models` directory and the file must be named as the model class (e.g., `sgd.py` for the `Sgd` class).

That's it! You have now built a simple model in Mammoth. You can now use this model by specifying its name with the `--model` argument in the training script. For example, to train the `sgd` model on the `seq-cifar100` dataset, you can run the following command:

.. code-block:: bash

    python main.py --model sgd --dataset seq-cifar100 --lr 0.1

In the `next chapter <model_arguments_docs>`, we will show how to define and handle the model arguments.