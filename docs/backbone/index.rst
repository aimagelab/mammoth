.. _module-backbones:

Backbones
=========

Backbones constitute the underlying architecture of the network. They are responsible for extracting features from the input.
Once loaded by the model (:ref:`module-models`), the backbone is accessible as the ``self.net`` attribute.

The specific choice of the backbone depends on the benchmark (i.e, the dataset - see :ref:`module-datasets`), which defines the backbone from the **get_backbone** method.

Features and logits
-------------------

Models may exploit the features extracted by the backbone at different levels. For example, **DER** uses the soft targets produced after the classification layer of the backbone (i.e., the logits), while **iCaRL** uses the features extracted by the backbone before the classification layer to compute the class means. In order to allow for this flexibility, all backbones **must** accept the ``returnt`` argument, which specifies the level at which the features are extracted. The possible values are:

- ``returnt='out'``: the backbone returns the logits produced *after* the classification layer.

- ``returnt='features'``: the backbone returns the features extracted immediately *before* the classification layer.

- ``returnt='both'``: the backbone returns both the logits and the features (a tuple ``(logits, feature)``).

In addition, some models require the output of *all* the layers of the backbone (e.g, **TwF**). In this case, the ``returnt`` argument can be set to:

- ``returnt='full'``: the backbone returns the output of all the layers (a list of tensors).

.. note::

    Other values of ``returnt`` may be supported by the backbone, but they are not guaranteed to work with all the models.


.. _backbone-registration:

Backbone registration and selection
-----------------------------------

To be used in Mammoth, backbones must be registered with the `register_backbone` decorator. The decorator can be applied either to subclasses of the **MammothBackbone** class (see below) or to functions that return the backbone. The decorator takes a single argument, the name of the backbone, which will be used to select the backbone from the command line (with ``--backbone``) or from the dataset configuration file.

Since each backbone requires different arguments, the decorator will automatically try to infer the arguments from the *signature* of the function (or the `__init__` method if applied to a class). For each argument in the signature, its default value will be used as the default value during parsing. If the default is not set, the argument is *required*; otherwise, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

For example, the following code registers a backbone with a single required argument (``depth``) and an optional argument (``width``) with default value 1:

.. code-block:: python

    @register_backbone('resnet')
    def resnet(depth: int, width: int = 1):
        return ResNet(depth, width)

.. important::

    The name of the arguments should not overlap with other arguments, but must not be unique across backbones. Only the arguments of the backbone selected by the user will be processed and available in the command line. If the same argument is specified externally, the backbone one will be **ignored**.

Mammoth backbone base class
---------------------------

Backbones should inherit from the **MammothBackbone** class (below), which provides some useful methods.
This is not a strict requirement, but it is strongly recommended for compatibility with the existing models.

The **MammothBackbone** class provides the following methods:

- **features**: returns the features extracted by the backbone (before the classification layer). This is equivalent to calling ``self.net(x, returnt='features')``.

- **get_grads**: returns the gradients of the backbone with respect to the loss, concatenated in a single tensor.

- **set_grads**: sets the gradients of the backbone from a single (concatenated) tensor.

- **get_params**: returns all the parameters of the backbone concatenated in a single tensor. 

- **set_params**: sets the parameters of the backbone from a single (concatenated) tensor.