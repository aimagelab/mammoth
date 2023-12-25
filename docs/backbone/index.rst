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