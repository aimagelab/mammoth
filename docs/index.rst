.. Mammoth documentation master file, created by
   sphinx-quickstart on Tue Dec  5 23:41:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. autosummary::
   :toctree: generated
   :hidden:
   :template: custom-base-template.rst
   :recursive:
   
   models
   datasets
   backbone
   utils


.. toctree::
   :maxdepth: 1 
   :glob:
   :hidden:
   :caption: Getting started:

   getting_started/index.rst
   getting_started/checkpoints.rst
   getting_started/distributed_training.rst
   getting_started/scripts.rst
   Parseval <getting_started/parseval.rst>

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   generated/models.rst
   generated/datasets.rst
   generated/backbone.rst
   generated/utils.rst

.. include:: readme.rst
