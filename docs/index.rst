.. Mammoth documentation master file, created by
   sphinx-quickstart on Tue Dec  5 23:41:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. autosummary::
   :toctree: _autosummary
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
   getting_started/scripts.rst

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   _autosummary/models.rst
   _autosummary/datasets.rst
   _autosummary/backbone.rst
   _autosummary/utils.rst

.. include:: readme.rst
