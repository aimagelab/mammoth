# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)

project = 'Mammoth'
copyright = '2023, Pietro Buzzega, Matteo Boschini, Lorenzo Bonicelli, Aniello Panariello, Davide Abati, Angelo Porrello, Simone Calderara'
author = 'Pietro Buzzega, Matteo Boschini, Lorenzo Bonicelli, Aniello Panariello, Davide Abati, Angelo Porrello, Simone Calderara'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pytorch_sphinx_theme"
html_static_path = ['_static']
