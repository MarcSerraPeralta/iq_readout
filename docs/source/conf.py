# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Autodoc configuration ---------------------------------------------------
# Autogenerate documentation from the comments of the code
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'iq_readout'
copyright = '2024, Marc Serra Peralta'
author = 'Marc Serra Peralta'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc', # autogenerate documentation
        'sphinx.ext.viewcode', # adds link to view source code
        'sphinx.ext.napoleon', # use numpy docstring format
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/logo.svg"
html_theme_options = {
        "logo": {
            "text": "IQ readout",
            "image_dark": "_static/logo_dark.svg",
            },
        }

