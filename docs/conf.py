# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "JaxMaterials"
copyright = "2026, Eike Mueller"
author = "Eike Mueller"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "myst_parser",
    "sphinx.ext.mathjax",
]
myst_enable_extensions = [
    "dollarmath",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "_backend.py", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
