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
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_class_signature = "separated"


templates_path = ["_templates"]
exclude_patterns = ["_build", "_backend.py", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
