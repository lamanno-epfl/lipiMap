# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

# -- Readthedocs theme -------------------------------------------------------
#on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

html_theme = "furo"

import lipiMap

# -- Retrieve notebooks ------------------------------------------------------

from urllib.request import urlretrieve

notebooks_url = 'https://github.com/lamanno-epfl/lipiMap/raw/master/notebooks/' # https://github.com/theislab/scarches/raw/master/notebooks/
notebooks = [
    # 'TUTORIAL.ipynb',
]

for nb in notebooks:
    try:
        urlretrieve(notebooks_url + nb, nb)
    except:
        pass

# -- Project information -----------------------------------------------------

project = 'lipiMap'
author = ' Francesca Venturi, Luca Fusar Bassini, Halima Hannah Schede, Gioele La Manno'
copyright = f'{datetime.now():%Y}, ' + author

pygments_style = 'sphinx'
todo_include_todos = True
html_theme_options = dict(navigation_depth=3, titles_only=False)
html_context = dict(
    display_github=True,
    github_user='lamanno-epfl',
    github_repo='lipiMap',
    github_version='master',
    conf_py_path='/docs/',
)
html_static_path = ['_static']


def setup(app):
    app.add_css_file('custom.css')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'nbsphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx_rtd_theme',
    'numpydoc',
    "sphinx.ext.viewcode",  # Adds links to source code
    "myst_parser",
]

add_module_names = True
autosummary_generate = True
numpydoc_show_class_members = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.readthedocs.io/en/latest/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/latest/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(lipiMap.__file__))

    github = f"https://github.com/lamanno-epfl/lipiMap/blob/master/lipiMap/{fn}{linespec}"
    return github
