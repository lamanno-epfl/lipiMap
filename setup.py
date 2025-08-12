from pathlib import Path
import os

from setuptools import setup, find_packages

long_description = Path("README.rst").read_text("utf-8")

try:
    from lipiMap import __author__, __email__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = "francesca.venturi@epfl.ch"
    __version__ = "0.1"

# otherwise readthedocs fails
# because somewhere in the dependency tree there is the sklearn deprecated package
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

setup(
    name="lipiMap",
    version=__version__,
    description="Lipid Mapping with Variational AutoEncoders and Architecture Surgery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lamanno-epfl/lipiMap",
    author=__author__,
    author_email=__email__,
    # license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "anndata>=0.10.6",
        "community>=1.0.0b1",
        "infomap>=2.8.0",
        "leidenalg>=0.10.2",
        "linex2>=1.1.16",
        "matplotlib>=3.10.5",
        "networkx>=3.5",
        "numpy>=2.3.2",
        "pandas>=2.3.1",
        "python-igraph>=0.11.9",
        "python-louvain>=0.16",
        "sankey>=0.0.2",
        "scikit-learn>=1.7.1",
        "seaborn>=0.13.2",
        "scipy>=1.16.1",
        "tables>=3.10.2",
        "torch>=2.8.0",
        "tqdm>=4.67.1",
    ],
)
