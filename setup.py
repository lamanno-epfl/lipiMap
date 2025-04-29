from pathlib import Path
import os

from setuptools import setup, find_packages

# long_description = Path('README.rst').read_text('utf-8')

# try:
#     from scarches import __author__, __email__, __version__
# except ImportError:  # Deps not yet installed
__author__ = __email__ = 'francesca.venturi@epfl.ch'
__version__ = '0.1'

# otherwise readthedocs fails
# because somewhere in the dependency tree there is the sklearn deprecated package
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

setup(name='lipiMap',
      version=__version__,
      description='Lipid Mapping with Variational AutoEncoders and Architecture Surgery',
      # long_description=long_description,
      long_description_content_type="text/markdown",
      # url='https://github.com/uplamamnno/lipimap', check
      author=__author__,
      author_email=__email__,
      # license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
        "scanpy>=1.6.0",
    	  "anndata>=0.7.4",
        "scHPL>=1.0.0",
        "h5py>=2.10.0",
        "torch>=1.8.0",
        "numpy>=1.19.2",
        "scipy>=1.5.2",
        "scikit-learn>=0.23.2",
        "matplotlib>=3.3.1",
        "scvi-tools>=0.12.1",
        "tqdm>=4.56.0",
        "pandas",
        "requests",
        "gdown",
        "leidenalg",  # TODO: set version criteria
        "muon",
      ],
      )
