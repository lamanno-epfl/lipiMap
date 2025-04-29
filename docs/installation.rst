Installation
============


lipiMap requires Python 3.12. We recommend to use Miniconda.

PyPI
--------

The easiest way to get lipiMap is through pip using the following command::

    sudo pip install -U lipiMap


Conda Environment
---------------------

You can also use our environment file. This will create the conda environment 'lipimap' with
the required dependencies::

    git clone https://github.com/lamanno-epfl/lipiMap
    cd lipiMap
    conda env create -f envs/lipimap.yaml
    conda activate lipimap


Development
---------------

You can also get the latest development version of lipiMap from `Github <https://github.com/lamanno-epfl/lipiMap/>`_ using the following steps:
First, clone lipiMap using ``git``::

    git clone https://github.com/lamanno-epfl/lipiMap


Then, ``cd`` to the lipiMap folder and run the install command::

    cd lipiMap
    python3 setup.py install

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself.

Dependencies
------------

The list of dependencies for lipiMap can be found in the `requirements.txt <https://github.com/lamanno-epfl/lipiMap/blob/master/docs/requirements.txt>`_ file in the repository.

If you run into issues, do not hesitate to approach us.
