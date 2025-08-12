.. raw:: html

LipiMap: Biologically-Informed VAE for Lipidomics
=================================================
.. image:: images/model.png
   :width: 300px
   :height: 200px
   :align: center

Why LipiMap
-----------
LipiMap addresses the complexity of biological systems by leveraging advanced machine learning to analyze lipidomics data.

Despite the critical roles of lipids in energy storage, cell structure, and signaling, lipidomics has been relatively under-explored with modern ML techniques.
LipiMap fills this gap by distilling high-dimensional data into interpretable Lipid Programs (LPs).

Lipid Programs (LPs)
~~~~~~~~~~~~~~~~~~~~
Lipid Programs are groups of lipids that function as biological modules within lipid metabolism. They aim to simplify the understanding of lipid metabolism,
a complex network of biochemical processes involving the synthesis, breakdown, and regulation of lipids.

LipiMap uses LPs to analyze interactions and behaviors across different cellular compartments and physiological conditions, providing insights into the functional landscape of lipids.

What is LipiMap: Probabilistic Mapping onto Biologically Interpretable Latent Space
-----------------------------------------------------------------------------------
LipiMap is a Biologically-Informed Variational Autoencoder tailored for lipidomic data analysis.
It constructs a biologically informed latent space using Lipid Programs as building blocks to model the dynamic behavior of lipid metabolism.

This capability enables to explore the active and inactive states of LPs across different brain units,
allowing for a spatially-aware analysis of lipid metabolism across the brain.


Installation
------------

**Sorry!** At the moment, LipiMap is only installable from source. We apologize for the inconvenience—rest assured, a PyPI package will be made available soon!

**(Recommended) Create a conda environment before installing:**

.. code-block:: bash

   conda create -n lipimap python=3.12.2
   conda activate lipimap


To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/lamanno-epfl/lipiMap.git
   cd lipimap
   pip install -e .

**Test your installation:**

.. code-block:: bash

   python -c "import lipiMap; print(lipiMap.__version__)"


We invite you to become familiar with the code and its features by exploring our comprehensive tutorial notebook:

   notebooks/TUTORIAL.ipynb

This notebook will guide you through the main functionalities and typical workflows of LipiMap.

.. See `documentation and tutorials <https://lipimap.readthedocs.io/>`_ for more information.

Support and contribute
----------------------
If you have questions or suggestions to be integrated into our pipeline, you can reach us by `email <francesca.venturi@alumni.epfl.ch>`_.