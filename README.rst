.. raw:: html

add a logo here if you ever happen to have one for lipimap
..  <img src="https://user-images.githubusercontent.com/33202701/187203672-e0415eec-1278-4b2a-a097-5bb8b6ab694f.svg" width="300px" height="200px" align="center">

|PyPI| |PyPIDownloads| |Docs|

LipiMap: Biologically-Informed VAE for Lipidomics
=================================================

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
TODO

See `documentation and tutorials <https://lipimap.readthedocs.io/>`_ for more information.

Support and contribute
----------------------
If you have questions or suggestions to be integrated into our pipeline, you can reach us by `email <francesca.venturi@alumni.epfl.ch>`_.



.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io
