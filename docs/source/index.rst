Welcome to the PeakPerformance documentation!
=============================================

.. image:: https://img.shields.io/pypi/v/peak-performance
   :target: https://pypi.org/project/peak-performance

.. image:: https://img.shields.io/badge/code%20on-Github-lightgrey
   :target: https://github.com/JuBiotech/peak-performance

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10255543.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10255543


``peak_performance`` is a Python toolbox for Bayesian inference of peak areas.

It defines PyMC models describing the intensity curves of chromatographic peaks.

Using Bayesian inference, this enables the fitting of peaks, yielding uncertainty estimates for retention times, peak height, area and much more.

Installation
============

.. code-block:: bash

   pip install peak-performance

You can also download the latest version from `GitHub <https://github.com/JuBiotech/peak-performance>`_.

Tutorials
=========

The documentation features various notebooks that demonstrate the usage and investigate certain aspects of peak modeling.

.. toctree::
   :maxdepth: 1

   notebooks/Ex1_Simple_Pipeline.ipynb
   notebooks/Ex2_Custom_Use_of_PeakPerformance.ipynb
   notebooks/Ex3_Pipeline_with_larger_example_dataset.ipynb
   notebooks/Investigation_doublepeak_separation.ipynb
   notebooks/Investigation_noise_sigma.ipynb


API Reference
=============

.. toctree::
   :maxdepth: 2

   pp_models
   pp_pipeline
   pp_plots
