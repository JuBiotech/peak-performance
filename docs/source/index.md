---
title: PeakPerformance documentation
---

# Welcome to the PeakPerformance documentation!

[![](https://img.shields.io/pypi/v/peak-performance)](https://pypi.org/project/peak-performance)
[![](https://img.shields.io/badge/code%20on-Github-lightgrey)](https://github.com/JuBiotech/peak-performance)
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.10255543.svg)](https://zenodo.org/doi/10.5281/zenodo.10255543)


``PeakPerformance`` is a Python toolbox for Bayesian inference of peak areas.

It defines PyMC models describing the intensity curves of chromatographic peaks.

Using Bayesian inference, this enables the fitting of peaks, yielding uncertainty estimates for retention times, peak height, area and much more.

This documentation features various notebooks that demonstrate the usage.

# Installation

It is highly recommended to follow the following steps and install ``PeakPerformance`` in a fresh Python environment:
1. Install the package manager [Mamba](https://github.com/conda-forge/miniforge/releases).
Choose the latest installer at the top of the page, click on "show all assets", and download an installer denominated by "Mambaforge-version number-name of your OS.exe", so e.g. "Mambaforge-23.3.1-1-Windows-x86_64.exe" for a Windows 64 bit operating system. Then, execute the installer to install mamba and activate the option "Add Mambaforge to my PATH environment variable".

```{caution}
If you have already installed Miniconda, you can install Mamba on top of it but there are compatibility issues with Anaconda.
```

```{note}
The newest conda version should also work, just replace `mamba` with `conda` in step 2.)
```

2. Create a new Python environment in the command line using the provided [`environment.yml`](https://github.com/JuBiotech/peak-performance/blob/main/environment.yml) file from the repo.
   Download `environment.yml` first, then navigate to its location on the command line interface and run the following command:
```
mamba env create -f environment.yml
```

Naturally, it is alternatively possible to just install ``PeakPerformance`` via pip:

```bash
pip install peak-performance
```

You can also download the latest version from [GitHub](https://github.com/JuBiotech/peak-performance).

```{toctree}
:caption: Tutorials
:maxdepth: 1

notebooks/Preparing_raw_data_for_PeakPerformance
markdown/Peak_model_composition
markdown/PeakPerformance_workflow
markdown/PeakPerformance_validation
notebooks/Recreate_data_from_scratch
markdown/Diagnostic_plots
markdown/How_to_adapt_PeakPerformance_to_your_data
```


```{toctree}
:caption: Examples
:maxdepth: 1

notebooks/Ex1_Simple_Pipeline.ipynb
notebooks/Ex2_Custom_Use_of_PeakPerformance.ipynb
notebooks/Ex3_Pipeline_with_larger_example_dataset.ipynb
```


In the following case studies we investigate certain aspects of peak modeling.

```{toctree}
:caption: Case Studies
:maxdepth: 1

notebooks/Investigation_doublepeak_separation.ipynb
notebooks/Investigation_noise_sigma.ipynb
```


Below you can find documentation that was automatically generated from docstrings.

```{toctree}
:caption: API Reference
:maxdepth: 1

pp_models
pp_pipeline
pp_plots
```
