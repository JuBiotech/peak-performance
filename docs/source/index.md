---
title: PeakPerformance documentation
---

# Welcome to the PeakPerformance documentation!

[![](https://img.shields.io/pypi/v/peak-performance)](https://pypi.org/project/peak-performance)
[![](https://img.shields.io/badge/code%20on-Github-lightgrey)](https://github.com/JuBiotech/peak-performance)
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.10255543.svg)](https://zenodo.org/doi/10.5281/zenodo.10255543)


``peak_performance`` is a Python toolbox for Bayesian inference of peak areas.

It defines PyMC models describing the intensity curves of chromatographic peaks.

Using Bayesian inference, this enables the fitting of peaks, yielding uncertainty estimates for retention times, peak height, area and much more.

# Installation

```bash
pip install peak-performance
```

You can also download the latest version from [GitHub](https://github.com/JuBiotech/peak-performance).


The documentation features various notebooks that demonstrate the usage.

```{toctree}
:caption: Tutorials
:maxdepth: 1

markdown/Installation
markdown/Preparing_raw_data
markdown/Peak_model_composition
markdown/PeakPerformance_validation
markdown/PeakPerformance_workflow
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
