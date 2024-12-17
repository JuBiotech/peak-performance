[![PyPI version](https://img.shields.io/pypi/v/peak-performance)](https://pypi.org/project/peak-performance/)
[![pipeline](https://github.com/jubiotech/peak-performance/workflows/pipeline/badge.svg)](https://github.com/JuBiotech/peak-performance/actions)
[![coverage](https://codecov.io/gh/jubiotech/peak-performance/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuBiotech/peak-performance)
[![documentation](https://readthedocs.org/projects/peak-performance/badge/?version=latest)](https://peak-performance.readthedocs.io/en/latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07313/status.svg)](https://doi.org/10.21105/joss.07313)
[![DOI](https://zenodo.org/badge/713469041.svg)](https://zenodo.org/doi/10.5281/zenodo.10255543)

# About PeakPerformance
PeakPerformance employs Bayesian modeling for chromatographic peak data fitting.
This has the innate advantage of providing uncertainty quantification while jointly estimating all peak parameters united in a single peak model.
As Markov Chain Monte Carlo (MCMC) methods are utilized to infer the posterior probability distribution, convergence checks and the aformentioned uncertainty quantification are applied as novel quality metrics for a robust peak recognition.

# Installation

It is highly recommended to follow the following steps and install ``PeakPerformance`` in a fresh Python environment:
1. Install the package manager [Mamba](https://github.com/conda-forge/miniforge/releases).
Choose the latest installer at the top of the page, click on "show all assets", and download an installer denominated by "Mambaforge-version number-name of your OS.exe", so e.g. "Mambaforge-23.3.1-1-Windows-x86_64.exe" for a Windows 64 bit operating system. Then, execute the installer to install mamba and activate the option "Add Mambaforge to my PATH environment variable".

⚠ If you have already installed Miniconda, you can install Mamba on top of it but there are compatibility issues with Anaconda.

ℹ The newest conda version should also work, just replace `mamba` with `conda` in step 2.

2. Create a new Python environment in the command line using the provided [`environment.yml`](https://github.com/JuBiotech/peak-performance/blob/main/environment.yml) file from the repo.
   Download `environment.yml` first, then navigate to its location on the command line interface and run the following command:
```
mamba env create -f environment.yml
```

Naturally, it is alternatively possible to just install ``PeakPerformance`` via pip:

```bash
pip install peak-performance
```

# First steps
Be sure to check out our thorough [documentation](https://peak-performance.readthedocs.io/en/latest). It contains not only information on how to install PeakPerformance and prepare raw data for its application but also detailed treatises about the implemented model structures, validation with both synthetic and experimental data against a commercially available vendor software, exemplary usage of diagnostic plots and investigation of various effects.
Furthermore, you will find example notebooks and data sets showcasing different aspects of PeakPerformance.

# How to contribute
If you encounter bugs while using PeakPerformance, please bring them to our attention by opening an issue. When doing so, describe the problem in detail and add screenshots/code snippets and whatever other helpful material you can provide.
When contributing code, create a local clone of PeakPerformance, create a new branch, and open a pull request (PR).

# How to cite
Head over to Zenodo to [generate a BibTeX citation](https://doi.org/10.5281/zenodo.10255543) for the latest release.  
In addition to the utilized software version, cite our scientific publication over at the Journal of Open Source Software (JOSS).
A detailed citation can be found in CITATION.cff and in the sidebar.
