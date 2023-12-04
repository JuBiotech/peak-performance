[![PyPI version](https://img.shields.io/pypi/v/peak-performance)](https://pypi.org/project/peak-performance/)
[![pipeline](https://github.com/jubiotech/peak-performance/workflows/pipeline/badge.svg)](https://github.com/JuBiotech/peak-performance/actions)
[![coverage](https://codecov.io/gh/jubiotech/peak-performance/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuBiotech/peak-performance)
[![DOI](https://zenodo.org/badge/713469041.svg)](https://zenodo.org/doi/10.5281/zenodo.10255543)

# How to use PeakPerformance
For installation instructions, see `Installation.md`.
For instructions regarding the use of PeakPerformance, check out the example notebook(s) under `notebooks`, the complementary example data under `example`, and the following introductory explanations.

## Preparing raw data
This step is crucial when using PeakPerformance. Raw data has to be supplied as time series meaning for each signal you want to analyze, save a NumPy array consisting of time in the first dimension and intensity in the second dimension (compare example data). Both time and intensity should also be NumPy arrays. If you e.g. have time and intensity of a singal as lists, you can use the following code to convert, format, and save them in the correct manner:
```
import numpy as np
from pathlib import Path

time_series = np.array([np.array(time), np.array(intensity)])
np.save(Path(r"example_path/time_series.npy"), time_series)
```
The naming convention of raw data files is `<acquisition name>_<precursor ion m/z or experiment number>_<product ion m/z start>_<product ion m/z end>.npy`. There should be no underscores within the named sections such as `acquisition name`. Essentially, the raw data names include the acquisition and mass trace, thus yielding a recognizable and unique name for each isotopomer/fragment/metabolite/sample.

## Model selection
When it comes to selecting models, PeakPerformance has a function performing an automated selection process by analyzing one acquisiton per mass trace with all implemented models. Subsequently, all models are ranked based on an information criterion (either pareto-smoothed importance sampling leave-one-out cross-validation or widely applicable information criterion). For this process to work as intended, you need to specify acquisitions with representative peaks for each mass trace (see example notebook 1). If e.g. most peaks of an analyte show a skewed shape, then select an acquisition where this is the case. For double peaks, select an acquision where the peaks are as distinct and comparable in height as possible.
Since model selection is a computationally demanding and time consuming process, it is suggested to state the model type as the user (see example notebook 1) if possible.

## Troubleshooting
### A batch run broke and I want to restart it.
If an error occured in the middle of a batch run, then you can use the `pipeline_restart` function in the `pipeline` module to create a new batch which will analyze only those samples, which have not been analyzed previously.

### The model parameters don't converge and/or the fit does not describe the raw data well.
Check the separate file `How to adapt PeakPerformance to your data`.

# How to contribute
If you encounter bugs while using PeakPerformance, please bring them to our attention by opening an issue. When doing so, describe the problem in detail and add screenshots/code snippets and whatever other helpful material you can provide.
When contributing code, create a local clone of PeakPerformance, create a new branch, and open a pull request (PR).

# How to cite
Head over to Zenodo to [generate a BibTeX citation](https://doi.org/10.5281/zenodo.10255543) for the latest release.
A publication has just been submitted to a scientific journal. Once published, this section will be updated.
