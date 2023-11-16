# Installation
It is highly recommended to follow these steps:
1. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Install [Mamba](https://github.com/conda-forge/miniforge/releases)
3. Create a new Python environment in the command line via
```
mamba create -c conda-forge -n name_of_environment pymc nutpie arviz jupyter matplotlib openpyxl "python=3.10"
```
4. Install PeakPerformance:
- For __regular users__: Download the latest Python wheel, then open the command line, navigate to the directory containing the wheel, activate the Python environment created above, and install PeakPerformance via
```
pip install name_of_wheel.whl
```
- For __developers__: Clone the PeakPerformance repository, then open the command line, navigate to your local clone, activate a Python environment, and install PeakPerformance via
```
pip install -e .
```

# How to use PeakPerformance
Check out the example notebook(s) under `notebooks` and the complementary example data under `example`.
Also, there are some introductory explanations in the next sections.

## Preparing raw data
This step is crucial when using PeakPerformance. Raw data has to be supplied as time series meaning for each signal you want to analyze, save a NumPy array consisting of time in the first dimension and intensity in the second dimension (compare example data). Both time and intensity should also be NumPy arrays. If you e.g. have time and intensity of a singal as lists, you can use the following code to convert, format, and save them in the correct manner:
```
import numpy as np
from pathlib import Path

time_series = np.array([np.array(time), np.array(intensity)])
np.save(Path(r"example_path/time_series.npy"), time_series)
```
The naming convention of raw data files is "<acquisition name>_<precursor ion m/z or experiment number>_<product ion m/z start>_<product ion m/z end>.npy". There should be no underscores within the named sections such as `acquisition name`. Essentially, the raw data names include the acquisition and mass trace, thus yielding a recognizable and unique name for each isotopomer/fragment/metabolite/sample.

## Model selection
When it comes to selecting models, PeakPerformance has a function performing an automated selection process by analyzing one acquisiton per mass trace with all implemented models. Subsequently, all models are ranked based on an information criterion (either pareto-smoothed importance sampling leave-one-out cross-validation or widely applicable information criterion). For this process to work as intended, you need to specify acquisitions with representative peaks for each mass trace (see example notebook 1). If e.g. most peaks of an analyte show a skewed shape, then select an acquisition where this is the case. For double peaks, select an acquision where the peaks are as distinct and comparable in height as possible.
Since model selection is a computationally demanding and time consuming process, it is suggested to state the model type as the user (see example notebook 1) if possible.

## Troubleshooting
### A batch run broke and I want to restart it.
If an error occured in the middle of a batch run, then you can use the `pipeline_restart` function in the `pipeline` module to create a new batch which will analyze only those samples, which have not been analyzed previously.

### The model parameters don't converge and/or the fit does not describe the raw data well.
Due to the vast number of LC-MS methods out there, it is probably not possible to formulate befitting prior probability distributions (priors) for all of them. Therefore, one of the first things should be to check in `models.py` whether the model priors make sense for your application and change them according to your data in case they don't. Also, make sure the time series containing the signal to be analyzed contains the peak or double peak (preferrably in the center) and a) no other peaks as well as b) an area around the peak for estimating the baseline (a window size of roughly 5 times the peak width should be fine).

# How to contribute
If you encounter bugs while using PeakPerformance, please bring them to our attention by opening an issue. When doing so, describe the problem in detail and add screenshots/code snippets and whatever other helpful material you can provide.
When contributing code, create a local clone of PeakPerformance, create a new branch, and open a pull request (PR).

# How to cite
Will be updated once the paper has been released and a zenodo DOI has been created.
