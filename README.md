# Installation
## Manual installation via Python wheel
Navigate to the download section, download the latest Python wheel (.whl file). Then, open the command line, navigate to the directory containing the wheel, activate a Python environment, and install Peak Performance via `pip install <name of wheel.whl>`.
## Installation via pypi
Pending. An installationg via pypi is in the works but has not been implemented, yet.
## For developers
Clone the Peak Performance repository, then open the command line, navigate to your local clone, activate a Python environment, and install Peak Performance via the command `pip install -e .`.

# How to use Peak Performance
Check out the example notebook(s) under `notebooks` and the complementary example data under `example`.  
Also, there are some introductory explanations in the next sections.
## Preparing raw data
This step is crucial when using Peak Performance. Raw data has to be supplied as time series meaning for each signal you want to analyze, save a NumPy array consisting of time in the first dimension and intensity in the second dimension. Both time and intensity should also be NumPy arrays. If you e.g. have time and intensity of a singal as lists, you can use the following code to convert, format, and save them in the correct manner:  
```
import numpy as np
from pathlib import Path

time_series = np.array([np.array(time), np.array(intensity)])
np.save(Path(r"example_path/time_series.npy"), time_series)
```  
Also, the naming convention of raw data files is "<acquisition_name>_<precursor ion m/z or experiment number>_<product ion m/z>.npy".
## Model selection
When it comes to selecting models, Peak Performance has a function performing an automated selection process by analyzing one acquisiton per mass trace with all implemented models. Subsequently, all models are ranked based on an information criterion (either pareto-smoothed importance sampling leave-one-out cross-validation or widely applicable information criterion). For this process to work as intended, you need to specify acquisitions with representative peaks for each mass trace (see example notebook 1). If e.g. most peaks of an analyte show a skewed shape, then select an acquisition where this is the case. For double peaks, select an acquision where the peaks are as distinct and comparable in height as possible.   
Since model selection is a computationally demanding and time consuming process, it is suggested to state the model type as the user (see example notebook 1) if you happen to know it.
## Examples

## Troubleshooting
If an error occured in the middle of a batch run, then you can use the `pipeline_restart` function to create a new batch which will analyze only those samples, which have not been analyzed previously.

# How to contribute
If you encounter bugs while using Peak Performance, bring it to our attention by opening an issue. When doing so, describe the problem in detail and add screenshots/code snippets or whatever helpful material you can provide.  
When contributing code, create a local clone of Peak Performance, create a new branch, and open a pull request (PR).

# How to cite