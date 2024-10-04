# Preparing raw data

This step is crucial when using PeakPerformance.
Raw data has to be supplied as time series meaning for each signal you want to analyze, save a shape `(2, ?)` NumPy array consisting of time in the first, and intensity in the second entry in the first dimension (compare example data in the repository).
Both time and intensity should also be NumPy arrays.
If you e.g. have time and intensity of a signal as lists, you can use the following code to convert, format, and save them in the correct manner:

```python
import numpy as np

time_series = np.array([np.array(time), np.array(intensity)])
np.save("time_series.npy", time_series)
```

The naming convention of raw data files is `<acquisition name>_<precursor ion m/z or experiment number>_<product ion m/z start>_<product ion m/z end>.npy`. There should be no underscores within the named sections such as `acquisition name`. Essentially, the raw data names include the acquisition and mass trace, thus yielding a recognizable and unique name for each isotopomer/fragment/metabolite/sample. This is of course only relevant when using the pre-manufactured data pipeline and does not apply to user-generated custom data pipelines.
