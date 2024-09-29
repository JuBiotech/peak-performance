---
bibliography:
  - literature.bib
---

# PeakPerformance workflow
$\texttt{PeakPerformance}$ accommodates the use of a pre-manufactured data pipeline for standard applications as well as the creation of custom data pipelines using only its core functions.
The provided data analysis pipeline was designed in a user-friendly way and requires minimal programming knowledge ([Fig. 1](#fig_w1)).
As portrayed in an example notebook in the code repository, only a few simple Python commands need to be executed.
Instead of relying on these convenience functions, experienced users can also directly access the core functions of $\texttt{PeakPerformance}$ for a more flexible application which is demonstrated in yet another example notebook.

```{figure-md} fig_w1
![](./Fig3_PP-standalone.png)

__Figure 1:__ Overview of the pre-manufactured data analysis pipeline featured in `PeakPerformance`.
```

Before using $\texttt{PeakPerformance}$, the user has to supply raw data files containing a NumPy array with time in the first and intensity in the second dimension.
For each peak, such a file has to be provided according to the naming convention specified in $\texttt{PeakPerformance}$'s documentation and gathered in one directory.
If a complete time series of a 30 - 90 min LC-MS/MS run were to be submitted to the program, however, the target peak would make up an extremely small portion of this data.
Additionally, other peaks with the same mass and fragmentation pattern may have been observed at different retention times.
Therefore, it was decided from the outset that in order to enable proper peak fitting, only a fraction of such a time series with a range of 3 - 5 times the peak width and roughly centered on the target peak would be accepted as an input.
This guarantees that there is a sufficient number of data points at the beginning and end of the time frame for estimating the baseline and noise level, as well.
The provided data pipeline starts by defining a path to this raw data directory and one to a local clone of the $\texttt{PeakPerformance}$ code repository.
Using the $\texttt{prepare\_model\_selection()}$ method, an Excel template file ("Template.xlsx") for inputting user information is prepared and stored in the raw data directory.
It is the user's task, then, to select the settings for the pipeline within the file itself.
Accordingly, the file contains detailed explanations of all parameters and the parsing functions of the software feature clear error messages in case mandatory entries are missing or filled out incorrectly.

Since targeted LC-MS/MS analyses essentially cycle through a list of mass traces for every sample, a model type has to be assigned to each mass trace.
Preferably, this is done by the user which is of course only possible when the model choice is self-evident.
If this is not the case, an optional automated model selection step can be performed, where one exemplary peak per mass trace is analyzed with all models to identify the most appropriate one.
It is then assumed that within one batch run, all instances of a mass trace across all acquisitions can be fitted with the same type of model.
For this purpose, the user must provide the name of an acquisition, i.e. sample, where a clear and representative peak for the given mass trace was observed.
If e.g. a standard mixture containing all targets was measured, this would be considered a prime candidate.
An additional feature lets the user exclude specific model types to save computation time and improve the accuracy of model selection by for example excluding double peak models when a single peak was observed.
Upon provision of the required information, the automated model selection can be started using the $\texttt{model\_selection()}$ function from the pipeline module and will be performed successively for each mass trace.
Essentially, every type of model which has not been excluded by the user needs to be instantiated, sampled, and the log-likelihood needs to be calculated.
Subsequently, the results for each model are ranked with the $\texttt{compare()}$ function of the ArviZ package based on Pareto-smoothed importance sampling leave-one-out cross-validation (LOO-PIT) [@RN146; @RN145].
This function returns a DataFrame showing the results of the models in order of their placement on the ranking which is decided by the expected log pointwise predictive density.
The best model for each mass trace is then written to the Excel template file.

After a model was chosen either manually or automatically for each mass trace, the peak analysis pipeline can be started with the function $\texttt{pipeline()}$ from the $\texttt{pipeline}$ module.
The first step consists of parsing the information from the Excel sheet.
Since the pipeline, just like model selection, acts successively, a time series is read from its data file next and the information contained in the name of the file according to the naming convention is parsed.
All this information is combined in an instance of $\texttt{PeakPerformance}$'s $\texttt{UserInput}$ class acting as a centralized source of data for the program.
Depending on whether the "pre-filtering" option was selected, an optional filtering step will be executed to reject signals where clearly no peak is present before sampling, thus saving computation time.
This filtering step uses the $\texttt{find\_peaks()}$ function from the SciPy package [@scipy] which simply checks for data points directly neighboured by points with lower intensity values.
If no data points within a certain range around the expected retention time of an analyte fulfill this most basic requirement of a peak, the signal is rejected.
Furthermore, if none of the candidate data points exceed a signal-to-noise ratio threshold defined by the user, the signal will also be discarded.
Depending on the origin of the samples, this step may reject a great many signals before sampling saving potentially hours of computation time across a batch run of the $\texttt{PeakPerformance}$ pipeline.
For instance, in bioreactor cultivations, a product might be quantified but if it is only produced during the stationary growth phase, it will not show up in early samples.
Another pertinent example of such a use case are isotopic labeling experiments for which every theoretically achievable mass isotopomer needs to be investigated, yet depending on the input labeling mixture, the majority of them might not be present in actuality.
Upon passing the first filter, a Markov chain Monte Carlo (MCMC) simulation is conducted using a No-U-Turn Sampler (NUTS) [@RN173], preferably - if installed in the Python environment - the nutpie sampler [@nutpie] due to its highly increased performance compared to the default sampler of PyMC.
Before sampling from the posterior distribution, a prior predictive check is performed the results of which can be accessed and evaluated after the fact.
When a posterior distribution has been obtained, the main filtering step is next in line.
The first criterion is constituted by checking the convergence of the Markov chains towards a common solution for the posterior represented by the potential scale reduction factor [@RN152], also referred to as the $\hat{R}$ statistic or Gelman-Rubin diagnostic.
If this factor is above 1.05 for any parameter, convergence was not reached and the sampling will be repeated once with a much higher number of tuning samples.
If the filter is not passed a second time, the pertaining signal is rejected.
Harnessing the advantages of the uncertainty quantification, a second criterion calculates the ratio of the resulting standard deviation of a peak parameter to its mean and discards signals exceeding a threshold.
Usually, false positives passing the first criterion are rather noisy signals where a fit was achieved but the uncertainty on the peak parameters is extremely high.
These signals will then be rejected by the second criterion, ultimately reducing the number of false positive peaks significantly if not eliminating them.
If a signal was accepted as a peak, the final simulation step is a posterior predictive check which is added to the inference data object resulting from the model simulation.

# Bibliography
