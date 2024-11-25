---
title: 'PeakPerformance - A tool for Bayesian inference-based fitting of LC-MS/MS peaks'
tags:
  - Peak fitting
  - Bayesian inference
  - Chromatography
  - LC-MS/MS
  - Python
  - Uncertainty quantification

authors:
  - name: Jochen Nießer
    orcid: 0000-0001-5397-0682
    affiliation: 1, 2
  - name: Michael Osthege
    orcid: 0000-0002-2734-7624
    affiliation: 1
  - name: Eric von Lieres
    orcid: 0000-0002-0309-8408
    affiliation: "1, 3"
  - name: Wolfgang Wiechert
    orcid: 0000-0001-8501-0694
    affiliation: "1, 3"
  - name: Stephan Noack
    orcid: 0000-0001-9784-3626
    affiliation: 1

affiliations:
 - name: Institute for Bio- and Geosciences (IBG-1), Forschungszentrum Jülich GmbH, Jülich, Germany
   index: 1
 - name: RWTH Aachen University, Aachen, Germany
   index: 2
 - name: Computational Systems Biotechnology, RWTH Aachen University, Aachen, Germany
   index: 3

date: 12th of August 2024
bibliography:
  - literature.bib
---

# Summary

A major bottleneck of chromatography-based analytics has been the elusive fully automated identification and integration of peak data without the need of extensive human supervision.
The presented Python package $\texttt{PeakPerformance}$ applies Bayesian inference to chromatographic peak fitting, and provides an automated approach featuring model selection and uncertainty quantification.
Regarding peak acceptance, it improves on vendor software solutions with more sophisticated, multi-layered metrics for decision making based on convergence of the parameter estimation, as well as the uncertainties of peak parameters.
Currently, its application is focused on data from targeted liquid chromatography tandem mass spectrometry (LC-MS/MS), but its design allows for an expansion to other chromatographic techniques and accommodates users with little programming experience by supplying convenience functions and relying on Microsoft Excel for data input and reporting.
$\texttt{PeakPerformance}$ is implemented in Python, its source code is available on [GitHub](https://github.com/JuBiotech/peak-performance), and a through documentation is available under [https://peak-performance.rtfd.io](https://peak-performance.rtfd.io).
It is unit-tested on Linux and Windows and accompanied by documentation as well as example notebooks.

# Statement of need

In biotechnological research and industrial applications, chromatographic techniques are ubiquitously used to analyze samples from fermentations, e.g. to determine the concentration of substrates and products.
Over the course of a regular lab-scale bioreactor fermentation, hundreds of samples and subsequently thousands of chromatographic peaks may accrue.
This is exacerbated by the spread of microbioreactors causing a further increase in the amount of samples per time [@RN149; @RN148].
While the recognition and integration of peaks by vendor software is – in theory – automated, it typically requires visual inspection and occasional manual re-integration by the user due to a large number of false positives, false negatives or incorrectly determined baselines, ultimately downgrading it to a semi-automated process.
Since this is a time-consuming, not to mention tedious, procedure and introduces the problem of comparability between purely manual and algorithm-based integration as well as user-specific differences, we instead propose a peak fitting solution based on Bayesian inference.
The advantage of this approach is the complete integration of all relevant parameters – i.e. baseline, peak area and height, mean, signal-to-noise ratio etc. – into one single model through which all parameters are estimated simultaneously.
Furthermore, Bayesian inference comes with uncertainty quantification for all peak model parameters, and thus does not merely yield a point estimate as would commonly be the case.
It also grants access to novel metrics for avoiding false positives and negatives by rejecting signals where a) a convergence criterion of the peak fitting procedure was not fulfilled or b) the uncertainty of the estimated parameters exceeded a user-defined threshold.
By employing peak fitting to uncover peak parameters – most importantly the area – this approach thus differs from recent applications of Bayesian statistics to chromatographic peak data which e.g. focussed on peak detection [@vivo2012bayesian; @woldegebriel2015probabilistic], method optimization [@wiczling2016much] and simulations of chromatography [@briskot2019prediction; @yamamoto2021uncertainty].
The first studies to be published about this topic contain perhaps the technique most similar in spirit to the present one since functions made of an idealized peak shape and a noise term are fitted but beyond this common starting point the methodology is quite distinct [@kelly1971estimation; @kelly1971application].

# Materials and Methods
## Implementation
$\texttt{PeakPerformance}$ is an open source Python package compatible with Windows and Linux/Unix platforms.
At the time of manuscript submission, it features three modules: `pipeline`, `models`, and `plotting`.
Due to its modular design, $\texttt{PeakPerformance}$ can easily be expanded by adding e.g. additional models for deviating peak shapes or different plots.
Currently, the featured peak models describe peaks in the shape of normal or skew normal distributions, as well as double peaks of normal or skewed normal shape.
The normal distribution is regarded as the ideal peak shape and common phenomena like tailing and fronting can be expressed by the skew normal distribution [@RN144].
Bayesian inference is conducted utilizing the PyMC package [@RN150] with the external sampler $\texttt{nutpie}$ for improved performance [@nutpie].
Both model selection and analysis of inference data objects are realized with the ArviZ package [@RN147].
Since the inference data is stored alongside graphs and report sheets, users may employ the ArviZ package or others for further analysis of the results if necessary.


# Results and Discussion

## PeakPerformance workflow
$\texttt{PeakPerformance}$ accommodates the use of a pre-manufactured data pipeline for standard applications (Fig. 1) as well as the creation of custom data pipelines using only its core functions.
The provided data analysis pipeline was designed in a user-friendly way and is covered by multiple example notebooks.

Before using $\texttt{PeakPerformance}$, the user has to supply raw data files containing a NumPy array with time in the first and intensity in the second dimension for each peak as described in detail in the documentation.
Using the $\texttt{prepare\_model\_selection()}$ method, an Excel template file ("Template.xlsx") for inputting user information is prepared and stored in the raw data directory.

Since targeted LC-MS/MS analyses essentially cycle through a list of mass traces for every sample, a model type has to be assigned to each mass trace.
If this is not done by the user, an optional automated model selection step will be performed, where one exemplary peak per mass trace is analyzed with all models to identify the most appropriate one.
Its results for each model are ranked based on Pareto-smoothed importance sampling leave-one-out cross-validation (LOO-PIT) [@RN146; @RN145].

![](./Fig3_PP-standalone.png)
__Figure 1:__ Overview of the pre-manufactured data analysis pipeline featured in $\texttt{PeakPerformance}$.

Subsequently, the peak analysis pipeline can be started with the function $\texttt{pipeline()}$ from the $\texttt{pipeline}$ module.
Depending on whether the "pre-filtering" option was selected, an optional filtering step will be executed to reject signals where clearly no peak is present before sampling, thus saving computation time.
Upon passing the first filter, a Markov chain Monte Carlo (MCMC) simulation is conducted using a No-U-Turn Sampler (NUTS) [@RN173], preferably - if installed in the Python environment - the nutpie sampler [@nutpie] due to its highly increased performance compared to the default sampler of PyMC.
When a posterior distribution has been obtained, the main filtering step is next in line which checks the convergence of the Markov chains via the potential scale reduction factor [@RN152] or $\hat{R}$ statistic and based on the uncertainty of the determined peak parameters.
If a signal was accepted as a peak, a posterior predictive check is conducted and added to the inference data object resulting from the model simulation.
Regarding the performance of the simulation, in our tests an analysis of a single peaks took 20 s to 30 s and of a double peaks 25 s to 90 s.
This is of course dependent on the power of the computer as well as whether an additional simulation with an increased number of samples needs to be conducted.


## Peak fitting results and diagnostic plots
The most complete report created after completing a cycle of the data pipeline is found in an Excel file called "peak_data_summary.xlsx".
Here, each analyzed time series has multiple rows (one per peak parameter) with the columns containing estimation results in the form of mean and standard deviation (sd) of the marginal posterior distribution, highest density interval (HDI), and the $\hat{R}$ statistic among other metrics.
The second Excel file created is denominated as "area_summary.xlsx" and is a more handy version of "peak_data_summary.xlsx" with a reduced degree of detail since subsequent data analyses will most likely rely on the peak area.
The most valuable result, however, are the inference data objects saved to disk for each signal for which a peak function was successfully fitted.
Conveniently, the inference data objects saved as $\texttt{*.nc}$ files contain all data and metadata related to the Bayesian parameter estimation, enabling the user to perform diagnostics or create custom visualizations not already provided by $\texttt{PeakPerformance}$.
Regarding data visualization with the matplotlib package [@matplotlib; @matplotlibzenodo], $\texttt{PeakPerformance}$'s $\texttt{plots}$ module offers the generation of two diagram types for each successfully fitted peak.
The posterior plot presents the fit of the intensity function alongside the raw data points.
The first row of Figure 2 presents two such examples where the single peak diagram shows the histidine (His) fragment with a m/z ratio of 110 Da and the double peak diagram the leucine (Leu) and isoleucine (Ile) fragments with a m/z ratio of 86 Da.

![](./Fig4_peak_results.png)
__Figure 2:__ Results plots for a single His peak and a double Leu and Ile peak depicting the peak fit (first row) and the posterior predictive checks (second row) alongside the raw data. The numerical results are listed in table 2.

The posterior predictive plots in the second row of Figure 4 are provided for visual posterior predictive checks, namely the comparison of observed and predicted data distribution.
Since a posterior predictive check is based on drawing samples from the likelihood function, the result represents the theoretical range of values encompassed by the model.
Accordingly, this plot enables users to judge whether the selected model can accurately explain the data.
To complete the example, Table 2 shows the results of the fit in the form of mean, standard deviation, and HDI of each parameter's marginal posterior.

\pagebreak

__Table 2:__ Depiction of the results for the most important peak parameters of a single peak fit with the skew normal model and a double peak fit with the double normal model. Mean, area, and height have been highlighted in bold print as they constitute the most relevant parameters for further data evaluation purposes. The results correspond to the fits exhibited in Figure 2.

![](./summary_joint.svg){width="100%"}

In this case, the fits were successful and convergence was reached for all parameters.
Most notably and for the first time, the measurement noise was taken into account when determining the peak area as represented by its standard deviation and as can be observed in the posterior predictive plots where the noisy data points fall within the boundary of the 95 % HDI.
In the documentation, there is a study featuring simulated and experimental data to validate $\texttt{PeakPerformance}$'s results against a commercially available vendor software for peak integration showing that comparable results are indeed obtained.

### Author contributions
$\texttt{PeakPerformance}$ was conceptualized by JN and MO.
Software implementation was conducted by JN with code review by MO.
The original draft was written by JN with review and editing by MO, SN, EvL, and WW.
The work was supervised by SN and funding was acquired by SN, EvL, and WW.

### Acknowledgements
The authors thank Tobias Latour for providing experimental LC-MS/MS data for the comparison with the vendor software MultiQuant. Funding was received from the German Federal Ministry of Education and Research (BMBF) (grant number 031B1134A) as part of the innovation lab "AutoBioTech" within the project "Modellregion, BioRevierPLUS: BioökonomieREVIER Innovationscluster Biotechnologie \& Kunststofftechnik".

### Competing interests
No competing interest is declared.

### Data availability
The datasets generated during and/or analysed during the current study are available in version 0.7.1 of the [Zenodo record](https://zenodo.org/records/11189842).

# Bibliography
