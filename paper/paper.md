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
Currently, its application is focused on data from targeted liquid chromatography tandem mass spectrometry (LC-MS/MS), but its design allows for an expansion to other chromatographic techniques.
$\texttt{PeakPerformance}$ is implemented in Python and the source code is available on [GitHub](https://github.com/JuBiotech/peak-performance).
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

# Materials and Methods
## Implementation
$\texttt{PeakPerformance}$ is an open source Python package compatible with Windows and Linux/Unix platforms.
At the time of manuscript submission, it features three modules: `pipeline`, `models`, and `plotting`.
Due to its modular design, $\texttt{PeakPerformance}$ can easily be expanded by adding e.g. additional models for deviating peak shapes or different plots.
Currently, the featured peak models describe peaks in the shape of normal or skew normal distributions, as well as double peaks of normal or skewed normal shape.
The normal distribution is regarded as the ideal peak shape and common phenomena like tailing and fronting can be expressed by the skew normal distribution [@RN144].\\
Bayesian inference is conducted utilizing the PyMC package [@RN150] with the external sampler $\texttt{nutpie}$ for improved performance [@nutpie].
Both model selection and analysis of inference data objects are realized with the ArviZ package [@RN147].
Since the inference data is stored alongside graphs and report sheets, users may employ the ArviZ package or others for further analysis of the results if necessary.

## Validation of $\texttt{PeakPerformance}$
Several stages of validation were employed to prove the suitability of $\texttt{PeakPerformance}$ for chromatographic peak data analysis.
For the first and second tests, 500 random data sets were generated with the NumPy random module [@harris2020array] by drawing from the normal distributions detailed in Table 1 except for the mean parameter which was held constant at a value of 6.
Additionally, in case of the second test, the area was set to 8 and the skewness parameter $\alpha$ to 1.
Subsequently, normally distributed random noise ($\mathcal{N}(0, 0.6)$ or $\mathcal{N}(0, 1.2)$ for data sets with the tag "higher noise") was added to each data point.
The amount of data points per time was chosen based on an LC-MS/MS method routinely utilized by the authors and accordingly set to one data point per 1.8 s.

__Table 1:__ Normal distributions from which parameters were drawn randomly to create synthetic data sets for the validation of $\texttt{PeakPerformance}$.

| **parameter**      | **model (1st test)**    | **model (2nd test)**  |
| ------------------ | ----------------------- | ----------------------- |
| area               | $\mathcal{N}(8, 0.5)$   | -                       |
| standard deviation | $\mathcal{N}(0.5, 0.1)$ | $\mathcal{N}(0.5, 0.1)$ |
| skewness           | $\mathcal{N}(0, 2)$    | -                       |
| baseline intercept | $\mathcal{N}(25, 1)$    | $\mathcal{N}(25, 1)$    |
| baseline slope     | $\mathcal{N}(0, 1)$     | $\mathcal{N}(0, 1)$     |

For the third and final test, experimental peak data was analyzed with both $\texttt{PeakPerformance}$ (version 0.7.0) and Sciex MultiQuant (version 3.0.3) with human supervision, i.e. the results were visually inspected and corrected if necessary.
The data set consisted of 192 signals comprised of 123 single peaks, 50 peaks as part of double peaks, and 19 noise signals.


# Results and Discussion

## PeakPerformance workflow
$\texttt{PeakPerformance}$ accommodates the use of a pre-manufactured data pipeline for standard applications (Fig. 1) as well as the creation of custom data pipelines using only its core functions.
The provided data analysis pipeline was designed in a user-friendly way and is covered by multiple example notebooks.

![](./Fig3_PP-standalone.png)
__Figure 1:__ Overview of the pre-manufactured data analysis pipeline featured in $\texttt{PeakPerformance}$.

Before using $\texttt{PeakPerformance}$, the user has to supply raw data files containing a NumPy array with time in the first and intensity in the second dimension for each peak according as described in detail in the documentation.
Using the $\texttt{prepare\_model\_selection()}$ method, an Excel template file ("Template.xlsx") for inputting user information is prepared and stored in the raw data directory.

Since targeted LC-MS/MS analyses essentially cycle through a list of mass traces for every sample, a model type has to be assigned to each mass trace.
If this is not done by the user, an optional automated model selection step will be performed, where one exemplary peak per mass trace is analyzed with all models to identify the most appropriate one.
The automated model selection can be started using the $\texttt{model\_selection()}$ function from the pipeline module and will be performed successively for each mass trace.
The results for each model are ranked with the $\texttt{compare()}$ function of the ArviZ package based on Pareto-smoothed importance sampling leave-one-out cross-validation (LOO-PIT) [@RN146; @RN145].

Subsequently, the peak analysis pipeline can be started with the function $\texttt{pipeline()}$ from the $\texttt{pipeline}$ module.
Depending on whether the "pre-filtering" option was selected, an optional filtering step will be executed to reject signals where clearly no peak is present before sampling, thus saving computation time.
This filtering step combines the $\texttt{find\_peaks()}$ function from the SciPy package [@scipy] with a user-defined minimum signal-to-noise threshold and may reject a great many signals before sampling, e.g. in the case of isotopic labeling experiments where every theoretically achievable mass isotopomer needs to be investigated, yet depending on the input labeling mixture, the majority of them might not be present in actuality.
Upon passing the first filter, a Markov chain Monte Carlo (MCMC) simulation is conducted using a No-U-Turn Sampler (NUTS) [@RN173], preferably - if installed in the Python environment - the nutpie sampler [@nutpie] due to its highly increased performance compared to the default sampler of PyMC.
Before sampling from the posterior distribution, a prior predictive check is performed.
When a posterior distribution has been obtained, the main filtering step is next in line which checks the convergence of the Markov chains via the potential scale reduction factor [@RN152] or $\hat{R}$ statistic and based on the uncertainty of the determined peak parameters.
If a signal was accepted as a peak, a posterior predictive check is conducted and added to the inference data object resulting from the model simulation.


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

__Table 2:__ Depiction of the results for the most important peak parameters of a single peak fit with the skew normal model and a double peak fit with the double normal model. Mean, area, and height have been highlighted in bold print as they constitute the most relevant parameters for further data evaluation purposes. The results correspond to the fits exhibited in Figure 2.

![](./summary_joint.svg){width="100%"}

In this case, the fits were successful and convergence was reached for all parameters.
Most notably and for the first time, the measurement noise was taken into account when determining the peak area as represented by its standard deviation and as can be observed in the posterior predictive plots where the noisy data points fall within the boundary of the 95 % HDI.


## Validation
In the first stage of validation, peak fitting with normal and skew normal peak models was tested regarding the ability to reproduce the ground truth of randomly generated noisy synthetic data sets.
The arithmetic means portrayed in Figure 3a were calculated based on a measure of similarity

$$\tag{12}F_{y / \hat{y}} = \frac{y}{\hat{y}}$$

where $y$ represents the estimated parameter value and $\hat{y}$ its pertaining ground truth.
As they exhibit values close to 1, this demonstrates a near identity between estimation and ground truth.
Additionally, the normal-shaped peak model was paired with skew normally distributed noisy data and vice versa.
In both cases, $\sigma$ was not reproduced well, especially by the normal-shaped model.
Nevertheless, the peak area and height were still identified correctly with the skew normal model and merely slightly underestimated by the normal model.

![](./Fig6_PP-validation.png)
__Figure 3:__ Validation of results from $\texttt{PeakPerformance}$. **a)** Noisy synthetic data was randomly generated from one of the implemented distributions and the program's ability to infer the ground truth was observed. Portrayed are the fractions of estimated parameter to ground truth. **b)** The influence of model choice between normal and skew normal model in marginal cases with little to no skew was tested and the ratios between results from both models are plotted. **c)** Lastly, experimental data was analyzed with $\texttt{PeakPerformance}$ version 0.7.0 and compared to results achieved with the commercial software Sciex MultiQuant version 3.0.3.

In the second stage, marginal cases in the form of slightly skewed peaks were investigated to observe whether their estimation with a normal- or skew normal-shaped intensity function would result in significant differences in terms of peak area and height.
Here, a slight skew was defined as an $\alpha$ parameter of 1 resulting in peak shapes not visibly discernible as clearly normal or skew normal.
With a sample size of 100 noisy, randomly generated data sets, we show that nearly identical estimates for peak area and height, as well as their respective uncertainties are obtained regardless of the utilized model (Fig. 3b).
The exhibited mean values are based on fractions of the key peak parameters area and height between results obtained with a normal and skew normal model which were defined as

$$\tag{13}F_{n / \mathrm{sn}} = \frac{A_{\mathcal{N}}}{A_{\mathrm{skew \ normal}}}$$

where $A_{normal}$ and $A_{skew \ normal}$ are the estimated areas with normal and skew normal models, respectively.

In the third stage, experimental peak data was analyzed with both $\texttt{PeakPerformance}$ (version 0.7.0) and Sciex MultiQuant (version 3.0.3) and the fraction of the obtained areas was determined as

$$\tag{14}F_{\mathrm{MQ} / \mathrm{PP}} = \frac{A_{\mathrm{MQ}}}{A_{\mathrm{PP}}}$$

where $A_{\mathrm{MQ}}$ denominates the area yielded by MultiQuant and $A_{\mathrm{PP}}$ the area from $\texttt{PeakPerformance}$.
Beyond the comparability of the resulting peak area ratio means portrayed in Figure 3c, it is relevant to state that 103 signals from MultiQuant (54 % of total signals) were manually modified.
Of these, 31 % were false positives and 69 % were manually re-integrated.
These figures are the result of a relatively high share of double peaks in the test sample which generally give a lot more cause for manual interference than single peaks.
In contrast, however, the $\texttt{PeakPerformance}$ pipeline was only started once and merely two single peaks and one double peak were fit again with a different model and/or increased sample size after the original pipeline batch run had finished.
Among the 192 signals of the test data set, there were 7 noisy, low intensity signals without a clear peak which were recognized as a peak only by either one or the other software and were hence omitted from this comparison.
By showing not only the mean area ratio of all peaks but also the ones for the single and double peak subgroups, it is evident that the variance is significantly higher for double peaks.
It could be demonstrated that $\texttt{PeakPerformance}$ yields comparable peak area results to a commercially available vendor software.

# Conclusions
$\texttt{PeakPerformance}$ is a tool for automated LC-MS/MS peak data analysis employing Bayesian inference.
It provides built-in uncertainty quantification by Bayesian parameter estimation and thus for the first time takes the measurement noise of an LC-MS/MS device into account when integrating peaks.
Regarding peak acceptance, it improves on vendor software solutions with more sophisticated, multi-layered metrics for decision making based on convergence of the parameter estimation, as well as the uncertainties of peak parameters.
Finally, it allows the addition of new models to describe peak intensity functions with just a few minor code changes, thus lending itself to expansion to data from other chromatographic techniques.
The design of $\texttt{PeakPerformance}$ accommodates users with little programming experience by supplying convenience functions and relying on Microsoft Excel for data input and reporting.
Its code repository on GitHub features automated unit tests, and an automatically built documentation (https://peak-performance.rtfd.io).


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
The datasets generated during and/or analysed during the current study are available from the corresponding author on reasonable request.

# Bibliography
