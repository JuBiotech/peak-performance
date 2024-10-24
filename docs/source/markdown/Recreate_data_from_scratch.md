# Recreate the presented data in paper and documentation from scratch

## Recreate Figure 2 from the PeakPerformance publication

Navigate to `docs/source/notebooks` and run the `Create_results_in_figure_2.ipynb` notebook.  
  
It is separated into two sections which work and are structured in an analogous manner.
The first creates the results figure for the single peak and the second for the double peak.
Both sections walk through the following sequential steps:  
  1. open and plot example raw data
  2. define a model
  3. perform both sampling and posterior predictive sampling
  4. display the summary DataFrame containing the results of the peak fitting
  5. display cumulative plot of the posterior predictive check
  6. display the posterior predictive check and the peak fit against the raw data points.

## Recreate the validation plot from the documentation

To actually recreate the validation plot, navigate to `docs/source/notebooks` and run the notebook `Create_validation_plot_from_raw_data.ipynb`.  
  
However, not all data loaded in this notebook is raw data.
Particularly, the data from the first stage of validation using synthetic data sets is pre-processed based on the results of said test using the notebook `Processing_test_1_raw_data.ipynb`.
Since all necessary files are present for both notebooks, they can be run in any order.  
  
Also, the data for the comparison with the commercial software MultiQuant in the third stage of validation is contained in `docs/source/notebooks/test3_df_comparison.xlsx`.
The `PeakPerformance` results listed in this file have been obtained by executing a batch run with the raw data stored in `docs/source/notebooks/paper raw data` using the settings detailed in the `Template.xlsx` file in the same directory.
