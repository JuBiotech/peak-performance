# Contents
Welcome to the peak fitting tool **Peak Performance** using a Bayesian approach to fit data peaks from **targeted LC-MS/MS analyses**, yielding **probability distributions** for key parameters such as the peak area. The astute reader will have noticed that this introduces uncertainty quantification for every single analyzed peak, thus enabling (and necessitating) error propagation in subsequent calculations or continuing with the posterior distributions in the vein of the Bayesian approach. Thereby, a more realistic and honest interpretation and presentation of experimental data is enabled. Because as per our adage...
![logo](./docs/Peak_Performance_Logo.png)

# Installation
  1) Find the Python wheel of the latest version of Peak Performance in this repository under `Packages and Registries` in the `Package Registry`.
  2) Download the wheel.
  3) Access the command line interface, and navigate to the directory containing the wheel.
  4) Activate a conda Python environment and install Peak Performance via the command `pip install name_of_wheel.whl`.

# Usage
You can find Jupyter notebooks with exemplary pipelines in the `notebooks` folder. The results of the examples are stored in the `example` folder.

Basically, you have to provide the raw data as time series (time vs. intensity) and provide the path to the folder with (only) the raw data as well as some general information about the data which is explained in more detail in the examples. Then, you can make use of a general pipeline for filtering and sampling the signals. The results will be stored in a fresh folder for each run of Peak Performance.
