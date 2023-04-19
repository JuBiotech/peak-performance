# TODO: write generic data loading method by scanning directory, then dividing names into parts that allow the identification of both time and intensity for the correct time series as well as retrieval of t_ret and such parameters from a dictionary
# 1) Methode, die Daten in Verzeichnis auflistet und in einzelne Infoblöcke trennt (Format: starten mit Fragmentname, dann mit time oder intensity, Rest Probenname) (Welches Fileformat? Oder verschiedene Methoden bereitstellen? Nein, es geht hier ja um das Ergebnis, nicht um eine allumfassende Pipeline)
# 2) einzelne Methoden für verschiedene Modelle: normal, skew normal, double; evtl. darin noch Funktionen für versch. Baselines aufrufen
# 3) read input-Methode, die user info besser lesbaren Variablen zuteilt? Oder eine, die Excelsheet mit Userinfos ausliest?
# 4) einzelne Methoden für pre-tests
# 5) Speichermethode für einzelne summaries
# 6) Plotmethoden für die verschiedenen Plots

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from matplotlib import pyplot
import scipy.stats as st
import scipy.signal
import scipy.integrate
import pandas
import openpyxl
import time
import json
import math

def detect_npy(path):
    """
    Detect all .npy files with time and intensity data for peaks in a given directory.
    """
    path = r"C:\Users\joche\Desktop\FZTest\His110 test data"
    all_files = os.listdir(path)
    return [file for file in all_files if ".npy" in file]

def parse_data_name(files):
    """
    Extract names of data files.
    """
    # data files are supposed to be denominated with a unique identifier and then time or intensity, sepearted by an underscore
    # e.g. "Flask1-t30_time.npy", "Flask1-t30_intensity.npy", "Flask2-t30_time.npy", "Flask2-t30_intensity.npy" etc.
    identifier = set([line.split("_")[0] for line in files])
    return identifier

def prefiltering(t_ret, sn_min):
    """
    Method to skip signals where clearly no peak is present. Saves a lot of computation time.
    """
    pass


def initial_guesses(time_np, intensity, unique_identifier):
    """
    Provide initial guesses for priors
    """
    # select lowest third of all data points as noise -> noise_tuple
    intensity_tuple = list(enumerate(intensity))
    intensity_tuple.sort(key=lambda x: x[1])
    noise_range = int(np.round(1/3 * len(intensity), decimals=0))
    noise_tuple = intensity_tuple[:noise_range]
    # sort noise_tuple by time, then use the first and last data points for estimating starting values for the priors of noise width and the slope and intercept of baseline
    noise_tuple.sort(key=lambda x: x[0])
    slope_guess = (np.mean([noise_tuple[n][1] for n in range(len(noise_tuple)-5, len(noise_tuple))])-np.mean([noise_tuple[n][1] for n in range(5)]))/(time_np[-1] - time_np[0])
    intercept_guess = np.mean([noise_tuple[n][1] for n in range(5)])
    noise_width_guess = np.max([noise_tuple[n][1] for n in range(len(noise_tuple))]) - np.min([noise_tuple[n][1] for n in range(len(noise_tuple))])
    # TODO: somehow save the guesses? Or discard them, after all?
    # guess_dict["slope_guess"][f"{unique_identifier}"] = slope_guess
    # guess_dict["intercept_guess"][f"{unique_identifier}"] = intercept_guess
    # guess_dict["noise_width_guess"][f"{unique_identifier}"] = noise_width_guess
    
    return intercept_guess, slope_guess, noise_width_guess

def define_model_normal(time_np, intensity, unique_identifier):
    """
    Define a model for fitting a normal distribution to the peak data.
    """
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time_np, intensity, unique_identifier)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, intercept_guess / 2)
        baseline_slope = pm.Normal("baseline_slope", slope_guess, np.abs(slope_guess*2)+1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time_np)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        mean = pm.Normal("mean", np.mean(time_np[[0, -1]]), np.ptp(time_np) / 2)
        std = pm.HalfNormal("std", np.ptp(time_np) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity))
        area = pm.Deterministic("area", height/(1 / (std * np.sqrt(2 * np.pi))))
        sn = pm.Deterministic("sn", height/noise)
        # posterior
        y = baseline + height * pt.exp(-0.5 * ((time_np - mean) / std)**2)
        y = pm.Deterministic("y", y)

        # likelihood (auf der y-Achse liegende Normalverteilung um Datenpunkte, entspricht hier Normalverteilung um Wert mit Standardabweichung = noise)
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity)
    
    return pmodel

def define_model_doublepeak(time_np, intensity):
    """
    Define a model for fitting two ordered normal distributions to the peak data (for when data contains two peaks or a double peak without baseline separation).
    """
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time_np, intensity, unique_identifier)
    with pm.Model() as pmodel:
        # priors (two peaks, hence two sets of parameters)
        baseline = pm.Normal("baseline", 500, 50)
        noise = pm.LogNormal("noise", np.log(100), 1)
        std = pm.HalfNormal("std", np.ptp(time_np) / 3)
        std2 = pm.HalfNormal("std2", np.ptp(time_np) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity))
        height2 = pm.HalfNormal("height2", 0.95 * np.max(intensity))
        area = pm.Deterministic("area", height/(1 / (std * np.sqrt(2 * np.pi))))
        area2 = pm.Deterministic("area2", height2/(1 / (std2 * np.sqrt(2 * np.pi))))
        # use univariate ordered normal distribution
        mean = pm.Normal("mean", mu=[time_np[0] + np.ptp(time_np) * 1/4, time_np[0] + np.ptp(time_np) * 3/4], sigma=1, transform=pm.distributions.transforms.univariate_ordered)

        # posterior
        y = baseline + height * pt.exp(-0.5 * ((time_np - mean[0]) / std)**2) + height2 * pt.exp(-0.5 * ((time_np - mean[1]) / std2)**2)
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel
    

def define_model_skew():
    """
    Define a model for fitting a skew normal distribution to the peak data.
    """
    pass

def sampling():
    pass

def posterior_predictive_sampling():
    pass

def add_nan_to_summary(acquisition, fragment, masstrace, df_summary):
    """
    Method to add NaN values to the summary DataFrame in case a signal did not contain a peak.
    """
    # create DataFrame with correct format and fill it with NaN
    df = pandas.DataFrame(
        {
        "baseline_intercept": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        "baseline_slope": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        "mean": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        "noise": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        "std": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]}, 
        "area": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]}, 
        "height": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        "sn": {"mean": [np.nan], "sd": [np.nan], "hdi_3%": [np.nan], "hdi_97%": [np.nan], "mcse_mean": [np.nan], "mcse_sd": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan], "r_hat": [np.nan]},
        }
    ).transpose()
    # add information about the signal
    df["acquisition"] = 8 * [f"{acquisition}"]
    df["fragment"] = 8 * [f"{fragment}"]
    df["mass_trace"] = 8 * [f"{masstrace}"]
    # concatenate to existing summary DataFrame
    df_summary = pandas.concat([df_summary,df])
    return df_summary


def postfiltering(t_ret, sn_min):
    """
    Method to skip signals where clearly no peak is present. Saves a lot of computation time.
    """
    pass

def plot_raw_data():
    """
    Plot just the raw data in case no peak was found.
    """
    pass

def plot_posterior_predictive():
    pass

def plot_posterior():
    pass