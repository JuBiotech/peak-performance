# TODO: write generic data loading method by scanning directory, then dividing names into parts that allow the identification of both time and intensity for the correct time series as well as retrieval of t_ret and such parameters from a dictionary
# 1) Methode, die Daten in Verzeichnis auflistet und in einzelne Infoblöcke trennt (Format: starten mit Fragmentname, dann mit time oder intensity, Rest Probenname) (Welches Fileformat? Oder verschiedene Methoden bereitstellen? Nein, es geht hier ja um das Ergebnis, nicht um eine allumfassende Pipeline)
# 2) einzelne Methoden für verschiedene Modelle: normal, skew normal, double; evtl. darin noch Funktionen für versch. Baselines aufrufen
# 3) read input-Methode, die user info besser lesbaren Variablen zuteilt? Oder eine, die Excelsheet mit Userinfos ausliest?
# 4) einzelne Methoden für pre-tests
# 5) Speichermethode für einzelne summaries
# 6) Plotmethoden für die verschiedenen Plots

import os

import arviz as az
import numpy as np
import openpyxl
import pandas
import pymc as pm
import pytensor.tensor as pt
import scipy.integrate
import scipy.signal
import scipy.stats as st
from matplotlib import pyplot


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


def prefiltering(intensity, noise_width, t_ret, sn_min, est_width):
    """
    Method to skip signals where clearly no peak is present. Saves a lot of computation time.
    """
    # pre-fit tests for peaks to save computation time (optional)
    # find potential peaks with scipy
    peaks, _ = scipy.signal.find_peaks(intensity)
    peak_candidates = []
    for peak in peaks:
        # test
        if (
            user_info[fragment]["time"][2] - est_width
            <= time[peak]
            <= user_info[fragment]["time"][2] + est_width
            and intensity[peak] / noise_width > 5
            and intensity[peak - 1] / noise_width > 2
            and intensity[peak + 1] / noise_width > 2
        ):
            peak_candidates.append(peak)
    # in case of a double peak, test second retention time, too
    if user_info[fragment]["double_peak"] == True:
        for peak in peaks:
            if (
                user_info[fragment]["time"][3] - est_width
                <= time[peak]
                <= user_info[fragment]["time"][3] + est_width
                and intensity[peak] / noise_width > 5
                and intensity[peak - 1] / noise_width > 2
                and intensity[peak + 1] / noise_width > 2
            ):
                peak_candidates.append(peak)
    if not peak_candidates:
        add_nan_to_summary()
        return False
    return True


def sampling(pmodel):
    with pmodel:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(draws=2000, tune=2000))
    return idata


def resampling(pmodel):
    with pmodel:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(draws=2000, tune=4000))
    return idata


def posterior_predictive_sampling(pmodel, idata):
    with pmodel:
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata


def add_nan_to_summary(acquisition, fragment, masstrace, df_summary):
    """
    Method to add NaN values to the summary DataFrame in case a signal did not contain a peak.
    """
    # create DataFrame with correct format and fill it with NaN
    df = pandas.DataFrame(
        {
            "baseline_intercept": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "baseline_slope": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "mean": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "noise": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "std": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "area": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "height": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
            "sn": {
                "mean": [np.nan],
                "sd": [np.nan],
                "hdi_3%": [np.nan],
                "hdi_97%": [np.nan],
                "mcse_mean": [np.nan],
                "mcse_sd": [np.nan],
                "ess_bulk": [np.nan],
                "ess_tail": [np.nan],
                "r_hat": [np.nan],
            },
        }
    ).transpose()
    # add information about the signal
    df["acquisition"] = 8 * [f"{acquisition}"]
    df["fragment"] = 8 * [f"{fragment}"]
    df["mass_trace"] = 8 * [f"{masstrace}"]
    # concatenate to existing summary DataFrame
    df_summary = pandas.concat([df_summary, df])
    return df_summary


def postfiltering(t_ret, sn_min):
    """
    Method to skip signals where clearly no peak is present. Saves a lot of computation time.
    """
    pass
