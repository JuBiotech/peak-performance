import datetime
import os
import zipfile

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
    all_files = os.listdir(path)
    return [file for file in all_files if ".npy" in file]


def scan_folder(path):
    """
    Detect all files in a given directory and returns them as a list. 
    The files should a) contain time and intensity data and b) be named according to the naming scheme (will automatically be correct when downloaded from the MS data cluster). 

    Parameters
    ----------
    path
        path to the folder containing raw data
    """
    return os.listdir(path)


def parse_data_name(files):
    """
    Extract names of data files.
    """
    #TODO: change according to new scheme
    # data files are supposed to be denominated with a unique identifier and then time or intensity, sepearted by an underscore
    # e.g. "Flask1-t30_time.npy", "Flask1-t30_intensity.npy", "Flask2-t30_time.npy", "Flask2-t30_intensity.npy" etc.
    identifier = set([line.split("_")[0] for line in files])
    return identifier


def initiate(path):
    """
    Create a folder for the results. Also create a zip file inside that folder. Also create summary_df.
    
    Parameters
    ----------
    path
        Path to the directory containing the raw data
    
    Returns
    -------
    df_summary
        DataFrame for storing results
    path
        Updated path variable pointing to the newly created folder
    """
    # get current date and time
    today = datetime.date.today()
    current_date = today.strftime('%Y_%m_%d__%H_%M_%S.')
    # create a directory
    path = path + "/" + current_date + "run"
    os.mkdir(rf"{path}")
    # write text file, zip it, then delete it (cannot create an empty zip)
    text_file = open(rf"{path}/readme.txt", "w")
    txt = text_file.write(f'This batch was started on the {current_date}.')
    text_file.close()
    with zipfile.ZipFile(rf"{path}/idata.zip", mode="w") as archive:
        archive.write(txt)
    os.remove(rf"{path}/readme.txt")
    # create DataFrame for data report
    df_summary = pandas.DataFrame(columns=["acquisition", "fragment", "mass_trace", "baseline_intercept","baseline_slope", "mean", "noise", "std", "area", "height", "sn"])
    return df_summary, path


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


def sampling(pmodel, **sample_kwargs):
    sample_kwargs.setdefault("tune", 2000)
    sample_kwargs.setdefault("draws", 2000)
    with pmodel:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(**sample_kwargs))
    return idata


def postfiltering(t_ret, sn_min):
    """
    Method to skip signals where clearly no peak is present. Saves a lot of computation time.
    """

    pass


def posterior_predictive_sampling(pmodel, idata):
    with pmodel:
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata


def report_save_idata(idata):
    """
    Saves inference data object within a zip file.

    Parameters
    ----------
    idata
        Inference data object resulting from sampling
    """
    with zipfile.ZipFile("idata.zip", mode="a") as archive:
        archive.write(idata.to_netcdf("idata"))
    return


def report_add_data_to_summary(idata, summary_df, path):
    """
    Extracts the relevant information from idata, concatenates it to the summary DataFrame, and saves the DataFrame as an Excel file.
    Error handling prevents stop of the pipeline in case the saving doesn't work (e.g. because the file was opened by someone).

    Parameters
    ----------
    path
        Path to the directory containing the raw data
    idata
        Inference data object resulting from sampling
    """
    parameters = ["baseline_intercept","baseline_slope", "mean", "noise", "std", "area", "height", "sn"]
    df = az.summary(idata).loc[parameters,:]
    df["acquisition"] = len(parameters) * [f"{acquisition}"]
    df["fragment"] = len(parameters) * [f"{fragment}"]
    df["mass_trace"] = len(parameters) * [f"{mass_trace}"]
    df_summary = pandas.concat([df_summary, df])
    pandas.concat(summary_df, df)
    # save summary df as Excel file
    with pandas.ExcelWriter(path=path, engine="openpyxl", mode="w") as writer:
        summary_df.to_excel(writer)
    return summary_df


def report_area_sheet(path, df_summary):
    """
    Save a different, more minimalist report sheet focussing on the area data.
    
    Parameters
    ----------
    path
        Path to the directory containing the raw data
    df_summary
        DataFrame for storing results
    """
    # also save a version of df_summary only for areas with correct order and only necessary data
    df_area_summary = df_summary[df_summary.index == "area"]
    sorted_area_summary = df_area_summary.sort_values(["acquisition", "mass_trace"])
    sorted_area_summary = sorted_area_summary.drop(labels=["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail"], axis=1)
    sorted_area_summary.to_excel(fr"{path}/area_summary.xlsx")
    return


def report_add_nan_to_summary(acquisition, fragment, masstrace, df_summary):
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
