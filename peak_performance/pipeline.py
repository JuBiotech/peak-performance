import datetime
import os
import zipfile
from typing import Any, Dict, List, Sequence, Tuple, Union

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


class ParsingError(Exception):
    """Base type of parsing exceptions."""


class InputError(Exception):
    """Base type of exceptions related to information given by the user."""


class UserInput:
    """Collect all information required from the user and format them in the correct manner."""

    def __init__(
        self,
        path: str,
        files: list,
        double_peak: List[bool],
        retention_time_estimate: Union[List[float], List[int]],
        peak_width_estimate: Union[float, int],
        pre_filtering: bool = True,
    ):
        """
        Parameters
        ----------
        path
            Path to the folder containing raw data.
        files
            List of raw data file names in path.
        double_peak
            List with Booleans in the same order as files. Set to True, if the corresponding file contains a double peak, and set to False, if it contains a single peak.
        retention_time_estimate
            In case you set pre_filtering to True, give a retention time estimate (float) for each signal in files. In case of a double peak, give two retention times (in chronological order) as a tuple containing two floats.
        peak_width_estimate
            Rough estimate of the average peak width in minutes expected for the LC-MS method with which the data was obtained.
        pre_filtering
            If True, potential peaks will be filtered based on retention time and signal to noise ratio before sampling.
        """
        self.path = path
        self.files = files
        self.double_peak = double_peak
        self.retention_time_estimate = retention_time_estimate
        self.peak_width_estimate = peak_width_estimate
        self.pre_filtering = pre_filtering
        super().__init__()

    @property
    def user_info(self):
        """Create a dictionary with the necessary user information based on the class attributes."""
        # first, some sanity checks
        if len(self.files) != len(self.double_peaks):
            raise InputError(
                f"The length of 'files' ({len(self.files)}) and of 'double_peak' ({len(self.double_peak)}) are not identical."
            )
        if self.pre_filtering:
            # check length of lists
            if len(self.files) != len(self.pre_filtering) or len(self.double_peak) != len(
                self.retention_time_estimate
            ):
                raise InputError(
                    f"The length of 'files' ({len(self.files)}), 'double_peak' ({self.double_peak}), and retention_time_estimate ({len(self.retention_time_estimate)}) are not identical."
                )
        else:
            # if pre_filtering is False, then retention_time_estimate is not needed but the dictionary still needs to be created without errors -> set it to None
            if len(self.retention_time_estimate) == 1:
                self.retention_time_estimate = len(self.files) * None
            elif not self.retention_time_estimate:
                self.retention_time_estimate = len(self.files) * None
        if self.retention_time_estimate.any() < 0:
            raise InputError("Retention time estimates below 0 are not valid.")
        # actually create the dictionary
        user_dict = dict(
            zip(self.raw_data_files, zip(self.double_peak, self.retention_time_estimate))
        )
        user_dict["peak_width_estimate"] = self.peak_width_estimate
        user_dict["pre_filtering"] = self.pre_filtering
        return user_dict


def detect_npy(path):
    """
    Detect all .npy files with time and intensity data for peaks in a given directory.

    Parameters
    ----------
    path
        Path to the folder containing raw data.

    Returns
    -------
    npy_files
        List with names of all .npy files in path.
    """
    all_files = os.listdir(path)
    npy_files = [file for file in all_files if ".npy" in file]
    if not npy_files:
        raise ParsingError(f"In the given directory '{path}', there are no .npy files.")
    return npy_files


def scan_folder(path):
    """
    Detect all files in a given directory and returns them as a list.
    The files should a) contain time and intensity data and b) be named according to the naming scheme (will automatically be correct when downloaded from the MS data cluster).

    Parameters
    ----------
    path
        Path to the folder containing raw data.
    """
    return os.listdir(path)


def parse_data(filename: str):
    """
    Extract names of data files. Use this in a for-loop with the data file names from detect_npy() or scane_folder().

    Parameters
    ----------
    filename
        Name of a raw date file containing a numpy array with a time series (time as first, intensity as second element of the array).

    Returns
    -------
    timeseries
        Numpy array with a time series (time as first, intensity as second element of the array).
    """
    # load time series
    timeseries = np.load(f"{filename}")
    # get information from the raw data file name
    splits = filename.split("_")
    acquisition = splits[0]
    experiment = splits[1]
    precursor_mz = splits[2]
    product_mz_start = splits[3]
    # remove the .npy suffix from the last split
    product_mz_end = splits[4][:-4]
    return timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end


def initiate(path):
    """
    Create a folder for the results. Also create a zip file inside that folder. Also create summary_df.

    Parameters
    ----------
    path
        Path to the directory containing the raw data.

    Returns
    -------
    df_summary
        DataFrame for storing results.
    path
        Updated path variable pointing to the newly created folder for this batch.
    """
    # get current date and time
    today = datetime.date.today()
    current_date = today.strftime("%Y_%m_%d__%H_%M_%S.")
    # create a directory
    path = path + "/" + current_date + "run"
    os.mkdir(rf"{path}")
    # write text file, zip it, then delete it (cannot create an empty zip)
    text_file = open(rf"{path}/readme.txt", "w")
    txt = text_file.write(f"This batch was started on the {current_date}.")
    text_file.close()
    with zipfile.ZipFile(rf"{path}/idata.zip", mode="w") as archive:
        archive.write(txt)
    os.remove(rf"{path}/readme.txt")
    # create DataFrame for data report
    df_summary = pandas.DataFrame(
        columns=[
            "acquisition",
            "fragment",
            "mass_trace",
            "baseline_intercept",
            "baseline_slope",
            "mean",
            "noise",
            "std",
            "area",
            "height",
            "sn",
        ]
    )
    return df_summary, path


def prefiltering(filename: str, user_info: dict, timeseries: np.array, noise_width: float):
    """
    Optional method to skip signals where clearly no peak is present. Saves a lot of computation time.

    Parameters
    ----------
    filename
        Name of the raw data file.
    user_info
        Dictionary with user specified information.
    timeseries
        Numpy array with a time series (time as first, intensity as second element of the array).
    noise_width
        Estimated width of the noise of a particular measurement.

    Returns
    -------
    Bool
        True, if any peak candidate was found within the time frame; False, if not.
    """
    # pre-fit tests for peaks to save computation time (optional)
    t_ret = user_info[filename][1]
    est_width = user_info[est_width]
    sn_min = user_info[sn_min]
    # find all potential peaks with scipy
    peaks, _ = scipy.signal.find_peaks(timeseries[1])
    peak_candidates = []
    # differentiate between single and double peaks
    if not user_info[filename]["double_peak"]:
        # single peaks
        for peak in peaks:
            if (
                # check proximity of any peak candidate to the estimated retention time
                t_ret - est_width <= timeseries[0][peak] <= t_ret + est_width
                # check signal to noise ratio
                and timeseries[1][peak] / noise_width > 5
                # check the neighbouring data points to prevent classification of a single elevated data point as a peak
                and timeseries[1][peak - 1] / noise_width > 2
                and timeseries[1][peak + 1] / noise_width > 2
            ):
                peak_candidates.append(peak)
    else:
        # double peaks
        for peak in peaks:
            if (
                # check proximity of any peak candidate to the estimated retention time of either the first or the second peak
                t_ret[0] - est_width <= timeseries[0][peak] <= t_ret[0] + est_width
                or t_ret[1] - est_width <= timeseries[0][peak] <= t_ret[1] + est_width
                # check signal to noise ratio
                and timeseries[1][peak] / noise_width > 5
                # check the neighbouring data points to prevent classification of a single elevated data point as a peak
                and timeseries[1][peak - 1] / noise_width > 2
                and timeseries[1][peak + 1] / noise_width > 2
            ):
                peak_candidates.append(peak)
    if not peak_candidates:
        report_add_nan_to_summary()
        return False
    return True


def sampling(pmodel, **sample_kwargs):
    sample_kwargs.setdefault("tune", 2000)
    sample_kwargs.setdefault("draws", 2000)
    with pmodel:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(**sample_kwargs))
    return idata


def postfiltering(idata):
    """
    Method to filter out false positive peaks after sampling based on the obtained uncertainties of several peak parameters.

    Parameters
    ----------
    idata
        Inference data object resulting from sampling
    user_info
        Dictionary with user specified information.

    Returns
    -------
    Bool
        True, if the signal passed the test; False, if the signal was not recognized as a peak.
    """
    # check whether convergence, i.e. r_hat <= 1.05, was not reached OR peak criteria (explanation see next comment) were not met
    if not user_info[fragment]["double_peak"] == True:
        if (
            any(list(az.summary(idata).loc[:, "r_hat"])) > 1.05
            or az.summary(idata).loc["std", :]["mean"] <= 0.1
            or az.summary(idata).loc["area", :]["sd"]
            > az.summary(idata).loc["area", :]["mean"] * 0.2
            or az.summary(idata).loc["height", :]["sd"]
            > az.summary(idata).loc["height", :]["mean"] * 0.2
        ):
            # decide whether to discard signal or sample with more tune samples based on size of sigma parameter of normal distribution (std) and on the relative sizes of standard deviations of area and heigt
            if (
                az.summary(idata).loc["std", :]["mean"] <= 0.1
                or az.summary(idata).loc["area", :]["sd"]
                > az.summary(idata).loc["area", :]["mean"] * 0.2
                or az.summary(idata).loc["height", :]["sd"]
                > az.summary(idata).loc["height", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                report_add_nan_to_summary()
                continue
            else:
                # r_hat failed but rest of post-fit check passed
                # sample again with more tune samples to possibly reach convergence yet
                with pmodel:
                    idata2 = pm.sample_prior_predictive()
                    idata2.extend(pm.sample(draws=2000, tune=4000))
                # if still no convergence, kick it
                if any(list(az.summary(idata2).loc[:, "r_hat"])) > 1.05:
                    # add NaN values to summary DataFrame
                    report_add_nan_to_summary()
                    continue
                # if results still don't meet plausibility/quality criteria, kick it
                elif (
                    az.summary(idata2).loc["std", :]["mean"] <= 0.1
                    or az.summary(idata2).loc["area", :]["sd"]
                    > az.summary(idata2).loc["area", :]["mean"] * 0.2
                    or az.summary(idata2).loc["height", :]["sd"]
                    > az.summary(idata2).loc["height", :]["mean"] * 0.2
                ):
                    # add NaN values to summary DataFrame
                    report_add_nan_to_summary()
                    continue
                # if result is improved, accept new inference data and go on
                else:
                    idata = idata2
    else:
        if (
            any(list(az.summary(idata).loc[:, "r_hat"])) > 1.05
            or az.summary(idata).loc["std", :]["mean"] <= 0.1
            or az.summary(idata).loc["area", :]["sd"]
            > az.summary(idata).loc["area", :]["mean"] * 0.2
            or az.summary(idata).loc["height", :]["sd"]
            > az.summary(idata).loc["height", :]["mean"] * 0.2
            or az.summary(idata).loc["std2", :]["mean"] <= 0.1
            or az.summary(idata).loc["area2", :]["sd"]
            > az.summary(idata).loc["area2", :]["mean"] * 0.2
            or az.summary(idata).loc["height2", :]["sd"]
            > az.summary(idata).loc["height2", :]["mean"] * 0.2
        ):
            # subdivide name of fragment into its two parts
            fragment1 = fragment.split("_")[0]
            fragment2 = fragment.split("_")[1]
            # Booleans to differentiate which peak is or is not detected
            double_not_found_first = False
            double_not_found_second = False
            # decide whether to discard signal or sample with more tune samples based on size of sigma parameter of normal distribution (std) and on the relative sizes of standard deviations of area and heigt
            if (
                az.summary(idata).loc["std", :]["mean"] <= 0.1
                or az.summary(idata).loc["area", :]["sd"]
                > az.summary(idata).loc["area", :]["mean"] * 0.2
                or az.summary(idata).loc["height", :]["sd"]
                > az.summary(idata).loc["height", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                report_add_nan_to_summary()
                double_not_found_first = True
            if (
                az.summary(idata).loc["std2", :]["mean"] <= 0.1
                or az.summary(idata).loc["area2", :]["sd"]
                > az.summary(idata).loc["area2", :]["mean"] * 0.2
                or az.summary(idata).loc["height2", :]["sd"]
                > az.summary(idata).loc["height2", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                report_add_nan_to_summary()
                double_not_found_second = True
            print(
                f"end of check 1: {fragment}, {acquisition}: {double_not_found_first}, {double_not_found_second}"
            )
            # if both peaks failed the r_hat and peak criteria tests, then continue
            if double_not_found_first and double_not_found_second:
                continue
            # r_hat failed but rest of post-fit check passed
            # sample again with more tune samples to possibly reach convergence yet
    return


def posterior_predictive_sampling(pmodel, idata):
    with pmodel:
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata


def report_save_idata(path, idata):
    """
    Saves inference data object within a zip file.

    Parameters
    ----------
    idata
        Inference data object resulting from sampling
    """
    with zipfile.ZipFile(rf"{path}/idata.zip", mode="a") as archive:
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
    # TODO split double peak up when reporting into first and second peak (when extracting the data from az.summary(idata))
    parameters = [
        "baseline_intercept",
        "baseline_slope",
        "mean",
        "noise",
        "std",
        "area",
        "height",
        "sn",
    ]
    df = az.summary(idata).loc[parameters, :]
    df["acquisition"] = len(parameters) * [f"{acquisition}"]
    df["fragment"] = len(parameters) * [f"{fragment}"]
    df["mass_trace"] = len(parameters) * [f"{mass_trace}"]
    df_summary = pandas.concat([df_summary, df])
    pandas.concat(summary_df, df)
    # save summary df as Excel file
    with pandas.ExcelWriter(path=path, engine="openpyxl", mode="w") as writer:
        summary_df.to_excel(writer)
    return summary_df


def add_double_peak_data_to_summary():
    return


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
    sorted_area_summary = sorted_area_summary.drop(
        labels=["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail"], axis=1
    )
    sorted_area_summary.to_excel(rf"{path}/area_summary.xlsx")
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
