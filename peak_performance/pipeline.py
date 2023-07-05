import os
import zipfile
from datetime import date, datetime
from numbers import Number
from pathlib import Path
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

from peak_performance import models
from peak_performance import plots


class ParsingError(Exception):
    """Base type of parsing exceptions."""


class InputError(Exception):
    """Base type of exceptions related to information given by the user."""


class UserInput:
    """Collect all information required from the user and format them in the correct manner."""

    def __init__(
        self,
        path: Union[str, os.PathLike],
        files: Sequence[Union[str, os.PathLike]],
        double_peak: Sequence[bool],
        retention_time_estimate: Union[Sequence[float], Sequence[int]],
        peak_width_estimate: Union[float, int],
        pre_filtering: bool,
        minimum_sn: Union[float, int],
        timeseries: np.ndarray,
        acquisition: str,
        experiment: int,
        precursor_mz: Number,
        product_mz_start: Number,
        product_mz_end: Number,
    ):
        """
        Parameters
        ----------
        path
            Path to the folder containing the results of the current run.
        files
            List of raw data file names in path.
        double_peak
            List with Booleans in the same order as files. Set to True, if the corresponding file contains a double peak, and set to False, if it contains a single peak.
        retention_time_estimate
            In case you set pre_filtering to True, give a retention time estimate (float) for each signal in files.
            In case of a double peak, give two retention times (in chronological order) as a tuple containing two floats.
        peak_width_estimate
            Rough estimate of the average peak width in minutes expected for the LC-MS method with which the data was obtained.
        pre_filtering
            If True, potential peaks will be filtered based on retention time and signal to noise ratio before sampling.
        minimum_sn
            Minimum signal to noise ratio for a signal to be recognized as a peak during pre-filtering.
        timeseries
            NumPy Array containing time (at first position) and intensity (at second position) data as NumPy arrays.
        acquisition
            Name of a single acquisition.
        experiment
            Experiment number of the signal within the acquisition (each experiment = one mass trace).
        precursor_mz
            Mass to charge ratio of the precursor ion selected in Q1.
        product_mz_start
            Start of the mass to charge ratio range of the product ion in the TOF.
        product_mz_end
            End of the mass to charge ratio range of the product ion in the TOF.
        """
        self.path = path
        self.files = list(files)
        self.double_peak = double_peak
        self.retention_time_estimate = retention_time_estimate
        self.peak_width_estimate = peak_width_estimate
        self.pre_filtering = pre_filtering
        self.minimum_sn = minimum_sn
        self._timeseries = timeseries
        self._acquisition = acquisition
        self._experiment = experiment
        self._precursor_mz = precursor_mz
        self._product_mz_start = product_mz_start
        self._product_mz_end = product_mz_end
        super().__init__()

    @property
    def timeseries(self):
        """Getting the value of the timeseries attribute."""
        return self._timeseries

    @timeseries.setter
    def timeseries(self, data):
        """Setting the value of the timeseries attribute."""
        if data is None:
            raise InputError(f"The timeseries parameter is a None type.")
        self._timeseries = np.asarray(data)

    @property
    def acquisition(self):
        """Getting the value of the acquisition attribute."""
        return self._acquisition

    @acquisition.setter
    def acquisition(self, name):
        """Setting the value of the acquisition attribute."""
        if not isinstance(name, str):
            raise InputError(f"The acquisition parameter is {type(name)} but needs to be a string.")
        if name is None:
            raise InputError(f"The acquisition parameter is a None type.")
        self._acquisition = name

    @property
    def experiment(self):
        """Getting the value of the experiment attribute."""
        return self._experiment

    @experiment.setter
    def experiment(self, number):
        """Setting the value of the experiment attribute."""
        if not isinstance(number, int):
            try:
                number = int(number)
            except:
                raise InputError(
                    f"The experiment parameter is {type(number)} but needs to be an integer."
                )
        if number is None:
            raise InputError(f"The experiment parameter is a None type.")
        self._experiment = number

    @property
    def precursor_mz(self):
        """Getting the value of the precursor_mz attribute."""
        return self._precursor_mz

    @precursor_mz.setter
    def precursor_mz(self, mz):
        """Setting the value of the precursor_mz attribute."""
        if not isinstance(mz, int) and not isinstance(mz, float):
            try:
                mz = float(mz)
            except:
                raise InputError(
                    f"The precursor_mz parameter is {type(mz)} but needs to be an integer or a float."
                )
        if mz is None:
            raise InputError(f"The precursor_mz parameter is a None type.")
        self._precursor_mz = mz

    @property
    def product_mz_start(self):
        """Getting the value of the product_mz_start attribute."""
        return self._product_mz_start

    @product_mz_start.setter
    def product_mz_start(self, mz):
        """Setting the value of the product_mz_start attribute."""
        if not isinstance(mz, int) and not isinstance(mz, float):
            try:
                mz = float(mz)
            except:
                raise InputError(
                    f"The precursor_mz parameter is {type(mz)} but needs to be an integer or a float."
                )
        if mz is None:
            raise InputError(f"The product_mz_start parameter is a None type.")
        self._product_mz_start = mz

    @property
    def product_mz_end(self):
        """Getting the value of the product_mz_end attribute."""
        return self._product_mz_end

    @product_mz_end.setter
    def product_mz_end(self, mz):
        """Setting the value of the product_mz_end attribute."""
        if not isinstance(mz, int) and not isinstance(mz, float):
            try:
                mz = float(mz)
            except:
                raise InputError(
                    f"The precursor_mz parameter is {type(mz)} but needs to be an integer or a float."
                )
        if mz is None:
            raise InputError(f"The product_mz_end parameter is a None type.")
        self._product_mz_end = mz

    @property
    def user_info(self):
        """Create a dictionary with the necessary user information based on the class attributes."""
        # # first, some sanity checks
        # if len(self.files) != len(self.double_peak):
        #     raise InputError(
        #         f"The length of 'files' ({len(self.files)}) and of 'double_peak' ({len(self.double_peak)}) are not identical."
        #     )
        # if self.pre_filtering:
        #     # check length of lists
        #     if len(self.files) != len(self.pre_filtering) or len(self.double_peak) != len(
        #         self.retention_time_estimate
        #     ):
        #         raise InputError(
        #             f"The length of 'files' ({len(self.files)}), 'double_peak' ({self.double_peak}), and retention_time_estimate ({len(self.retention_time_estimate)}) are not identical."
        #         )
        # else:
        #     # if pre_filtering is False, then retention_time_estimate is not needed but the dictionary still needs to be created without errors -> set it to None
        #     if len(self.retention_time_estimate) == 1:
        #         self.retention_time_estimate = len(self.files) * None
        #     elif not self.retention_time_estimate:
        #         self.retention_time_estimate = len(self.files) * None
        # if any(self.retention_time_estimate) < 0:
        #     raise InputError("Retention time estimates below 0 are not valid.")
        # actually create the dictionary
        user_info = dict(zip(self.files, zip(self.double_peak, self.retention_time_estimate)))
        user_info["peak_width_estimate"] = self.peak_width_estimate
        user_info["pre_filtering"] = self.pre_filtering
        user_info["minimum_sn"] = self.minimum_sn
        return user_info


def detect_npy(path: Union[str, os.PathLike]):
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
        raise FileNotFoundError(f"In the given directory '{path}', there are no .npy files.")
    return npy_files


def scan_folder(path: Union[str, os.PathLike]):
    """
    Detect all files in a given directory and returns them as a list.

    The files should
    a) contain time and intensity data and
    b) be named according to the naming scheme (will automatically be correct when downloaded from the MS data cluster).

    Parameters
    ----------
    path
        Path to the folder containing raw data.
    """
    return os.listdir(path)


def parse_data(path: Union[str, os.PathLike], filename: str):
    """
    Extract names of data files.

    Use this in a for-loop with the data file names from detect_npy() or scan_folder().

    Parameters
    ----------
    path
        Path to the raw data files.
    filename
        Name of a raw date file containing a NumPy array with a time series (time as first, intensity as second element of the array).

    Returns
    -------
    timeseries
        NumPy Array containing time and intensity data as NumPy arrays at fist and second position, respectively.
    acquisition
        Name of a single acquisition.
    experiment
        Experiment number of the signal within the acquisition (each experiment = one mass trace).
    precursor_mz
        Mass to charge ratio of the precursor ion selected in Q1.
    product_mz_start
        Start of the mass to charge ratio range of the product ion in the TOF.
    product_mz_end
        End of the mass to charge ratio range of the product ion in the TOF.
    """
    # load time series
    timeseries = np.load(Path(path) / filename)
    # get information from the raw data file name
    splits = filename.split("_")
    acquisition = splits[0]
    experiment = splits[1]
    precursor_mz = splits[2]
    product_mz_start = splits[3]
    # remove the .npy suffix from the last split
    product_mz_end = splits[4][:-4]
    return timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end


def initiate(path: Union[str, os.PathLike]):
    """
    Create a folder for the results. Also create a zip file inside that folder. Also create df_summary.

    Parameters
    ----------
    path
        Path to the directory containing the raw data.

    Returns
    -------
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
    path
        Updated path variable pointing to the newly created folder for this batch.
    """
    # get current date and time
    today = str(date.today())
    now = datetime.now().strftime("%H-%M-%S")
    timestamp = today + "_" + now
    run_dir = timestamp + "_run"
    # create a directory
    path = Path(path) / run_dir
    path.mkdir(exist_ok=True)
    # # write text file, zip it, then delete it (cannot create an empty zip)
    # text_file = open(rf"{path}/readme.txt", "w")
    # txt = text_file.write(f"This batch was started on the {timestamp}.")
    # text_file.close()
    # with zipfile.ZipFile(rf"{path}/idata.zip", mode="w") as archive:
    #     archive.write(rf"./{run_dir}/readme.txt")
    # os.remove(rf"{path}/readme.txt")
    # create DataFrame for data report
    df_summary = pandas.DataFrame(
        columns=[
            "mean",
            "sd",
            "hdi_3%",
            "hdi_97%",
            "mcse_mean",
            "mcse_sd",
            "ess_bulk",
            "ess_tail",
            "r_hat",
            "acquisition",
            "experiment",
            "precursor_mz",
            "product_mz_start",
            "product_mz_end",
            "double_peak",
        ]
    )
    return df_summary, path


def prefiltering(
    filename: str, ui: UserInput, noise_width_guess: float, df_summary: pandas.DataFrame
):
    """
    Optional method to skip signals where clearly no peak is present. Saves a lot of computation time.

    Parameters
    ----------
    filename
        Name of the raw data file.
    ui
        Instance of the UserInput class
    noise_width_guess
        Estimated width of the noise of a particular measurement.

    Returns
    -------
    found_peak
        True, if any peak candidate was found within the time frame; False, if not.
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
    """
    # pre-fit tests for peaks to save computation time (optional)
    doublepeak = ui.user_info[filename][0]
    t_ret = ui.user_info[filename][1]
    est_width = ui.peak_width_estimate
    # find all potential peaks with scipy
    peaks, _ = scipy.signal.find_peaks(ui.timeseries[1])
    peak_candidates = []
    # differentiate between single and double peaks
    if not doublepeak:
        # single peaks
        for peak in peaks:
            # define conditions for passing the pre-filtering
            # check proximity of any peak candidate to the estimated retention time
            retention_time_condition = (
                t_ret - est_width <= ui.timeseries[0][peak] <= t_ret + est_width
            )
            # check signal to noise ratio
            signal_to_noise_condition = ui.timeseries[1][peak] / noise_width_guess > ui.minimum_sn
            # check the neighbouring data points to prevent classification of a single elevated data point as a peak
            check_preceding_point = ui.timeseries[1][peak - 1] / noise_width_guess > 2
            check_succeeding_point = ui.timeseries[1][peak + 1] / noise_width_guess > 2
            if (
                retention_time_condition
                and signal_to_noise_condition
                and check_preceding_point
                and check_succeeding_point
            ):
                peak_candidates.append(peak)
    else:
        # double peaks
        for peak in peaks:
            # define conditions for passing the pre-filtering
            # check proximity of any peak candidate to the estimated retention time
            retention_time_condition = (
                t_ret[0] - est_width <= ui.timeseries[0][peak] <= t_ret[0] + est_width
                or t_ret[1] - est_width <= ui.timeseries[0][peak] <= t_ret[1] + est_width
            )
            # check signal to noise ratio
            signal_to_noise_condition = ui.timeseries[1][peak] / noise_width_guess > ui.minimum_sn
            # check the neighbouring data points to prevent classification of a single elevated data point as a peak
            check_preceding_point = ui.timeseries[1][peak - 1] / noise_width_guess > 2
            check_succeeding_point = ui.timeseries[1][peak + 1] / noise_width_guess > 2
            if (
                retention_time_condition
                and signal_to_noise_condition
                and check_preceding_point
                and check_succeeding_point
            ):
                peak_candidates.append(peak)
    if not peak_candidates:
        df_summary = report_add_nan_to_summary(filename, ui, df_summary)
        return False, df_summary
    return True, df_summary


def sampling(pmodel, **sample_kwargs):
    """Performs sampling.

    Parameters
    ----------
    pmodel
        A pymc model.
    **kwargs
        The keyword arguments are used in pm.sample().
    tune
        Number of tuning samples (default = 2000).
    draws
        Number of samples after tuning (default = 2000).

    Returns
    -------
    idata
        Inference data object.
    """
    sample_kwargs.setdefault("tune", 2000)
    sample_kwargs.setdefault("draws", 2000)
    with pmodel:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(**sample_kwargs))
    return idata


def postfiltering(filename: str, idata, ui: UserInput, df_summary: pandas.DataFrame):
    """
    Method to filter out false positive peaks after sampling based on the obtained uncertainties of several peak parameters.

    Parameters
    ----------
    filename
        Name of the raw data file.
    idata
        Inference data object resulting from sampling.
    ui
        Instance of the UserInput class.
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.

    Returns
    -------
    acceptance
        True if the signal was accepted as a peak -> save data and continue with next signal.
        False if the signal was not accepted as a peak -> re-sampling with more tuning samples or discard signal.
    resample
        True: re-sample with more tuning samples, False: don't.
    discard
        True: discard sample.
    """
    # check whether convergence, i.e. r_hat <= 1.05, was not reached OR peak criteria were not met
    doublepeak = ui.user_info[filename][0]
    resample = False
    discard = False
    az_summary: pandas.DataFrame = az.summary(idata)
    if not doublepeak == True:
        # for single peak
        if (
            any(list(az_summary.loc[:, "r_hat"])) > 1.05
            or az_summary.loc["std", :]["mean"] <= 0.1
            or az_summary.loc["area", :]["sd"] > az_summary.loc["area", :]["mean"] * 0.2
            or az_summary.loc["height", :]["sd"] > az_summary.loc["height", :]["mean"] * 0.2
        ):
            # decide whether to discard signal or sample with more tune samples based on size of sigma parameter of normal distribution (std) and on the relative sizes of standard deviations of area and height
            if (
                az_summary.loc["std", :]["mean"] <= 0.1
                or az_summary.loc["area", :]["sd"] > az_summary.loc["area", :]["mean"] * 0.2
                or az_summary.loc["height", :]["sd"] > az_summary.loc["height", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                df_summary = report_add_nan_to_summary(filename, ui, df_summary)
                resample = False
                discard = True
            else:
                # r_hat failed but rest of post-fit check passed
                # sample again with more tune samples to possibly reach convergence yet
                resample = True
                discard = False
    else:
        # for double peak
        if (
            any(list(az_summary.loc[:, "r_hat"])) > 1.05
            or az_summary.loc["std", :]["mean"] <= 0.1
            or az_summary.loc["area", :]["sd"] > az_summary.loc["area", :]["mean"] * 0.2
            or az_summary.loc["height", :]["sd"] > az_summary.loc["height", :]["mean"] * 0.2
            or az_summary.loc["std2", :]["mean"] <= 0.1
            or az_summary.loc["area2", :]["sd"] > az_summary.loc["area2", :]["mean"] * 0.2
            or az_summary.loc["height2", :]["sd"] > az_summary.loc["height2", :]["mean"] * 0.2
        ):
            # Booleans to differentiate which peak is or is not detected
            double_not_found_first = False
            double_not_found_second = False
            # decide whether to discard signal or sample with more tune samples based on size of sigma parameter of normal distribution (std) and on the relative sizes of standard deviations of area and heigt
            if (
                az_summary.loc["std", :]["mean"] <= 0.1
                or az_summary.loc["area", :]["sd"] > az_summary.loc["area", :]["mean"] * 0.2
                or az_summary.loc["height", :]["sd"] > az_summary.loc["height", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                double_not_found_first = True
            if (
                az_summary.loc["std2", :]["mean"] <= 0.1
                or az_summary.loc["area2", :]["sd"] > az_summary.loc["area2", :]["mean"] * 0.2
                or az_summary.loc["height2", :]["sd"] > az_summary.loc["height2", :]["mean"] * 0.2
            ):
                # post-fit check failed
                # add NaN values to summary DataFrame
                double_not_found_second = True
            # if both peaks failed the r_hat and peak criteria tests, then continue
            if double_not_found_first and double_not_found_second:
                df_summary = report_add_nan_to_summary(filename, ui, df_summary)
                resample = False
                discard = True
            else:
                # r_hat failed but rest of post-fit check passed
                # sample again with more tune samples to possibly reach convergence yet
                resample = True
                discard = False
    return resample, discard, df_summary


def posterior_predictive_sampling(pmodel, idata):
    """Performs posterior predictive sampling for signals recognized as peaks.

    Parameters
    ----------
    pmodel
        A pymc model.
    idata
        Previously sampled inference data object.

    Returns
    -------
    idata
        Inference data object updated with the posterior predictive samples.
    """
    with pmodel:
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata


def report_save_idata(idata, ui: UserInput, filename: str):
    """
    Saves inference data object within a zip file.

    Parameters
    ----------
    idata
        Inference data object resulting from sampling.
    ui
        Instance of the UserInput class.
    filename
        Name of a raw date file containing a NumPy array with a time series (time as first, intensity as second element of the array).
    """
    # with zipfile.ZipFile(rf"{ui.path}/idata.zip", mode="a") as archive:
    #     archive.write(idata.to_netcdf(f"{filename[:-4]}"))
    idata.to_netcdf(rf"{ui.path}/{filename[:-4]}")
    return


def report_add_data_to_summary(filename: str, idata, df_summary: pandas.DataFrame, ui: UserInput):
    """
    Extracts the relevant information from idata, concatenates it to the summary DataFrame, and saves the DataFrame as an Excel file.
    Error handling prevents stop of the pipeline in case the saving doesn't work (e.g. because the file was opened by someone).

    Parameters
    ----------
    idata
        Inference data object resulting from sampling.
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
    ui
        Instance of the UserInput class.

    Returns
    -------
    df_summary
        Updated DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
    """
    az_summary: pandas.DataFrame = az.summary(idata)
    # split double peak into first and second peak (when extracting the data from az.summary(idata))
    if ui.user_info[filename][0]:
        # first peak of double peak
        parameters = [
            "baseline_intercept",
            "baseline_slope",
            "mean[0]",
            "noise",
            "std",
            "area",
            "height",
            "sn",
        ]
        df = az_summary.loc[parameters, :]
        df.rename(columns={"mean[0]": "mean"})
        df["acquisition"] = len(parameters) * [f"{ui.acquisition}"]
        df["experiment"] = len(parameters) * [ui.experiment]
        df["precursor_mz"] = len(parameters) * [ui.precursor_mz]
        df["product_mz_start"] = len(parameters) * [ui.product_mz_start]
        df["product_mz_end"] = len(parameters) * [ui.product_mz_end]
        df["double_peak"] = len(parameters) * ["1st"]

        # second peak of double peak
        parameters = [
            "baseline_intercept",
            "baseline_slope",
            "mean[1]",
            "noise",
            "std2",
            "area2",
            "height2",
            "sn2",
        ]
        df2 = az_summary.loc[parameters, :]
        df2.rename(
            columns={
                "area2": "area",
                "height2": "height",
                "sn2": "sn",
                "std2": "std",
                "mean[1]": "mean",
            }
        )
        df2["acquisition"] = len(parameters) * [f"{ui.acquisition}"]
        df2["experiment"] = len(parameters) * [f"{ui.experiment}"]
        df2["precursor_mz"] = len(parameters) * [f"{ui.precursor_mz}"]
        df2["product_mz_start"] = len(parameters) * [f"{ui.product_mz_start}"]
        df2["product_mz_end"] = len(parameters) * [f"{ui.product_mz_end}"]
        df2["double_peak"] = len(parameters) * ["2nd"]
        df_double = pandas.concat([df, df2])
        df_summary = pandas.concat([df_summary, df_double])

    else:
        # for single peak
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
        df = az_summary.loc[parameters, :]
        df["acquisition"] = len(parameters) * [f"{ui.acquisition}"]
        df["experiment"] = len(parameters) * [ui.experiment]
        df["precursor_mz"] = len(parameters) * [ui.precursor_mz]
        df["product_mz_start"] = len(parameters) * [ui.product_mz_start]
        df["product_mz_end"] = len(parameters) * [ui.product_mz_end]
        df["double_peak"] = len(parameters) * [""]
        df_summary = pandas.concat([df_summary, df])
    # pandas.concat(df_summary, df)
    # save summary df as Excel file
    with pandas.ExcelWriter(
        path=rf"{ui.path}/peak_data_summary.xlsx", engine="openpyxl", mode="w"
    ) as writer:
        df_summary.to_excel(writer)
    return df_summary


def report_area_sheet(path: Union[str, os.PathLike], df_summary: pandas.DataFrame):
    """
    Save a different, more minimalist report sheet focussing on the area data.

    Parameters
    ----------
    path
        Path to the directory containing the raw data.
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
    """
    # also save a version of df_summary only for areas with correct order and only necessary data
    df_area_summary = df_summary[df_summary.index == "area"]
    # TODO: test whether this still works with the new layout of the report sheet
    sorted_area_summary = df_area_summary.sort_values(
        ["acquisition", "precursor_mz", "product_mz_start"]
    )
    sorted_area_summary = sorted_area_summary.drop(
        labels=["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail"], axis=1
    )
    sorted_area_summary.to_excel(rf"{path}/area_summary.xlsx")
    return


def report_add_nan_to_summary(filename: str, ui: UserInput, df_summary: pandas.DataFrame):
    """
    Method to add NaN values to the summary DataFrame in case a signal did not contain a peak.

    Parameters
    ----------
    ui
        Instance of the UserInput class.
    df_summary
        DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.

    Returns
    -------
    df_summary
        Updated DataFrame for collecting the results (i.e. peak parameters) of every signal of a given pipeline.
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
    df["acquisition"] = len(df.index) * [f"{ui.acquisition}"]
    df["experiment"] = len(df.index) * [ui.experiment]
    df["precursor_mz"] = len(df.index) * [ui.precursor_mz]
    df["product_mz_start"] = len(df.index) * [ui.product_mz_start]
    df["product_mz_end"] = len(df.index) * [ui.product_mz_end]
    # if no peak was detected, there is no need for splitting double peaks, just give the info whether one was expected or not
    if ui.user_info[filename][0]:
        df["double_peak"] = len(df.index) * [True]
    else:
        df["double_peak"] = len(df.index) * [False]
    # concatenate to existing summary DataFrame
    df_summary = pandas.concat([df_summary, df])
    # save summary df as Excel file
    with pandas.ExcelWriter(
        path=rf"{ui.path}/peak_data_summary.xlsx", engine="openpyxl", mode="w"
    ) as writer:
        df_summary.to_excel(writer)
    return df_summary


def pipeline(path_raw_data: Union[str, os.PathLike], raw_data_file_format: str, pre_filtering: bool, double_peak: dict, retention_time_estimate: Dict[str, Union[float, int]], peak_width_estimate: Union[float, int], minimum_sn: Union[float, int]):
    """
    Method to run the complete Peak Performance pipeline.

    Parameters
    ----------
    path_raw_data
        Path to the raw data files. Files should be in the given raw_data_file_format, default is '.npy'. 
        In any case, time and intensity have to be saved as NumPy arrays at the first and second position of the stored data object, respectively.
    raw_data_file_format
        Data format (suffix) of the raw data, default is '.npy'.
    pre_filtering
        Select whether to include (True) or exclude (False) the pre-filtering step. Pre-filtering checks for peaks based on retention time and signal-to-noise ratio before fitting/sampling to potentially save a lot of computation time.
        If True is selected, specification of the parameters retention_time_estimate, peak_width_estimate, and minimum_sn is mandatory.
    double_peak
        Dictionary with the raw data file names as keys and Booleans as values. 
        Set to True for a given signal, if the signal contains a double peak, and set to False, if it contains a single peak. Visually check this beforehand.
    retention_time_estimate
        Dictionary with the raw data file names as keys and floats or ints of the expected retention time of the given analyte as values.
        In case you set pre_filtering to True, give a retention time estimate (float or int) for each signal. In case of a double peak, give two retention times (in chronological order) as a tuple containing two floats or ints.
    peak_width_estimate
        In case you set pre_filtering to True, give a rough estimate of the average peak width in minutes you would expect for your LC-MS/MS method.
    minimum_sn
        In case you set pre_filtering to True, give a minimum signal to noise ratio for a signal to be defined as a peak during pre-filtering.
    """ 
    # obtain a list of raw data file names.
    raw_data_files = detect_npy(path_raw_data)
    # create data structure and DataFrame(s) for results 
    df_summary, path_results = initiate(path_raw_data)
    for file in raw_data_files:
        # parse the data and extract information from the (standardized) file name
        timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end = parse_data(path_raw_data, file)
        # instantiate the UserInput class all given information
        ui = UserInput(path_results, raw_data_files, double_peak, retention_time_estimate, peak_width_estimate, pre_filtering, minimum_sn, timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end)
        # calculate initial guesses for pre-filtering and defining prior probability distributions
        slope_guess, intercept_guess, noise_guess = models.initial_guesses(ui.timeseries[0], ui.timeseries[1])
        # apply pre-sampling filter (if selected)
        if pre_filtering:
            prefilter, df_summary = prefiltering(file, ui, noise_guess, df_summary)
            if not prefilter:
                # if no peak candidates were found, continue with the next signal
                plots.plot_raw_data(file, ui)
                continue
        # model selection
        if ui.user_info[file][0]:
            # double peak model
            pmodel = models.define_model_doublepeak(ui)
        else:
            # single peaks are first modeled with a skew normal distribution
            pmodel = models.define_model_skew(ui)
        # sample the chosen model
        idata = sampling(pmodel)
        # apply post-sampling filter
        resample, discard, df_summary = postfiltering(file, idata, ui, df_summary)
        # if peak was discarded, continue with the next signal
        if discard:
            plots.plot_posterior(file, ui, idata, True)
            print("discarded")
            continue
        # if convergence was not yet reached, sample again with more tuning samples
        if resample:
            print("start resampling")
            idata = sampling(pmodel, tune = 4000)
            resample, discard, df_summary = postfiltering(file, idata, ui, df_summary)
            if discard:
                plots.plot_posterior(f"{file}", ui, idata, True)
                continue
            if resample:
                # if signal was flagged for re-sampling a second time, discard it
                # TODO: should this really be discarded or should the contents of idata be added with an additional comment? (would need to add a comment column)
                df_summary = report_add_nan_to_summary(file, ui, df_summary)
                plots.plot_posterior(f"{file}", ui, idata, True)
                continue
        # add inference data to df_summary and save it as an Excel file
        df_summary = report_add_data_to_summary(file, idata, df_summary, ui)
        # perform posterior predictive sampling
        idata = posterior_predictive_sampling(pmodel, idata)
        # save the inference data object in a zip file
        report_save_idata(idata, ui, file)
        # plot data
        plots.plot_posterior_predictive(file, ui, idata, False)
        plots.plot_posterior(file, ui, idata, False)
    # save condesed Excel file with area data
    report_area_sheet(path_results, df_summary)
    return
