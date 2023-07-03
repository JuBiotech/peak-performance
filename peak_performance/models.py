import math

import arviz as az
import numpy as np
import pandas
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as st

from . import pipeline as pi


def initial_guesses(time: np.ndarray, intensity: np.ndarray):
    """
    Provide initial guesses for priors.

    Parameters
    ----------
    time
        NumPy array with the time values of the relevant timeframe.
    intensity
        NumPy array with the intensity values of the relevant timeframe.

    Returns
    -------
    baseline_fit.slope : float or int
        Guess for the slope of the linear baseline prior.
    baseline_fit.intercept : float or int
        Guess for the intercept of the linear baseline prior.
    noise_width_guess : float or int
        Guess for the width of the noise.
    """
    # first create a simple baseline guess only to be able to "correct" the intensity data (y_corrected = y - y_baseline)
    # then, use the corrected data to determine which data points are going to be defined as noise
    # this is the only use of the corrected data
    average_initial_intensity = np.mean([intensity[n] for n in range(3)])
    average_final_intensity = np.mean(
        [intensity[n] for n in range(len(intensity) - 3, len(intensity))]
    )
    slope_guess = (average_final_intensity - average_initial_intensity) / (time[-1] - time[0])
    # calculate intercept_guess based on the slope_guess and the formula for a linear equation
    first_intercept_guess = average_initial_intensity - slope_guess * time[0]
    intensity_corrected = [
        intensity[n] - (slope_guess * time[n] + first_intercept_guess) for n in range(len(time))
    ]

    # select lowest 35 % of all data points as noise -> noise_tuple
    intensity_tuple = list(enumerate(intensity_corrected))
    intensity_tuple.sort(key=lambda x: x[1])
    noise_range = int(np.round(0.35 * len(intensity_corrected), decimals=0))
    noise_tuple = intensity_tuple[:noise_range]
    noise_index = sorted([x[0] for x in noise_tuple])
    # use the indeces in noise_index to get the time and intensity of all noise data points
    noise_time = [time[n] for n in noise_index]
    noise_intensity = [intensity[n] for n in noise_index]
    # calculate the width of the noise
    noise_width_guess = max(noise_intensity) - min(noise_intensity)

    # use scipy to fit a linear regression through the noise as a prior for the eventual baseline
    baseline_fit = st.linregress(noise_time, noise_intensity)

    return baseline_fit.slope, baseline_fit.intercept, noise_width_guess


def normal_posterior(baseline, height, time: np.ndarray, mean, std):
    """
    Define a normally distributed posterior.

    Parameters
    ----------
    baseline
        Baseline of the data.
    height
        Height of the normal distribution.
    time
        NumPy array with the time values of the relevant timeframe.
    mean
        Arithmetic mean of the normal distribution.
    std
        Standard deviation of the normal distribution.

    Returns
    -------
    Probability density function (PDF) of the normally distributed posterior.
    """
    return baseline + height * pt.exp(-0.5 * ((time - mean) / std) ** 2)


def define_model_normal(ui: pi.UserInput) -> pm.Model:
    """
    Define a model for fitting a normal distribution to the peak data.

    Parameters
    ----------
    ui
        Instance of the UserInput class.

    Returns
    -------
    pmodel
        PyMC model.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time, intensity)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        # define priors for parameters of a normally distributed posterior
        mean = pm.Normal("mean", np.mean(time[[0, -1]]), np.ptp(time) / 2)
        std = pm.HalfNormal("std", np.ptp(time) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity))
        area = pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))))
        sn = pm.Deterministic("sn", height / noise)
        # posterior
        y = normal_posterior(baseline, height, time, mean, std)
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel


def double_normal_posterior(baseline, height, height2, time: np.ndarray, mean, std, std2):
    """
    Define a univariate ordered normal distribution as the posterior.

    Parameters
    ----------
    baseline
        Baseline of the data.
    height
        Height of the first peak.
    height2
        Height of the second peak.
    time
        NumPy array with the time values of the relevant timeframe.
    mean
        Arithmetic mean of the normal distribution.
    std
        Standard deviation of the first peak.
    std2
        Standard deviation of the second peak.

    Returns
    -------
    y
        Probability density function (PDF) of a univariate ordered normal distribution as the posterior.
    """
    y = (
        baseline
        + height * pt.exp(-0.5 * ((time - mean[0]) / std) ** 2)
        + height2 * pt.exp(-0.5 * ((time - mean[1]) / std2) ** 2)
    )
    return y


def define_model_doublepeak(ui: pi.UserInput) -> pm.Model:
    """
    Define a model for fitting two ordered normal distributions to the peak data (for when data contains two peaks or a double peak without baseline separation).

    Parameters
    ----------
    ui
        Instance of the UserInput class.

    Returns
    -------
    pmodel
        Pymc model.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time, intensity)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        std = pm.HalfNormal("std", np.ptp(time) / 3)
        std2 = pm.HalfNormal("std2", np.ptp(time) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity))
        height2 = pm.HalfNormal("height2", 0.95 * np.max(intensity))
        area = pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))))
        area2 = pm.Deterministic("area2", height2 / (1 / (std2 * np.sqrt(2 * np.pi))))
        sn = pm.Deterministic("sn", height / noise)
        sn2 = pm.Deterministic("sn2", height2 / noise)
        # use univariate ordered normal distribution
        mean = pm.Normal(
            "mean",
            mu=[time[0] + np.ptp(time) * 1 / 4, time[0] + np.ptp(time) * 3 / 4],
            sigma=1,
            transform=pm.distributions.transforms.univariate_ordered,
        )

        # posterior
        y = double_normal_posterior(baseline, height, height2, time, mean, std, std2)
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel


def std_skew_calculation(std, alpha):
    """Calculate the standard deviation of a skew normal distribution."""
    return np.sqrt(std**2 * (1 - (2 * alpha**2) / ((alpha**2 + 1) * np.pi)))


def mean_skew_calculation(mean, std, alpha):
    """Calculate the arithmetic mean of a skew normal distribution."""
    return mean + std * np.sqrt(2 / np.pi) * alpha / (np.sqrt(1 + alpha**2))


def mue_z_calculation(alpha):
    """Calculate the mue_z variable which is needed to compute a numerical approximation of the mode of a skew normal distribution."""
    return np.sqrt(2 / np.pi) * alpha / (np.sqrt(1 + alpha**2))


def sigma_z_calculation(mue_z):
    """Calculate the sigma_z variable which is needed to compute a numerical approximation of the mode of a skew normal distribution."""
    return np.sqrt(1 - mue_z**2)


def fit_skewness_calculation(intensity):
    """Calculate the skewness of a skew normal distribution via scipy."""
    return st.skew(intensity)


def mode_offset_calculation(mue_z, fit_skewness, sigma_z, alpha):
    """Calculate the offset between arithmetic mean and mode of a skew normal distribution."""
    return (
        mue_z
        - (fit_skewness * sigma_z) / 2
        - (alpha / abs(alpha)) / 2 * pt.exp(-(2 * np.pi) / abs(alpha))
    )


def mode_skew_calculation(mean_skew, mode_offset, alpha):
    """Calculate a numerical approximation of the mode of a skew normal distribution."""
    return mean_skew - (alpha / abs(alpha)) * mode_offset


def height_calculation(area, mean, std, alpha, mode_skew):
    """
    Calculate the height of a skew normal distribution.
    The formula is the result of inserting time = mode_skew into the posterior. Since the mode of a skew normal distribution is calculated as a numerical approximation,
    its accuracy is not perfect and thus the height's either. In tests, the height was still accurate up to and including the first two decimals.
    """
    return area * (
        2
        * (1 / (std * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((mode_skew - mean) / std) ** 2))
        * (0.5 * (1 + pt.erf(((alpha * (mode_skew - mean) / std)) / np.sqrt(2))))
    )


def skew_normal_posterior(baseline, area, time, mean, std, alpha):
    """
    Define a skew normally distributed posterior.

    Parameters
    ----------
    baseline
        Baseline of the data.
    area
        Peak area.
    time
        NumPy array with the time values of the relevant timeframe.
    intensity
        NumPy array with the intensity values of the relevant timeframe.
    mean
        Location parameter, i.e. arithmetic mean.
    std
        Scale parameter, i.e. standard deviation.
    alpha
        Skewness parameter.

    Returns
    -------
    y
        Probability density function (PDF) of a univariate ordered normal distribution as the posterior.
    """
    # posterior
    y = baseline + area * (
        2
        * (1 / (std * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((time - mean) / std) ** 2))
        * (0.5 * (1 + pt.erf(((alpha * (time - mean) / std)) / np.sqrt(2))))
    )
    return y


def define_model_skew(ui: pi.UserInput) -> pm.Model:
    """
    Define a model for fitting a skew normal distribution to the peak data.

    Parameters
    ----------
    ui
        Instance of the UserInput class.

    Returns
    -------
    pmodel
        Pymc model.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time, intensity)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        mean = pm.Normal("mean", np.mean(time[[0, -1]]), np.ptp(time) / 2)
        std = pm.HalfNormal("std", np.ptp(time) / 3)
        alpha = pm.HalfNormal("alpha", 2.5)
        area = pm.HalfNormal("area", np.max(intensity) * 0.9)
        # calculate standard deviation and arithmetic mean of a skew normal distribution
        std_skew_formula = std_skew_calculation(std, alpha)
        pm.Deterministic("std_skew", std_skew_formula)
        # height is defined as the posterior with x = mode
        # (difference to normal distribution: for normal distribution mean and mode are identical and inserting x = mean = mode leads to a simplification of the PDF)
        # first calculate the mode (via calculating the mean of a skew normal and using a numerical approach to calculating the offset between mean and mode)
        mean_skew_formula = mean_skew_calculation(mean, std, alpha)
        mean_skew = pm.Deterministic("mean_skew", mean_skew_formula)
        mue_z_formula = mue_z_calculation(alpha)
        mue_z = pm.Deterministic("mue_z", mue_z_formula)
        sigma_z_formula = sigma_z_calculation(mue_z)
        sigma_z = pm.Deterministic("sigma_z", sigma_z_formula)
        fit_skewness = fit_skewness_calculation(intensity)
        mode_offset_formula = mode_offset_calculation(mue_z, fit_skewness, sigma_z, alpha)
        # this formula originally contained the sign() function which led to an error -> use alpha/abs(alpha) instead for the same effect
        mode_offset = pm.Deterministic("mode_offset", mode_offset_formula)
        mode_skew_formula = mode_skew_calculation(mean_skew, mode_offset, alpha)
        # if alpha < 0: mode = mean + offset; if alpha > 0: mode = mean - offset;
        mode_skew = pm.Deterministic("mode_skew", mode_skew_formula)
        # then calculate the height based on the mode
        height_formula = height_calculation(area, mean, std, alpha, mode_skew)
        height = pm.Deterministic(
            "height",
            height_formula,
        )
        sn = pm.Deterministic("sn", height / noise)
        y = skew_normal_posterior(baseline, area, time, mean, std, alpha)
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel
