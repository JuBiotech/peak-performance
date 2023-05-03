import json
import math
import time

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


def initial_guesses(time_np, intensity_np):
    """
    Provide initial guesses for priors.
    """
    # select lowest third of all data points as noise -> noise_tuple
    intensity_tuple = list(enumerate(intensity_np))
    intensity_tuple.sort(key=lambda x: x[1])
    noise_range = int(np.round(1 / 3 * len(intensity_np), decimals=0))
    noise_tuple = intensity_tuple[:noise_range]
    # sort noise_tuple by time, then use the first and last data points for estimating starting values for the priors of noise width and the slope and intercept of baseline
    noise_tuple.sort(key=lambda x: x[0])
    slope_guess = (
        np.mean([noise_tuple[n][1] for n in range(len(noise_tuple) - 5, len(noise_tuple))])
        - np.mean([noise_tuple[n][1] for n in range(5)])
    ) / (time_np[-1] - time_np[0])
    # calculate intercept_guess based on the slope_guess and the formula for a linear equation
    intercept_guess = np.mean([noise_tuple[n][1] for n in range(5)]) - slope_guess * time_np[0]
    noise_width_guess = np.max([noise_tuple[n][1] for n in range(len(noise_tuple))]) - np.min(
        [noise_tuple[n][1] for n in range(len(noise_tuple))]
    )
    # TODO: somehow save the guesses? Or discard them, after all?

    return intercept_guess, slope_guess, noise_width_guess


def define_model_normal(time_np, intensity_np):
    """
    Define a model for fitting a normal distribution to the peak data.
    """
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time_np, intensity_np)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time_np)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        mean = pm.Normal("mean", np.mean(time_np[[0, -1]]), np.ptp(time_np) / 2)
        std = pm.HalfNormal("std", np.ptp(time_np) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity_np))
        area = pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))))
        sn = pm.Deterministic("sn", height / noise)
        # posterior
        y = baseline + height * pt.exp(-0.5 * ((time_np - mean) / std) ** 2)
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity_np)

    return pmodel


def define_model_doublepeak(time_np, intensity_np):
    """
    Define a model for fitting two ordered normal distributions to the peak data (for when data contains two peaks or a double peak without baseline separation).
    """
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time_np, intensity_np)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time_np)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        std = pm.HalfNormal("std", np.ptp(time_np) / 3)
        std2 = pm.HalfNormal("std2", np.ptp(time_np) / 3)
        height = pm.HalfNormal("height", 0.95 * np.max(intensity_np))
        height2 = pm.HalfNormal("height2", 0.95 * np.max(intensity_np))
        area = pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))))
        area2 = pm.Deterministic("area2", height2 / (1 / (std2 * np.sqrt(2 * np.pi))))
        # use univariate ordered normal distribution
        mean = pm.Normal(
            "mean",
            mu=[time_np[0] + np.ptp(time_np) * 1 / 4, time_np[0] + np.ptp(time_np) * 3 / 4],
            sigma=1,
            transform=pm.distributions.transforms.univariate_ordered,
        )

        # posterior
        y = (
            baseline
            + height * pt.exp(-0.5 * ((time_np - mean[0]) / std) ** 2)
            + height2 * pt.exp(-0.5 * ((time_np - mean[1]) / std2) ** 2)
        )
        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity_np)

    return pmodel


def define_model_skew(time_np, intensity_np):
    """
    Define a model for fitting a skew normal distribution to the peak data.
    """
    intercept_guess, slope_guess, noise_width_guess = initial_guesses(time_np, intensity_np)
    with pm.Model() as pmodel:
        # priors plus error handling in case of mathematically impermissible values
        if intercept_guess == 0:
            baseline_intercept = pm.Normal("baseline_intercept", intercept_guess, 20)
        else:
            baseline_intercept = pm.Normal(
                "baseline_intercept", intercept_guess, abs(intercept_guess) / 2
            )
        baseline_slope = pm.Normal("baseline_slope", slope_guess, abs(slope_guess * 2) + 1)
        baseline = pm.Deterministic("baseline", baseline_intercept + baseline_slope * time_np)
        # since log(0) leads to -inf, this case is handled by setting noise_width_guess to 10
        if noise_width_guess > 0:
            noise = pm.LogNormal("noise", np.log(noise_width_guess), 1)
        elif noise_width_guess == 0:
            noise = pm.LogNormal("noise", np.log(10), 1)
        mean = pm.Normal("mean", np.mean(time_np[[0, -1]]), np.ptp(time_np) / 2)
        std = pm.HalfNormal("std", np.ptp(time_np) / 3)
        alpha = pm.HalfNormal("alpha", 2.5)
        area = pm.HalfNormal("area", np.max(intensity_np) * 0.9)
        # calculate standard deviation of skew normal
        std_skew = pm.Deterministic(
            "std_skew", np.sqrt(std**2 * (1 - (2 * alpha**2) / ((alpha**2 + 1) * np.pi)))
        )

        # height is defined as the posterior with x = mode
        # (difference to normal distribution: for normal distribution mean and mode are identical and inserting x = mean = mode leads to a simplification of the PDF)
        # first calculate the mode (via calculating the mean of a skew normal and using a numerical approach to calculating the offset between mean and mode)
        mean_skew = pm.Deterministic(
            "mean_skew", mean + std * np.sqrt(2 / np.pi) * alpha / (np.sqrt(1 + alpha**2))
        )
        mue_z = pm.Deterministic("mue_z", np.sqrt(2 / np.pi) * alpha / (np.sqrt(1 + alpha**2)))
        sigma_z = pm.Deterministic("sigma_z", np.sqrt(1 - mue_z**2))
        fit_skewness = st.skew(intensity_np)
        # this formula originally contained the sign() function which led to an error -> use alpha/abs(alpha) instead for the same effect
        mode_offset = pm.Deterministic(
            "mode_offset",
            mue_z
            - (fit_skewness * sigma_z) / 2
            - (alpha / abs(alpha)) / 2 * pt.exp(-(2 * np.pi) / abs(alpha)),
        )
        # if alpha < 0: mode = mean + offset; if alpha > 0: mode = mean - offset;
        mode_skew = pm.Deterministic("mode_skew", mean_skew - (alpha / abs(alpha)) * mode_offset)
        # then calculate the height based on the mode
        pm.Deterministic(
            "height",
            area
            * (
                2
                * (1 / (std * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((mode_skew - mean) / std) ** 2))
                * (0.5 * (1 + pt.erf(((alpha * (mode_skew - mean) / std)) / np.sqrt(2))))
            ),
        )

        # posterior
        y = baseline + area * (
            2
            * (1 / (std * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((time_np - mean) / std) ** 2))
            * (0.5 * (1 + pt.erf(((alpha * (time_np - mean) / std)) / np.sqrt(2))))
        )

        y = pm.Deterministic("y", y)

        # likelihood
        L = pm.Normal("L", mu=y, sigma=noise, observed=intensity_np)
    
    return pmodel
