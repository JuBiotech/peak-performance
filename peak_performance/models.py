import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as st


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
    Model a peak shaped like the PDF of a normal distribution.

    Parameters
    ----------
    baseline
        Baseline of the data.
    height
        Height of the normal distribution (starting from the baseline, thus not the total height).
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


def define_model_normal(ui) -> pm.Model:
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
    slope_guess, intercept_guess, noise_width_guess = initial_guesses(time, intensity)
    with pm.Model() as pmodel:
        # add observations to the pmodel as ConstantData
        pm.ConstantData("time", time)
        pm.ConstantData("intensity", intensity)
        # add guesses to the pmodel as ConstantData
        pm.ConstantData("intercept_guess", intercept_guess)
        pm.ConstantData("slope_guess", slope_guess)
        pm.ConstantData("noise_width_guess", noise_width_guess)

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
        pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))))
        pm.Deterministic("sn", height / noise)
        # posterior
        y = normal_posterior(baseline, height, time, mean, std)
        y = pm.Deterministic("y", y)

        # likelihood
        pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel


def double_normal_posterior(baseline, height, time: np.ndarray, mean, std):
    """
    Define a univariate ordered normal distribution as the posterior.

    Parameters
    ----------
    baseline
        Baseline of the data.
    height
        Height of the first and second peak.
    time
        NumPy array with the time values of the relevant timeframe.
    mean
        Arithmetic mean of the normal distribution.
    std
        Standard deviation of the first and second peak.

    Returns
    -------
    y
        Probability density function (PDF) of a univariate ordered normal distribution as the posterior.
    """
    y = (
        baseline
        + height[0] * pt.exp(-0.5 * ((time - mean[0]) / std[0]) ** 2)
        + height[1] * pt.exp(-0.5 * ((time - mean[1]) / std[1]) ** 2)
    )
    return y


def define_model_doublepeak(ui) -> pm.Model:
    """
    Define a model for fitting two ordered normal distributions to the peak data
    (for when data contains two peaks or a double peak without baseline separation).

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
    slope_guess, intercept_guess, noise_width_guess = initial_guesses(time, intensity)
    coords = {"subpeak":["left", "right"]}
    with pm.Model(coords=coords) as pmodel:
        # add observations to the pmodel as ConstantData
        pm.ConstantData("time", time)
        pm.ConstantData("intensity", intensity)
        # add guesses to the pmodel as ConstantData
        pm.ConstantData("intercept_guess", intercept_guess)
        pm.ConstantData("slope_guess", slope_guess)
        pm.ConstantData("noise_width_guess", noise_width_guess)

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
        std = pm.HalfNormal(
            "std",
            sigma=[np.ptp(time) / 3, np.ptp(time) / 3.5],
        )
        height = pm.HalfNormal(
            "height",
            sigma=[0.95 * np.max(intensity), 0.96 * np.max(intensity)],
        )
        pm.Deterministic("area", height / (1 / (std * np.sqrt(2 * np.pi))), dims=("subpeak",))
        pm.Deterministic("sn", height / noise, dims=("subpeak",))
        # use univariate ordered normal distribution
        mean = pm.Normal(
            "mean",
            mu=[time[0] + np.ptp(time) * 1 / 4, time[0] + np.ptp(time) * 3 / 4],
            sigma=1,
            transform=pm.distributions.transforms.univariate_ordered,
        )

        # posterior
        y = double_normal_posterior(baseline, height, time, mean, std)
        y = pm.Deterministic("y", y)

        # likelihood
        pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel


def std_skew_calculation(scale, alpha):
    """
    Calculate the standard deviation of a skew normal distribution with f(x | loc, scale, alpha).

    Parameters
    ----------
    scale
        Scale parameter of the skew normal distribution.
    alpha
        Skewness parameter of the skew normal distribution.

    Returns
    ----------
    std
        Standard deviation of a skew normal distribution.
    -------
    """
    return np.sqrt(scale**2 * (1 - (2 * alpha**2) / ((alpha**2 + 1) * np.pi)))


def mean_skew_calculation(loc, scale, alpha):
    """
    Calculate the arithmetic mean of a skew normal distribution with f(x | loc, scale, alpha).

    Parameters
    ----------
    loc
        Location parameter of the skew normal distribution.
    scale
        Scale parameter of the skew normal distribution.
    alpha
        Skewness parameter of the skew normal distribution.

    Returns
    ----------
    mean
        Arithmetic mean of a skew normal distribution.
    """
    return loc + scale * np.sqrt(2 / np.pi) * alpha / (np.sqrt(1 + alpha**2))


def delta_calculation(alpha):
    """
    Calculate the delta term included in several subsequent formulae.

    Parameters
    ----------
    alpha
        Skewness parameter of the skew normal distribution.
    """
    return alpha / (np.sqrt(1 + alpha**2))


def mue_z_calculation(delta):
    """Calculate the mue_z variable which is needed to compute a numerical approximation of the mode of a skew normal distribution."""
    return np.sqrt(2 / np.pi) * delta


def sigma_z_calculation(mue_z):
    """Calculate the sigma_z variable which is needed to compute a numerical approximation of the mode of a skew normal distribution."""
    return np.sqrt(1 - mue_z**2)


def skewness_calculation(delta):
    """Calculate the skewness of a skew normal distribution."""
    return (
        (4 - np.pi)
        / 2
        * ((delta * np.sqrt(2 / np.pi)) ** 3)
        / ((1 - 2 * delta**2 / np.pi) ** 1.5)
    )


def mode_offset_calculation(mue_z, skewness, sigma_z, alpha):
    """Calculate the offset between arithmetic mean and mode of a skew normal distribution."""
    # this formula originally contained the sign() function which led to an error due to usage of pytensor variables
    # -> use alpha/abs(alpha) instead for the same effect
    return (
        mue_z
        - (skewness * sigma_z) / 2
        - (alpha / abs(alpha)) / 2 * pt.exp(-(2 * np.pi) / abs(alpha))
    )


def mode_skew_calculation(loc, scale, mode_offset):
    """Calculate a numerical approximation of the mode of a skew normal distribution."""
    return loc + scale * mode_offset


def height_calculation(area, loc, scale, alpha, mode_skew):
    """
    Calculate the height of a skew normal distribution.
    The formula is the result of inserting time = mode_skew into the posterior.

    Parameters
    ----------
    area
        Area of the peak described by the skew normal distribution (area between baseline and skew normal distribution).
    loc
        Location parameter of the skew normal distribution.
    scale
        Scale parameter of the skew normal distribution.
    alpha
        Skewness parameter of the skew normal distribution.
    mode_skew
        Mode of the skew normal distribution.

    Returns
    ----------
    mean
        Arithmetic mean of a skew normal distribution.
    """
    return area * (
        2
        * (1 / (scale * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((mode_skew - loc) / scale) ** 2))
        * (0.5 * (1 + pt.erf(((alpha * (mode_skew - loc) / scale)) / np.sqrt(2))))
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


def define_model_skew(ui) -> pm.Model:
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
    slope_guess, intercept_guess, noise_width_guess = initial_guesses(time, intensity)
    with pm.Model() as pmodel:
        # add observations to the pmodel as ConstantData
        pm.ConstantData("time", time)
        pm.ConstantData("intensity", intensity)
        # add guesses to the pmodel as ConstantData
        pm.ConstantData("intercept_guess", intercept_guess)
        pm.ConstantData("slope_guess", slope_guess)
        pm.ConstantData("noise_width_guess", noise_width_guess)

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
        alpha = pm.Normal("alpha", 0, 3.5)
        area = pm.HalfNormal("area", np.max(intensity) * 0.9)

        # calculate standard deviation and arithmetic mean of a skew normal distribution
        std_skew_formula = std_skew_calculation(std, alpha)
        pm.Deterministic("std_skew", std_skew_formula)
        mean_skew_formula = mean_skew_calculation(mean, std, alpha)
        pm.Deterministic("mean_skew", mean_skew_formula)

        # height is defined as the posterior with x = mode
        delta_formula = delta_calculation(alpha)
        delta = pm.Deterministic("delta", delta_formula)
        mue_z_formula = mue_z_calculation(delta)
        mue_z = pm.Deterministic("mue_z", mue_z_formula)
        sigma_z_formula = sigma_z_calculation(mue_z)
        sigma_z = pm.Deterministic("sigma_z", sigma_z_formula)
        skewness = skewness_calculation(delta)
        mode_offset_formula = mode_offset_calculation(mue_z, skewness, sigma_z, alpha)
        mode_offset = pm.Deterministic("mode_offset", mode_offset_formula)
        mode_skew_formula = mode_skew_calculation(mean, std, mode_offset)
        mode_skew = pm.Deterministic("mode_skew", mode_skew_formula)
        # then calculate the height based on the mode
        height_formula = height_calculation(area, mean, std, alpha, mode_skew)
        height = pm.Deterministic(
            "height",
            height_formula,
        )
        pm.Deterministic("sn", height / noise)
        y = skew_normal_posterior(baseline, area, time, mean, std, alpha)
        y = pm.Deterministic("y", y)

        # likelihood
        pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel


def double_skew_normal_posterior(baseline, area, time: np.ndarray, mean, std, alpha):
    """
    Define a univariate ordered skew normal distribution as the posterior.

    Parameters
    ----------
    baseline
        Baseline of the data.
    area
        Area of the first and second peak.
    time
        NumPy array with the time values of the relevant timeframe.
    mean
        Location parameter.
    std
        Scale parameter of the first and second peak.
    alpha
        Skewness parameter of the first and second peak.

    Returns
    -------
    y
        Probability density function (PDF) of a univariate ordered normal distribution as the posterior.
    """
    y = (
        baseline
        + area[0]
        * (
            2
            * (1 / (std[0] * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((time - mean[0]) / std[0]) ** 2))
            * (0.5 * (1 + pt.erf(((alpha[0] * (time - mean[0]) / std[0])) / np.sqrt(2))))
        )
        + area[1]
        * (
            2
            * (1 / (std[1] * np.sqrt(2 * np.pi)) * pt.exp(-0.5 * ((time - mean[1]) / std[1]) ** 2))
            * (0.5 * (1 + pt.erf(((alpha[1] * (time - mean[1]) / std[1])) / np.sqrt(2))))
        )
    )
    return y


def define_model_double_skew(ui) -> pm.Model:
    """
    Define a model for fitting two ordered skew normal distributions to the peak data
    (for when data contains two peaks or a double peak without baseline separation).

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
    slope_guess, intercept_guess, noise_width_guess = initial_guesses(time, intensity)
    coords = {"subpeak":["left", "right"]}
    with pm.Model(coords=coords) as pmodel:
        # add observations to the pmodel as ConstantData
        pm.ConstantData("time", time)
        pm.ConstantData("intensity", intensity)
        # add guesses to the pmodel as ConstantData
        pm.ConstantData("intercept_guess", intercept_guess)
        pm.ConstantData("slope_guess", slope_guess)
        pm.ConstantData("noise_width_guess", noise_width_guess)

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
        # use univariate ordered skew normal distribution
        mean = pm.Normal(
            "mean",
            mu=[time[0] + np.ptp(time) * 1 / 4, time[0] + np.ptp(time) * 3 / 4],
            sigma=1,
            transform=pm.distributions.transforms.univariate_ordered,
        )
        std = pm.HalfNormal(
            "std",
            sigma=[np.ptp(time) / 3, np.ptp(time) / 3],
        )
        area = pm.HalfNormal(
            "area",
            sigma=[np.max(intensity) * 0.9, np.max(intensity) * 0.9],
        )
        alpha = pm.Normal(
            "alpha",
            mu=[0, 0],
            sigma=3.5,
        )

        # height is defined as the posterior with x = mode
        delta_formula = delta_calculation(alpha)
        delta = pm.Deterministic("delta", delta_formula)
        mue_z_formula = mue_z_calculation(delta)
        mue_z = pm.Deterministic("mue_z", mue_z_formula)
        sigma_z_formula = sigma_z_calculation(mue_z)
        sigma_z = pm.Deterministic("sigma_z", sigma_z_formula)
        skewness = skewness_calculation(delta)
        mode_offset_formula = mode_offset_calculation(mue_z, skewness, sigma_z, alpha)
        mode_offset = pm.Deterministic("mode_offset", mode_offset_formula)
        mode_skew_formula = mode_skew_calculation(mean, std, mode_offset)
        mode_skew = pm.Deterministic("mode_skew", mode_skew_formula)
        # then calculate the height based on the mode
        height_formula = height_calculation(area, mean, std, alpha, mode_skew)
        pm.Deterministic(
            "height",
            height_formula,
        )

        # posterior
        y = double_skew_normal_posterior(baseline, area, time, mean, std, alpha)
        y = pm.Deterministic("y", y)

        # likelihood
        pm.Normal("L", mu=y, sigma=noise, observed=intensity)

    return pmodel
