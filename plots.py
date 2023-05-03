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


def plot_raw_data(identifier, time_np, intensity_np):
    """
    Plot just the raw data in case no peak was found.

    Parameters
    ----------
    identifier
        unique identifier of this particular signal

    time_np
        numpy array with the time values of the relevant timeframe

    intensity_np
        numpy array with the intensity values of the relevant timeframe
    """
    # plot the data to be able to check if peak detection was correct or not
    fig, ax = pyplot.subplots()
    ax.scatter(time_np, intensity_np, marker="x", color="black", label="data")
    pyplot.legend()
    ax.set_xlabel("time / min", fontsize=12, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=12, fontweight="bold")
    pyplot.xticks(size=11.5)
    pyplot.yticks(size=11.5)
    pyplot.tight_layout()
    pyplot.savefig(rf"./230427_Pipeline_test_Part1/{identifier}_No_Peak.png")
    pyplot.savefig(rf"./230427_Pipeline_test_Part1/{identifier}_No_Peak.svg", format="svg")
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    return


def plot_density(*, ax, x, samples, percentiles=(5, 95), percentile_kwargs=None, **kwargs):
    """
    Method to plot the original data points alongside the posterior predictive plot (percentiles marked with a black, dashed line).
    Serves as a more accurate comparison between data and model than comparing data and posterior distribution.
    """
    assert samples.ndim == 2

    # Step-function mode draws horizontal density bands inbetween the x coordinates
    step_mode = samples.shape[1] == x.shape[0] - 1
    fill_kwargs = {}
    if step_mode:
        samples = np.hstack([samples, samples[:, -1][:, None]])
        fill_kwargs["step"] = "post"

    # Plot the density band
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=x,
        samples=samples,
        fill_alpha=1,
        plot_samples=False,
        palette=pyplot.cm.Blues,
        fill_kwargs=fill_kwargs,
        **kwargs,
    )

    # Add percentiles for orientation
    pkwargs = dict(
        linestyle="--",
        color="black",
    )
    pkwargs.update(percentile_kwargs or {})
    for p in percentiles:
        values = np.percentile(samples, p, axis=0)
        if step_mode:
            ax.stairs(values[:-1], x, baseline=None, **pkwargs)
        else:
            ax.plot(x, values, **pkwargs)
        pass

    return


def plot_posterior_predictive(identifier, time_np, intensity_np, idata):
    """
    Save plot of posterior_predictive with 95 % HDI and original data points.

    Parameters
    ----------
    identifier
        unique identifier of this particular signal

    time_np
        numpy array with the time values of the relevant timeframe

    intensity_np
        numpy array with the intensity values of the relevant timeframe

    idata
        infernce data object
    """
    fig, ax = pyplot.subplots()
    # plot the posterior predictive
    plot_density(
        ax=ax,
        x=time_np,
        samples=idata.posterior_predictive.L.stack(sample=("chain", "draw")).T.values,
        percentiles=(2.5, 97.5),
    )
    # plot the raw data points
    ax.scatter(time_np, intensity_np, marker="x", color="black", label="data")
    ax.set_xlabel("time / min", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=11.5, fontweight="bold")
    pyplot.legend()
    pyplot.tight_layout()
    # TODO: fix paths
    pyplot.savefig(rf"./230427_Pipeline_test_Part1/{identifier}_predictive_posterior.png")
    pyplot.savefig(
        rf"./230427_Pipeline_test_Part1/{identifier}_predictive_posterior.svg", format="svg"
    )
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    return


def plot_posterior(identifier, time_np, intensity_np, idata):
    """
    Save plot of posterior, estimated baseline and original data points.

    Parameters
    ----------
    identifier
        unique identifier of this particular signal

    time_np
        numpy array with the time values of the relevant timeframe

    intensity_np
        numpy array with the intensity values of the relevant timeframe

    idata
        infernce data object
    """
    fig, ax = pyplot.subplots()
    # plot the posterior
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=time_np,
        samples=idata.posterior.y.stack(sample=("chain", "draw")).T.values,
    )
    # plot the raw data points
    ax.scatter(time_np, intensity_np, marker="x", color="black", label="data")
    # plot the baseline
    x = np.array(ax.get_xlim())
    y = (
        az.summary(idata).loc["baseline_intercept", "mean"]
        + az.summary(idata).loc["baseline_slope", "mean"] * x
    )
    pyplot.plot(x, y)
    pyplot.legend()
    ax.set_xlabel("time / min", fontsize=12, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=12, fontweight="bold")
    pyplot.xticks(size=11.5)
    pyplot.yticks(size=11.5)
    pyplot.tight_layout()
    pyplot.savefig(rf"./230427_Pipeline_test_Part1/{identifier}_posterior.png")
    pyplot.savefig(rf"./230427_Pipeline_test_Part1/{identifier}_posterior.svg", format="svg")
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    return
