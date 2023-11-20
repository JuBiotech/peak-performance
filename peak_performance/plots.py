"""
PeakPerformance
Copyright (C) 2023 Forschungszentrum Jülich GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from pathlib import Path
from typing import Sequence, Union

import arviz as az
import numpy as np
import pandas
import pymc as pm
from matplotlib import pyplot as plt


def plot_raw_data(
    identifier: str,
    time: np.ndarray,
    intensity: np.ndarray,
    path: Union[str, os.PathLike],
    save_formats: Sequence[str] = ("png", "svg"),
):
    """
    Plot just the raw data in case no peak was found.

    Parameters
    ----------
    identifier
        Unique identifier of this particular signal (e.g. filename).
    time
        NumPy array with the time values of the relevant timeframe.
    intensity
        NumPy array with the intensity values of the relevant timeframe.
    path
        Path to the folder containing the results of the current run.
    save_formats
        Which file formats to save as.
        Must be supported by `plt.savefig()`, e.g. ``("png", "svg", "pdf")``.
    """
    time = np.array(time)
    intensity = np.array(intensity)
    # plot the data to be able to check if peak detection was correct or not
    fig, ax = plt.subplots()
    ax.scatter(time, intensity, marker="x", color="black", label="data")
    plt.legend()
    ax.set_xlabel("time / min", fontsize=12, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=12, fontweight="bold")
    plt.xticks(size=11.5)
    plt.yticks(size=11.5)
    fig.tight_layout()
    for format in save_formats:
        fig.savefig(Path(path) / f"{identifier}_NoPeak.{format}", format=format)
    plt.close(fig)

    return


def plot_density(
    *, ax, x: np.ndarray, samples, percentiles=(5, 95), percentile_kwargs=None, **kwargs
):
    """
    Method to plot the original data points alongside the posterior predictive plot (percentiles marked with a black, dashed line).
    Serves as a more accurate comparison between data and model than comparing data and posterior distribution.

    Parameters
    ----------
    ax
        Axes of a matplotlib figure.
    x
        Values of the x dimension of the plot (here: time).
    samples
        Posterior predictive samples taken from an inference data obejct.
    percentiles
        Lower and upper percentiles to be plotted.
    **kwargs
        The keyword arguments are used for plotting with ax.plot() and ax.stairs(), e.g. the following:
    linestyle
        Style of the line marking the border of the chosen percentiles (default = "--", i.e. a dashed line).
    color
        Color of the line marking the border of the chosen percentiles (default = "black").
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
        palette=plt.cm.Blues,
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


def plot_posterior_predictive(
    identifier: str,
    time: np.ndarray,
    intensity: np.ndarray,
    path: Union[str, os.PathLike],
    idata: az.InferenceData,
    discarded: bool,
    save_formats: Sequence[str] = ("png", "svg"),
):
    """
    Save plot of posterior_predictive with 95 % HDI and original data points.

    Parameters
    ----------
    identifier
        Unique identifier of this particular signal (e.g. filename).
    time
        NumPy array with the time values of the relevant timeframe.
    intensity
        NumPy array with the intensity values of the relevant timeframe.
    path
        Path to the folder containing the results of the current run.
    idata
        Infernce data object.
    discarded
        Alters the name of the saved plot. If True, a "_NoPeak" is added to the name.
    save_formats
        Which file formats to save as.
        Must be supported by `plt.savefig()`, e.g. ``("png", "svg", "pdf")``.
    """
    time = np.array(time)
    intensity = np.array(intensity)
    fig, ax = plt.subplots()
    # plot the posterior predictive
    plot_density(
        ax=ax,
        x=time,
        samples=idata.posterior_predictive.y.stack(sample=("chain", "draw")).T.values,
        percentiles=(2.5, 97.5),
    )
    # plot the raw data points
    ax.scatter(time, intensity, marker="x", color="black", label="data")
    ax.set_xlabel("time / min", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=11.5, fontweight="bold")
    plt.xticks(size=11.5)
    plt.yticks(size=11.5)
    plt.legend()
    fig.tight_layout()
    # if signal was discarded, add a "_NoPeak" to the file name
    if discarded:
        for format in save_formats:
            fig.savefig(
                Path(path) / f"{identifier}_predictive_posterior_NoPeak.{format}", format=format
            )
    else:
        for format in save_formats:
            fig.savefig(Path(path) / f"{identifier}_predictive_posterior.{format}", format=format)
    plt.close(fig)

    return


def plot_posterior(
    identifier: str,
    time: np.ndarray,
    intensity: np.ndarray,
    path: Union[str, os.PathLike],
    idata: az.InferenceData,
    discarded: bool,
    save_formats: Sequence[str] = ("png", "svg"),
):
    """
    Saves plot of posterior, estimated baseline, and original data points.

    Parameters
    ----------
    identifier
        Unique identifier of this particular signal (e.g. filename).
    time
        NumPy array with the time values of the relevant timeframe.
    intensity
        NumPy array with the intensity values of the relevant timeframe.
    path
        Path to the folder containing the results of the current run.
    idata
        Infernce data object.
    discarded
        Alters the name of the saved plot. If True, a "_NoPeak" is added to the name.
    save_formats
        Which file formats to save as.
        Must be supported by `plt.savefig()`, e.g. ``("png", "svg", "pdf")``.
    """
    time = np.array(time)
    intensity = np.array(intensity)
    az_summary: pandas.DataFrame = az.summary(idata)

    fig, ax = plt.subplots()
    # plot the posterior
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=idata.posterior.y.stack(sample=("chain", "draw")).T.values,
    )
    # plot the raw data points
    ax.scatter(time, intensity, marker="x", color="black", label="data")
    # plot the baseline
    x = np.array(ax.get_xlim())
    y = az_summary.loc["baseline_intercept", "mean"] + az_summary.loc["baseline_slope", "mean"] * x
    plt.plot(x, y)
    plt.legend()
    ax.set_xlabel("time / min", fontsize=12, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=12, fontweight="bold")
    plt.xticks(size=11.5)
    plt.yticks(size=11.5)
    fig.tight_layout()
    # if signal was discarded, add a "_NoPeak" to the file name
    if discarded:
        for format in save_formats:
            fig.savefig(Path(path) / f"{identifier}_posterior_NoPeak.{format}", format=format)
    else:
        for format in save_formats:
            fig.savefig(Path(path) / f"{identifier}_posterior.{format}", format=format)
    plt.close(fig)

    return


def plot_model_comparison(
    df_comp: pandas.DataFrame,
    identifier: str,
    path: Union[str, os.PathLike],
    save_formats: Sequence[str] = ("png", "svg"),
):
    """
    Function to plot the results of a model comparison.

    Parameters
    ----------
    df_comp
        DataFrame containing the ranking of the given models.
    identifier
        Unique identifier of this particular signal (e.g. filename).
    path
        Path to the folder containing the results of the current run.
    save_formats
        Which file formats to save as.
        Must be supported by `plt.savefig()`, e.g. ``("png", "svg", "pdf")``.
    """
    axes = az.plot_compare(df_comp, insample_dev=False)
    fig = axes.figure
    plt.tight_layout()
    for format in save_formats:
        fig.savefig(Path(path) / f"model_comparison_{identifier}.{format}", format=format)
    plt.close(fig)

    return
