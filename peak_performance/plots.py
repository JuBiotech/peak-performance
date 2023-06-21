import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as st
from matplotlib import pyplot


def plot_raw_data(identifier, ui):
    """
    Plot just the raw data in case no peak was found.

    Parameters
    ----------
    identifier
        unique identifier of this particular signal (e.g. filename).
    ui
        Instance of the UserInput class.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    # plot the data to be able to check if peak detection was correct or not
    fig, ax = pyplot.subplots()
    ax.scatter(time, intensity, marker="x", color="black", label="data")
    pyplot.legend()
    ax.set_xlabel("time / min", fontsize=12, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=12, fontweight="bold")
    pyplot.xticks(size=11.5)
    pyplot.yticks(size=11.5)
    pyplot.tight_layout()
    pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_No_Peak.png")
    pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_No_Peak.svg", format="svg")
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


def plot_posterior_predictive(identifier, ui, idata, discarded):
    """
    Save plot of posterior_predictive with 95 % HDI and original data points.

    Parameters
    ----------
    ----------
    identifier
        unique identifier of this particular signal
    ui
        Instance of the UserInput class.
    idata
        infernce data object
    discarded
        Alters the name of the saved plot. If True, a "_NoPeak" is added to the name.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    fig, ax = pyplot.subplots()
    # plot the posterior predictive
    plot_density(
        ax=ax,
        x=time,
        samples=idata.posterior_predictive.L.stack(sample=("chain", "draw")).T.values,
        percentiles=(2.5, 97.5),
    )
    # plot the raw data points
    ax.scatter(time, intensity, marker="x", color="black", label="data")
    ax.set_xlabel("time / min", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("intensity / a.u.", fontsize=11.5, fontweight="bold")
    pyplot.legend()
    pyplot.tight_layout()
    # if signal was discarded, add a "_NoPeak" to the file name
    if discarded:
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_predictive_posterior_NoPeak.png")
        pyplot.savefig(
            rf"{ui.path}/{identifier[:-4]}_predictive_posterior_NoPeak.svg", format="svg"
        )
    else:
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_predictive_posterior.png")
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_predictive_posterior.svg", format="svg")
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    return


def plot_posterior(identifier, ui, idata, discarded):
    """
    Save plot of posterior, estimated baseline and original data points.

    Parameters
    ----------
    identifier
        unique identifier of this particular signal
    ui
        Instance of the UserInput class.
    idata
        infernce data object
    discarded
        Alters the name of the saved plot. If True, a "_NoPeak" is added to the name.
    """
    time = ui.timeseries[0]
    intensity = ui.timeseries[1]
    fig, ax = pyplot.subplots()
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
    # if signal was discarded, add a "_NoPeak" to the file name
    if discarded:
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_posterior_NoPeak.png")
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_posterior_NoPeak.svg", format="svg")
    else:
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_posterior.png")
        pyplot.savefig(rf"{ui.path}/{identifier[:-4]}_posterior.svg", format="svg")
    pyplot.cla()
    pyplot.clf()
    pyplot.close()

    return
