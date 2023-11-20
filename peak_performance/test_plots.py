import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas

from peak_performance import plots


def test_plot_raw_data():
    """
    Tests the plot_raw_data() function from the plots module.
    """
    identifier = "test_plot_raw_data"
    path = Path(__file__).absolute().parent.parent / "test_data"
    plots.plot_raw_data(
        identifier=identifier,
        time=np.array([1, 2, 3, 4, 5]),
        intensity=np.array([1, 10, 25, 10, 1]),
        path=path,
    )
    assert Path(path / f"{identifier}_NoPeak.png").exists()
    assert Path(path / f"{identifier}_NoPeak.svg").exists()
    os.remove(path / f"{identifier}_NoPeak.png")
    os.remove(path / f"{identifier}_NoPeak.svg")
    pass


def test_plot_posterior_predictive():
    """
    Tests the plot_posterior_predictive() function from the plots module.
    """
    path = Path(__file__).absolute().parent.parent / "test_data/test_plot_posterior_predictive"
    idata = az.from_netcdf(path / "idata.nc")
    time = idata.constant_data.time.values
    intensity = idata.constant_data.intensity.values
    identifier = "identifier"
    # test plots of discarded signals
    plots.plot_posterior_predictive(
        identifier=identifier,
        time=time,
        intensity=intensity,
        path=path,
        idata=idata,
        discarded=True,
    )
    assert (path / f"{identifier}_predictive_posterior_NoPeak.png").exists()
    assert (path / f"{identifier}_predictive_posterior_NoPeak.svg").exists()
    os.remove(path / f"{identifier}_predictive_posterior_NoPeak.png")
    os.remove(path / f"{identifier}_predictive_posterior_NoPeak.svg")
    # test plots of accepted signals
    plots.plot_posterior_predictive(
        identifier=identifier,
        time=time,
        intensity=intensity,
        path=path,
        idata=idata,
        discarded=False,
    )
    assert Path(path / f"{identifier}_predictive_posterior.png").exists()
    assert Path(path / f"{identifier}_predictive_posterior.svg").exists()
    os.remove(path / f"{identifier}_predictive_posterior.png")
    os.remove(path / f"{identifier}_predictive_posterior.svg")
    pass


def test_plot_posterior():
    """
    Tests the plot_posterior() function from the plots module.
    """
    path = Path(__file__).absolute().parent.parent / "test_data/test_plot_posterior"
    idata = az.from_netcdf(path / "idata.nc")
    time = idata.constant_data.time.values
    intensity = idata.constant_data.intensity.values
    identifier = "identifier"
    # test plots of discarded signals
    plots.plot_posterior(
        identifier=identifier,
        time=time,
        intensity=intensity,
        path=path,
        idata=idata,
        discarded=True,
    )
    assert (path / f"{identifier}_posterior_NoPeak.png").exists()
    assert (path / f"{identifier}_posterior_NoPeak.svg").exists()
    os.remove(path / f"{identifier}_posterior_NoPeak.png")
    os.remove(path / f"{identifier}_posterior_NoPeak.svg")
    # test plots of accepted signals
    plots.plot_posterior(
        identifier=identifier,
        time=time,
        intensity=intensity,
        path=path,
        idata=idata,
        discarded=False,
    )
    assert (path / f"{identifier}_posterior.png").exists()
    assert (path / f"{identifier}_posterior.svg").exists()
    os.remove(path / f"{identifier}_posterior.png")
    os.remove(path / f"{identifier}_posterior.svg")
    pass


def test_plot_model_comparison():
    """
    Tests the plot_model_comparison() function from the plots module.
    """
    path = Path(__file__).absolute().parent.parent / "test_data/test_plot_model_comparison"
    ranking = pandas.read_excel(path / "ranking.xlsx")
    identifier = "ranking"
    plots.plot_model_comparison(
        df_comp=ranking,
        identifier="ranking",
        path=path,
    )
    assert Path(path / f"model_comparison_{identifier}.png").exists()
    assert Path(path / f"model_comparison_{identifier}.svg").exists()
    os.remove(path / f"model_comparison_{identifier}.png")
    os.remove(path / f"model_comparison_{identifier}.svg")
    pass
