import numpy as np
import pandas
import pathlib
import pytest
from pathlib import Path

from peak_performance import pipeline as pl


def test_user_input_class():
    path = Path("./example")
    raw_data_files = ["A1t1R1Part2_1_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(r"example\A1t1R1Part2_1_110_109.9_110.1.npy")
    acquisition = "A1t1R1"
    experiment = 4
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    # test instantiation of the UserInput class
    ui = pl.UserInput(path, raw_data_files, data_file_format, double_peak, retention_time_estimate, peak_width_estimate, pre_filtering, minimum_sn, timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end)
    assert ui.timeseries == timeseries
    # test some of the error handling of the parameter setter of the UserInput class
    with pytest.raises(pl.InputError):
        ui = pl.UserInput(path, raw_data_files, data_file_format, double_peak, retention_time_estimate, peak_width_estimate, pre_filtering, minimum_sn, timeseries, 5, experiment, precursor_mz, product_mz_start, product_mz_end)
    with pytest.raises(pl.InputError):
        ui = pl.UserInput(path, raw_data_files, data_file_format, double_peak, retention_time_estimate, peak_width_estimate, pre_filtering, minimum_sn, timeseries, acquisition, experiment, "mz", product_mz_start, product_mz_end)
    pass


def test_detect_raw_data():
    path = Path("./example/")
    data_format = ".npy"
    files = pl.detect_raw_data(path, data_format)
    assert files == ['A1t1R1Part2_1_110_109.9_110.1.npy', 'A1t1R1Part2_2_111_109.9_110.1.npy', 'A1t1R1Part2_3_111_110.9_111.1.npy', 'A1t1R1Part2_4_112_110.9_111.1.npy', 'A1t1R1Part2_5_112_111.9_112.1.npy']
    pass


def test_parse_data():
    path = Path("./example/")
    data_format = ".npy"
    filename = "A1t1R1Part2_1_110_109.9_110.1.npy"
    timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end = pl.parse_data(path, filename, data_format)
    assert isinstance(timeseries[0], np.ndarray)
    assert isinstance(timeseries[1], np.ndarray)
    assert acquisition == "A1t1R1Part2"
    assert experiment == 1
    assert precursor_mz == 110
    assert product_mz_start == 109.9 
    assert product_mz_end == 110.1
    pass


def test_initiate():
    path = Path("./example/")
    run_dir = "test"
    df_summary, path = pl.initiate(path, run_dir=run_dir)
    assert df_summary == pandas.DataFrame(
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
    assert path == Path("./example/test")
    assert path.exists()
    path.rmdir()
    pass


def test_prefiltering():
    pass


def test_postfiltering():
    pass


def test_report_add_nan_to_summary():
    # create df_summary
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
    # create instance of the UserInput class
    path = Path("./example")
    raw_data_files = ["A1t1R1Part2_1_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(r"example\A1t1R1Part2_1_110_109.9_110.1.npy")
    acquisition = "A1t1R1"
    experiment = 4
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    ui = pl.UserInput(path, raw_data_files, data_file_format, double_peak, retention_time_estimate, peak_width_estimate, pre_filtering, minimum_sn, timeseries, acquisition, experiment, precursor_mz, product_mz_start, product_mz_end)
    filename = "A1t1R1Part2_1_110_109.9_110.1.npy"
    df_summary = pl.report_add_nan_to_summary(filename, ui, df_summary)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat', 'acquisition', 'experiment', 'precursor_mz', 'product_mz_start', 'product_mz_end', 'double_peak']
    assert list(df_summary.loc[:, "mean"]) == len(df_summary.index) * [[np.nan]]
    assert list(df_summary.loc[:, "acquisition"]) == ['A1t1R1', 'A1t1R1', 'A1t1R1', 'A1t1R1', 'A1t1R1', 'A1t1R1', 'A1t1R1', 'A1t1R1']
    assert list(df_summary.loc[:, "experiment"]) == len(df_summary.index) * [4]
    assert list(df_summary.loc[:, "precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert list(df_summary.loc[:, "double_peak"]) == len(df_summary.index) * [False]
    pass


def test_report_add_data_to_summary():
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

    pass


def test_pipeline_loop():
    # make sure the correctly named files are there; test the report sheets for similarity
    pass
