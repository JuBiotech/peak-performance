import shutil
from pathlib import Path

import arviz as az
import numpy as np
import pandas
import pytest

from peak_performance import pipeline as pl

# define columns for empty summary DataFrame for results
COLUMNS = [
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
    "experiment_or_precursor_mz",
    "product_mz_start",
    "product_mz_end",
    "is_peak",
    "cause_for_rejection",
    "double_peak",     
]


def test_user_input_class():
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A1t1R1Part2_110_109.9_110.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 118
    product_mz_start = "71.9"
    product_mz_end = 72.1
    # test instantiation of the UserInput class
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    assert ui.timeseries.all() == timeseries.all()
    assert ui.precursor == 118
    assert ui.product_mz_start == 71.9
    assert ui.product_mz_end == 72.1
    # test some of the error handling of the parameter setter of the UserInput class
    with pytest.raises(pl.InputError):
        ui = pl.UserInput(
            path,
            raw_data_files,
            data_file_format,
            double_peak,
            retention_time_estimate,
            peak_width_estimate,
            pre_filtering,
            minimum_sn,
            timeseries,
            5,
            precursor_mz,
            product_mz_start,
            product_mz_end,
        )
    with pytest.raises(pl.InputError):
        ui = pl.UserInput(
            path,
            raw_data_files,
            data_file_format,
            double_peak,
            retention_time_estimate,
            peak_width_estimate,
            pre_filtering,
            minimum_sn,
            timeseries,
            acquisition,
            "mz",
            product_mz_start,
            product_mz_end,
        )
    pass


def test_detect_raw_data():
    path = Path(__file__).absolute().parent.parent / "example"
    data_format = ".npy"
    files = pl.detect_raw_data(path, data_type=data_format)
    files = sorted(files)
    expected_files = sorted(
        [
            "A1t1R1Part2_110_109.9_110.1.npy",
            "A1t1R1Part2_111_109.9_110.1.npy",
            "A1t1R1Part2_111_110.9_111.1.npy",
            "A1t1R1Part2_112_110.9_111.1.npy",
            "A1t1R1Part2_112_111.9_112.1.npy",
            "A2t2R1Part1_132_85.9_86.1.npy",
            "A4t4R1Part2_137_72.9_73.1.npy",
        ]
    )
    assert files == expected_files
    pass


def test_parse_data():
    path = Path(__file__).absolute().parent.parent / "example"
    data_format = ".npy"
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    (
        timeseries,
        acquisition,
        precursor,
        product_mz_start,
        product_mz_end,
    ) = pl.parse_data(path, filename, data_format)
    assert isinstance(timeseries[0], np.ndarray)
    assert isinstance(timeseries[1], np.ndarray)
    assert acquisition == "A1t1R1Part2"
    assert precursor == 110.0
    assert product_mz_start == 109.9
    assert product_mz_end == 110.1
    pass


def test_initiate():
    path = Path(__file__).absolute().parent.parent / "example"
    run_dir = "test"
    df_summary, path = pl.initiate(path, run_dir=run_dir)
    df_summary2 = pandas.DataFrame(columns=COLUMNS)
    assert df_summary2.values.all() == df_summary.values.all()
    assert df_summary2.columns.all() == df_summary.columns.all()
    assert path == Path(__file__).absolute().parent.parent / "example" / "test"
    assert path.exists()
    shutil.rmtree(path)
    pass


def test_prefiltering():
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [26.3]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A1t1R1Part2_110_109.9_110.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    # positive test
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    found_peak, df_summary_1 = pl.prefiltering(filename, ui, 108, df_summary)
    assert found_peak
    assert df_summary_1.values.all() == df_summary.values.all()
    assert df_summary_1.columns.all() == df_summary.columns.all()
    # negative test due to retention time
    retention_time_estimate = [22.3]
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    found_peak, df_summary_1 = pl.prefiltering(filename, ui, 108, df_summary)
    assert not found_peak
    assert len(df_summary_1.loc[:, "mean"].values) == 8
    assert list(df_summary_1.columns) == COLUMNS
    assert list(df_summary_1.loc[:, "mean"]) == len(df_summary_1.index) * [[np.nan]]
    # negative test due to signal-to-noise ratio
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A4t4R1Part2_137_72.9_73.1.npy"
    )
    raw_data_files = ["A4t4R1Part2_137_72.9_73.1.npy"]
    retention_time_estimate = [26.3]
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A4t4R1Part2_137_72.9_73.1.npy"
    found_peak, df_summary_2 = pl.prefiltering(filename, ui, 108, df_summary)
    assert not found_peak
    assert len(df_summary_2.loc[:, "mean"].values) == 8
    assert list(df_summary_2.columns) == COLUMNS
    assert list(df_summary_2.loc[:, "mean"]) == len(df_summary_2.index) * [[np.nan]]
    pass


def test_postfiltering():
    # load exemplary inference data object
    idata = az.from_netcdf(Path(__file__).absolute().parent.parent / "example" / "idata_double")
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A2t2R1Part1_132_85.9_86.1.npy"]
    data_file_format = ".npy"
    double_peak = [True]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A2t2R1Part1_132_85.9_86.1.npy"
    )
    acquisition = "A2t2R1Part1"
    precursor_mz = 132
    product_mz_start = 85.9
    product_mz_end = 86.1
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A2t2R1Part1_132_85.9_86.1.npy"
    resample, discard, df_summary = pl.postfiltering(filename, idata, ui, df_summary)
    # tests
    assert not resample
    assert discard
    assert list(df_summary.loc[:, "mean"]) == len(df_summary.index) * [[np.nan]]
    pass


def test_single_peak_report_add_nan_to_summary():
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A1t1R1Part2_110_109.9_110.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    df_summary = pl.report_add_nan_to_summary(filename, ui, df_summary)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == COLUMNS
    assert list(df_summary.loc[:, "mean"]) == len(df_summary.index) * [[np.nan]]
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert list(df_summary.loc[:, "is_peak"]) == len(df_summary.index) * [True]
    assert list(df_summary.loc[:, "cause_for_rejection"]) == len(df_summary.index) * [""]
    assert list(df_summary.loc[:, "double_peak"]) == len(df_summary.index) * [False]
    pass


def test_double_peak_report_add_nan_to_summary():
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [True]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A1t1R1Part2_110_109.9_110.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    rejection_msg = "because I said so"
    df_summary = pl.report_add_nan_to_summary(filename, ui, df_summary, False, rejection_msg)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == COLUMNS
    assert list(df_summary.loc[:, "mean"]) == len(df_summary.index) * [[np.nan]]
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert list(df_summary.loc[:, "is_peak"]) == len(df_summary.index) * [False]
    assert list(df_summary.loc[:, "cause_for_rejection"]) == len(df_summary.index) * ["because I said so"]
    assert list(df_summary.loc[:, "double_peak"]) == len(df_summary.index) * [True]
    pass


def test_single_peak_report_add_data_to_summary():
    # load exemplary inference data object
    idata = az.from_netcdf(Path(__file__).absolute().parent.parent / "example" / "idata")
    # create empty DataFrame
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    double_peak = [False]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A1t1R1Part2_110_109.9_110.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 118
    product_mz_start = 71.9
    product_mz_end = 72.1
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A1t1R1Part2_110_109.9_110.1.npy"
    # add data to df_summary
    df_summary = pl.report_add_data_to_summary(filename, idata, df_summary, ui, True)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == COLUMNS
    assert list(df_summary.loc[:, "mean"]) == [
        5.565,
        8.446,
        25.989,
        132.743,
        0.516,
        2180.529,
        2762.695,
        20.924,
    ]
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert list(df_summary.loc[:, "is_peak"]) == len(df_summary.index) * [True]
    assert list(df_summary.loc[:, "cause_for_rejection"]) == len(df_summary.index) * [""]
    assert list(df_summary.loc[:, "double_peak"]) == len(df_summary.index) * [False]
    pass


def test_double_peak_report_add_data_to_summary():
    # load exemplary inference data object
    idata = az.from_netcdf(Path(__file__).absolute().parent.parent / "example" / "idata_double")
    # create empty DataFrame
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A2t2R1Part1_132_85.9_86.1.npy"]
    data_file_format = ".npy"
    double_peak = [True]
    retention_time_estimate = [22.5]
    peak_width_estimate = 1.5
    pre_filtering = True
    minimum_sn = 5
    timeseries = np.load(
        Path(__file__).absolute().parent.parent / "example" / "A2t2R1Part1_132_85.9_86.1.npy"
    )
    acquisition = "A1t1R1"
    precursor_mz = 132
    product_mz_start = 85.9
    product_mz_end = 86.1
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        double_peak,
        retention_time_estimate,
        peak_width_estimate,
        pre_filtering,
        minimum_sn,
        timeseries,
        acquisition,
        precursor_mz,
        product_mz_start,
        product_mz_end,
    )
    filename = "A2t2R1Part1_132_85.9_86.1.npy"
    # add data to df_summary
    df_summary = pl.report_add_data_to_summary(filename, idata, df_summary, ui, True)
    # tests
    assert list(df_summary.columns) == COLUMNS
    assert list(df_summary.loc[:, "mean"]) == [
        -17.786,
        -8.814,
        11.357,
        180.677,
        1.967,
        3828.652,
        954.279,
        5.288,
        -17.786,
        -8.814,
        12.659,
        180.677,
        1.563,
        10377.713,
        1896.595,
        10.52,
    ]
    assert len(df_summary.index) == 16
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [132]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [85.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [86.1]
    assert list(df_summary.loc[:, "is_peak"]) == len(df_summary.index) * [True]
    assert list(df_summary.loc[:, "cause_for_rejection"]) == len(df_summary.index) * [""]
    assert list(df_summary.loc[:, "double_peak"]) == 8 * ["1st"] + 8 * ["2nd"]
    pass


def test_pipeline_loop():
    # make sure the correctly named files are there; test the report sheets for similarity
    pass
