import os
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
    "model_type",
    "subpeak",
]


def test_user_input_class():
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    model_type = ["normal"]
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
        model_type,
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
            model_type,
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
            model_type,
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
    model_type = ["normal"]
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
        model_type,
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
    retention_time_estimate = [0]
    ui = pl.UserInput(
        path,
        raw_data_files,
        data_file_format,
        model_type,
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
    assert all(pandas.isna(df_summary_1["mean"]))
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
        model_type,
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
    assert all(pandas.isna(df_summary_2["mean"]))
    pass


def test_postfiltering_success():
    # load exemplary inference data object
    idata = az.from_netcdf(
        Path(__file__).absolute().parent.parent
        / "test_data/test_postfiltering_success/idata_double_normal.nc"
    )
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A2t2R1Part1_132_85.9_86.1.npy"]
    data_file_format = ".npy"
    model_type = ["double_normal"]
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
        model_type,
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
    assert not discard
    pass


def test_postfiltering_resample():
    # load exemplary inference data object
    idata = az.from_netcdf(
        Path(__file__).absolute().parent.parent
        / "test_data/test_postfiltering_resample/idata_double_skew_rhat_too_high.nc"
    )
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A2t2R1Part1_132_85.9_86.1.npy"]
    data_file_format = ".npy"
    model_type = ["double_normal"]
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
        model_type,
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
    assert resample
    assert not discard
    pass


def test_single_peak_report_add_nan_to_summary():
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    model_type = ["skew_normal"]
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
        model_type,
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
    df_summary = pl.report_add_nan_to_summary(filename, ui, df_summary, rejection_msg)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == COLUMNS
    assert all(pandas.isna(df_summary["mean"]))
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert not any(df_summary["is_peak"])
    assert all(df_summary["cause_for_rejection"] == rejection_msg)
    assert list(df_summary.loc[:, "model_type"]) == len(df_summary.index) * ["skew_normal"]
    pass


def test_double_peak_report_add_nan_to_summary():
    # create df_summary
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    model_type = ["double_normal"]
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
        model_type,
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
    df_summary = pl.report_add_nan_to_summary(filename, ui, df_summary, rejection_msg)
    # tests
    assert len(df_summary.loc[:, "mean"].values) == 8
    assert list(df_summary.columns) == COLUMNS
    assert all(pandas.isna(df_summary["mean"]))
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [118]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [71.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [72.1]
    assert not any(df_summary["is_peak"])
    assert all(df_summary["cause_for_rejection"] == rejection_msg)
    assert list(df_summary.loc[:, "model_type"]) == len(df_summary.index) * ["double_normal"]
    pass


def test_single_peak_report():
    # load exemplary inference data object
    idata = az.from_netcdf(
        Path(__file__).absolute().parent.parent / "test_data/test_single_peak_report/idata.nc"
    )
    # create empty DataFrame
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A1t1R1Part2_110_109.9_110.1.npy"]
    data_file_format = ".npy"
    model_type = ["skew_normal"]
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
        model_type,
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
    assert all(df_summary["is_peak"])
    assert all(df_summary["cause_for_rejection"] == "")
    assert list(df_summary.loc[:, "model_type"]) == len(df_summary.index) * ["skew_normal"]
    pass


@pytest.mark.parametrize("idata", ["idata_double_normal.nc", "idata_double_skew_normal.nc"])
def test_double_peak_report(idata):
    # load exemplary inference data object
    idata = az.from_netcdf(
        Path(__file__).absolute().parent.parent / "test_data/test_double_peak_report" / idata
    )
    # create empty DataFrame
    df_summary = pandas.DataFrame(columns=COLUMNS)
    # create instance of the UserInput class
    path = Path(__file__).absolute().parent.parent / "example"
    raw_data_files = ["A2t2R1Part1_132_85.9_86.1.npy"]
    data_file_format = ".npy"
    model_type = ["double_skew_normal"]
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
        model_type,
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
    assert len(df_summary.index) == 16
    assert list(df_summary.loc[:, "acquisition"]) == len(df_summary.index) * ["A1t1R1"]
    assert list(df_summary.loc[:, "experiment_or_precursor_mz"]) == len(df_summary.index) * [132]
    assert list(df_summary.loc[:, "product_mz_start"]) == len(df_summary.index) * [85.9]
    assert list(df_summary.loc[:, "product_mz_end"]) == len(df_summary.index) * [86.1]
    assert all(df_summary["is_peak"])
    assert all(df_summary["cause_for_rejection"] == "")
    assert list(df_summary.loc[:, "model_type"]) == 16 * ["double_skew_normal"]
    assert list(df_summary.loc[:, "subpeak"]) == 8 * ["1st"] + 8 * ["2nd"]
    pass


def test_parse_unique_identifiers():
    files = [
        "A1t1R1Part2_110_109.9_110.1.npy",
        "A1t1R1Part2_111_109.9_110.1.npy",
        "A1t1R1Part2_111_110.9_111.1.npy",
        "A1t1R1Part2_112_110.9_111.1.npy",
    ]
    unique_identifiers = pl.parse_unique_identifiers(files)
    assert sorted(unique_identifiers) == sorted(
        [
            "110_109.9_110.1",
            "111_109.9_110.1",
            "111_110.9_111.1",
            "112_110.9_111.1",
        ]
    )
    pass


def test_excel_template_prepare():
    path_raw_data = Path(__file__).absolute().parent.parent / "example"
    path_peak_performance = Path(__file__).absolute().parent.parent
    files = ["mp3", "flac", "wav", "m4a"]
    identifiers = ["1", "2", "3"]
    pl.excel_template_prepare(path_raw_data, path_peak_performance, files, identifiers)
    # test whether Template.xlsx was copied from peak-performance to example
    assert Path(path_raw_data / "Template.xlsx").exists()
    # remove Template.xlsx from example
    os.remove(Path(path_raw_data / "Template.xlsx"))
    pass


def test_parse_files_for_model_selection():
    path_peak_performance = Path(__file__).absolute().parent.parent
    # load empty signals sheet from Template.xlsx
    signals = pandas.read_excel(Path(path_peak_performance) / "Template.xlsx", sheet_name="signals")
    signals["unique_identifier"] = ["1", "2", "3", "4", "5", "6", "7"]
    with pytest.raises(pl.InputError):
        files = pl.parse_files_for_model_selection(signals)
    # have one unique_identifier where neither model nor acquisition were given
    # (and multiple different acquisitions were defined for other unique identifiers)
    signals["acquisition_for_choosing_model_type"] = ["A1", "B1", "C1", "D1", "E1", "F1", np.nan]
    signals["model_type"] = 7 * [np.nan]
    with pytest.raises(pl.InputError):
        files = pl.parse_files_for_model_selection(signals)
    # if models for every unique identifier were supplied, the result should be empty
    signals["model_type"] = 7 * ["normal"]
    with pytest.raises(pl.InputError):
        files = pl.parse_files_for_model_selection(signals)
    # mixture of supplied model and supplying different acquisitions for model selection
    signals["acquisition_for_choosing_model_type"] = [np.nan, "B1", "C1", "D1", "E1", "F1", "G1"]
    signals["model_type"] = ["normal"] + 6 * [np.nan]
    files = pl.parse_files_for_model_selection(signals)
    assert files == {"B1_2": "2", "C1_3": "3", "D1_4": "4", "E1_5": "5", "F1_6": "6", "G1_7": "7"}
    # mixture of supplied model and supplying one acquisition for model selection
    signals["acquisition_for_choosing_model_type"] = [np.nan, "B1"] + 5 * [np.nan]
    files = pl.parse_files_for_model_selection(signals)
    assert files == {"B1_2": "2", "B1_3": "3", "B1_4": "4", "B1_5": "5", "B1_6": "6", "B1_7": "7"}
    pass


def test_model_selection_check():
    # case 1: double peak is too close to single peak in elpd score
    result_df = pandas.DataFrame(
        {"elpd_loo": [50, 30, 29, -5], "ic": ["loo", "loo", "loo", "loo"]},
        index=["double_normal", "double_skew_normal", "normal", "skew_normal"],
    )
    selected_model = pl.model_selection_check(result_df, "loo", 25)
    assert selected_model == "normal"
    # case 2: double peak exceeds elpd score difference threshold and is thusly accepted
    result_df = pandas.DataFrame(
        {"elpd_loo": [50, 30, 10, -5], "ic": ["loo", "loo", "loo", "loo"]},
        index=["double_normal", "double_skew_normal", "normal", "skew_normal"],
    )
    selected_model = pl.model_selection_check(result_df, "loo", 25)
    assert selected_model == "double_normal"
    pass


def test_model_selection():
    """
    Test the model_selection function from the pipeline modul.
    The function contains the model selection pipeline.
    """
    path = Path(__file__).absolute().parent.parent / "test_data/test_model_selection"
    # Template.xlsx will be updated so copy it freshly and delete it in the end
    shutil.copy(path / "template/Template.xlsx", path / "Template.xlsx")
    result, model_dict = pl.model_selection(path)
    # make sure that the excluded model was really excluded
    assert "double_normal" not in result.index
    assert "normal" in result.index
    assert "skew_normal" in result.index
    assert model_dict
    os.remove(path / "Template.xlsx")
    pass
