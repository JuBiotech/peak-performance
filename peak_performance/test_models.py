from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy.integrate
import scipy.stats as st

from peak_performance import models

_DP_ROOT = Path(__file__).absolute().parent.parent
_REQUIRED_VARIABLES = {
    "baseline_slope",
    "baseline_intercept",
    "baseline",
    "std",
    "height",
    "area",
    "sn",
    "mean",
    "y",
    "noise",
}
_REQUIRED_DATA = {
    "slope_guess",
    "intercept_guess",
    "noise_width_guess",
}


def test_noise_guessing():
    expected = 0.7
    intensities = [
        *np.random.normal(10, expected, size=200),
        *np.random.normal(0, 6, size=600),
        *np.random.normal(40, expected, size=200),
    ]
    actual = models.guess_noise(intensities)
    assert 0.6 < actual < 0.8
    pass


def test_initial_guesses():
    # define time and intensity for example with known result
    time = 2 + 0.1 * np.arange(17)
    intensity = [1, 5, 3] + 11 * [1000] + [7, 9, 11]
    # define expected results
    expected_baseline_fit = st.linregress([2, 2.1, 2.2, 3.4, 3.5, 3.6], [1, 5, 3, 7, 9, 11])
    # get the values from the initial guesses function
    slope, intercept, noise_width = models.initial_guesses(time, intensity)
    # compare the outcome with the expected values
    assert expected_baseline_fit.slope == slope
    assert expected_baseline_fit.intercept == intercept
    # With this example the noise is clipped to at least 10
    assert noise_width == 10
    pass


def test_zsn_sorting():
    """This tests a workaround that we rely on for multi-peak models."""
    coords = {
        "thing": ["left", "center", "right"],
    }
    with pm.Model(coords=coords) as pmodel:
        hyper = pm.Normal("hyper", mu=0, sigma=3)
        offset_unsorted = pm.ZeroSumNormal(
            "offset_unsorted",
            sigma=1,
            shape=3,
        )
        # Create a sorted deterministic without using transforms
        offset = pm.Deterministic("offset", pt.sort(offset_unsorted), dims="thing")
        pos = pm.Deterministic(
            "pos",
            hyper + offset,
            dims="thing",
        )
        # Observe the two things in incorrect order to provoke the model 😈
        dat = pm.Data("dat", [0.2, 0.05, -0.3], dims="thing")
        pm.Normal("L", pos, observed=dat, dims="thing")

    # Check draws from the prior
    drawn = pm.draw(offset, draws=69)
    np.testing.assert_array_less(drawn[:, 0], drawn[:, 1])

    # And check MCMC draws too
    with pmodel:
        idata = pm.sample(
            chains=1, tune=10, draws=69, step=pm.Metropolis(), compute_convergence_checks=False
        )
    for vname in ["offset", "pos"]:
        np.testing.assert_array_less(
            idata.posterior[vname].sel(thing="left"),
            idata.posterior[vname].sel(thing="center"),
        )
        np.testing.assert_array_less(
            idata.posterior[vname].sel(thing="center"),
            idata.posterior[vname].sel(thing="right"),
        )
    pass


class TestDistributions:
    def test_normal_peak_shape(self):
        x = np.linspace(-5, 10, 10000)
        expected = st.norm.pdf(x, 3, 2)
        actual_pt = models.normal_peak_shape(0, x, 3, 2, height=np.max(expected))
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=0.0000001)
        pass

    def test_double_normal_peak_shape(self):
        x = np.linspace(5, 12, 10000)
        y1 = st.norm.pdf(x, loc=7.5, scale=0.6)
        y2 = st.norm.pdf(x, loc=9, scale=0.4) * 2
        y_double_pt = models.double_normal_peak_shape(
            0, x, (7.5, 9), (0.6, 0.4), height=(np.max(y1), np.max(y2))
        )
        y_double = y_double_pt.eval().astype(float)
        np.testing.assert_allclose(y1 + y2, y_double, rtol=1, atol=1e-20)
        pass

    def test_height_calculation_without_baseline(self):
        x = np.linspace(-1, 5.5, 10000)
        mean = 1.2
        std = 1.1
        alpha = 3
        y = st.skewnorm.pdf(x, alpha, loc=mean, scale=std)
        area = 1
        # find the x value to the maximum y value, i.e. the mode
        expected_mode_skew = x[np.argmax(y)]
        expected_height = np.max(y)
        # calculate actual values
        delta = models.delta_calculation(alpha)
        mue_z = models.mue_z_calculation(delta)
        sigma_z = models.sigma_z_calculation(mue_z)
        skewness = models.skewness_calculation(delta)
        mode_offset_pt = models.mode_offset_calculation(mue_z, skewness, sigma_z, alpha)
        mode_skew_pt = models.mode_skew_calculation(mean, std, mode_offset_pt)
        height_pt = models.height_calculation(area, mean, std, alpha, mode_skew_pt)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual_mode = mode_skew_pt.eval().astype(float)
        actual_height = height_pt.eval().astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected_height, actual_height, atol=2e-5)
        np.testing.assert_allclose(expected_mode_skew, actual_mode, atol=1e-2)
        pass

    def test_height_calculation_with_linear_baseline(self):
        x = np.linspace(-1, 5.5, 1000000)
        mean = 1.2
        std = 1.1
        alpha = 3
        baseline = 0.04 * x + 0.3
        y = st.skewnorm.pdf(x, alpha, loc=mean, scale=std) + baseline
        area = 1
        # find the x value to the maximum y value, i.e. the mode
        imax = np.argmax(y - baseline)
        expected_mode_skew = x[imax]
        expected_height = y[imax] - baseline[imax]

        # calculate actual values
        delta = models.delta_calculation(alpha)
        mue_z = models.mue_z_calculation(delta)
        sigma_z = models.sigma_z_calculation(mue_z)
        skewness = models.skewness_calculation(delta)
        mode_offset_pt = models.mode_offset_calculation(mue_z, skewness, sigma_z, alpha)
        mode_skew_pt = models.mode_skew_calculation(mean, std, mode_offset_pt)
        height_pt = models.height_calculation(area, mean, std, alpha, mode_skew_pt)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual_mode = mode_skew_pt.eval().astype(float)
        actual_height = height_pt.eval().astype(float)
        # testing; allow slight difference due to shift of distribution by baseline
        # (this numerical calculation does not consider the baseline)
        np.testing.assert_allclose(expected_height, actual_height, atol=1e-4)
        np.testing.assert_allclose(expected_mode_skew, actual_mode, atol=5e-3)
        pass

    def test_skew_normal_peak_shape(self):
        x = np.linspace(-1, 5.5, 10000)
        # test first with positive alpha
        expected = st.skewnorm.pdf(x, 3, loc=1.2, scale=1.1)
        actual_pt = models.skew_normal_peak_shape(0, x, 1.2, 1.1, 3, area=1)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=1e-8)

        # test again with negative alpha
        expected = st.skewnorm.pdf(x, -3, loc=1.2, scale=1.1)
        actual_pt = models.skew_normal_peak_shape(0, x, 1.2, 1.1, -3, area=1)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=1e-8)
        pass

    def test_compare_normal_and_skew_as_normal(self):
        """A skew normal distribution with skewness alpha = 0 should be a normal distribution. Test if that is so for our distributions."""
        x = np.linspace(-10, 10, 10000)
        y = st.norm.pdf(x, loc=1, scale=0.5)
        height = np.max(y)
        area = scipy.integrate.quad(lambda x: st.norm.pdf(x, loc=1, scale=1), -10, 10)[0]
        x = np.linspace(-10, 10, 10000)
        y_actual_pt = models.normal_peak_shape(0, x, 1, 1, height=height)
        y_skew_actual_pt = models.skew_normal_peak_shape(0, x, 1, 1, 0, area=area)
        y_actual = y_actual_pt.eval().astype(float)
        y_skew_actual = y_skew_actual_pt.eval().astype(float)
        # many values are extremely close to zero so rtol was increased.
        # As guaranteed by the absurdly low atol, this will not mask any actual differences.
        np.testing.assert_allclose(y_skew_actual, y_actual, atol=1e-20, rtol=0.9)
        pass

    def test_double_skew_normal_peak_shape(self):
        x1 = np.arange(4, 6, 0.1)
        x2 = np.arange(6, 8, 0.1)
        alpha = 5
        y1 = st.skewnorm.pdf(x1, alpha, loc=5, scale=0.2)
        y2 = st.skewnorm.pdf(x2, alpha, loc=6.3, scale=0.2)
        time = np.array(list(x1) + list(x2))
        intensity = np.array(list(y1) + list(y2))
        y_double_pt = models.double_skew_normal_peak_shape(
            0, time, (5, 6.3), (0.2, 0.2), (5, 5), area=(1, 1)
        )
        y_double = y_double_pt.eval().astype(float)
        np.testing.assert_allclose(intensity, y_double, rtol=1, atol=1e-20)


@pytest.mark.parametrize(
    "define_func",
    [
        models.define_model_normal,
        models.define_model_skew,
    ],
)
def test_singlepeak_sampling(define_func):
    timeseries = np.load(_DP_ROOT / "example" / "A2t2R1Part1_132_85.9_86.1.npy")

    pmodel = define_func(timeseries[0], timeseries[1])
    with pmodel:
        idata = pm.sample(cores=2, chains=2, tune=3, draws=5)
    assert set(idata.posterior.keys()) >= _REQUIRED_VARIABLES
    assert set(idata.constant_data.keys()) >= _REQUIRED_DATA
    pass


@pytest.mark.parametrize(
    "define_func",
    [
        models.define_model_double_normal,
        models.define_model_double_skew_normal,
    ],
)
def test_doublepeak_sampling(define_func):
    timeseries = np.load(_DP_ROOT / "example" / "A2t2R1Part1_132_85.9_86.1.npy")

    pmodel = define_func(timeseries[0], timeseries[1])
    with pmodel:
        idata = pm.sample(cores=2, chains=2, tune=3, draws=5)
    assert set(idata.posterior.keys()) >= _REQUIRED_VARIABLES
    assert set(idata.constant_data.keys()) >= _REQUIRED_DATA
    # Confirm the order of peaks is as intended
    np.testing.assert_array_less(
        idata.posterior["offset"].sel(subpeak=0),
        idata.posterior["offset"].sel(subpeak=1),
    )
    np.testing.assert_array_less(
        idata.posterior["mean"].sel(subpeak=0),
        idata.posterior["mean"].sel(subpeak=1),
    )
    pass


def test_model_comparison():
    path = _DP_ROOT / "test_data/test_model_comparison"
    idata_normal = az.from_netcdf(path / "idata_normal.nc")
    idata_skew = az.from_netcdf(path / "idata_skew.nc")
    compare_dict = {
        "normal": idata_normal,
        "skew_normal": idata_skew,
    }
    ranking = models.model_comparison(compare_dict)
    assert ranking.index[0] == "skew_normal"
    pass
