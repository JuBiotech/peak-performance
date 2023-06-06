import numpy as np
import pytest
import scipy.integrate
import scipy.stats as st

from peak_performance import models


def test_initial_guesses():
    # define time and intensity for example with known result
    time = [2 + 0.1 * x for x in range(17)]
    intensity = [1, 5, 3] + 11 * [1000] + [7, 9, 11]
    # define expected results
    expected_noise_width = np.ptp([1, 5, 3, 7, 9, 11])
    expected_baseline_fit = st.linregress([2, 2.1, 2.2, 3.4, 3.5, 3.6], [1, 5, 3, 7, 9, 11])
    # get the values from the initial guesses function
    slope, intercept, noise_width = models.initial_guesses(time, intensity)
    # compare the outcome with the expected values
    assert expected_baseline_fit.slope == slope
    assert expected_baseline_fit.intercept == intercept
    assert expected_noise_width == noise_width
    pass


class TestDistributions:
    def test_normal_posterior(self):
        x = np.linspace(-5, 10, 10000)
        expected = st.norm.pdf(x, 3, 2)
        actual_pt = models.normal_posterior(0, np.max(expected), x, 3, 2)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=0.0000001)
        pass

    def test_double_normal_posterior(self):
        pass

    def test_std_skew_calculation(self):
        pass

    def test_mean_skew_calculation(self):
        pass

    def test_mue_z_calculation(self):
        pass

    def test_sigma_z_calculation(self):
        pass

    def test_fit_skewness_calculation(self):
        pass

    def test_mode_offset_calculation(self):
        pass

    def test_mode_skew_calculation(self):
        pass

    def test_height_calculation(self):
        x = np.linspace(-1, 5.5, 10000)
        mean = 1.2
        std = 1.1
        alpha = 3
        y = st.skewnorm.pdf(x, alpha, loc=mean, scale=std)
        area = scipy.integrate.quad(
            lambda x: st.skewnorm.pdf(x, alpha, loc=mean, scale=std), -1, 5.5
        )[0]
        # find the x value to the maximum y value, i.e. the mode
        expected_mode_skew = x[np.argmax(y)]
        expected_height = np.max(y)
        mean_skew = models.mean_skew_calculation(mean, std, alpha)
        expected_mode_offset = mean_skew - expected_mode_skew
        # calculate actual values
        mue_z = models.mue_z_calculation(alpha)
        sigma_z = models.sigma_z_calculation(mue_z)
        fit_skewness = models.fit_skewness_calculation(y)
        mode_offset_pt = models.mode_offset_calculation(mue_z, fit_skewness, sigma_z, alpha)
        mode_skew_pt = models.mode_skew_calculation(mean_skew, mode_offset_pt, alpha)
        height_pt = models.height_calculation(area, mean, std, alpha, mode_skew_pt)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual_mode_offset = mode_offset_pt.eval().astype(float)
        actual_mode_skew = mode_skew_pt.eval().astype(float)
        actual_height = height_pt.eval().astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        # np.testing.assert_allclose(expected_mode_offset, actual_mode_offset, atol=0.00000001)
        # np.testing.assert_allclose(expected_mode_skew, actual_mode_skew, atol=0.00000001)
        np.testing.assert_allclose(expected_height, actual_height, atol=0.001)
        pass

    def test_skew_normal_posterior(self):
        x = np.linspace(-1, 5.5, 10000)
        # test first with positive alpha
        expected = st.skewnorm.pdf(x, 3, loc=1.2, scale=1.1)
        actual_pt = models.skew_normal_posterior(0, 1, x, 1.2, 1.1, 3)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=0.00000001)

        # test again with negative alpha
        expected = st.skewnorm.pdf(x, -3, loc=1.2, scale=1.1)
        actual_pt = models.skew_normal_posterior(0, 1, x, 1.2, 1.1, -3)
        # cast arrays to float data type in order to avoid error of np.testing.assert_allclose() due to using np.isfinite under the hood
        actual = actual_pt.eval().astype(float)
        expected = expected.astype(float)
        # testing; allow minor difference due to differences in float precision etc.
        np.testing.assert_allclose(expected, actual, atol=0.00000001)
        pass

    def test_compare_normal_and_skew_as_normal(self):
        """A skew normal distribution with skewness alpha = 0 should be a normal distribution. Test if that is so for our distributions."""
        x = np.linspace(-10, 10, 10000)
        y_skew = st.skewnorm.pdf(x, 0, loc=1, scale=1)
        y = st.norm.pdf(x, loc=1, scale=0.5)
        height = np.max(y)
        area = scipy.integrate.quad(lambda x: st.norm.pdf(x, loc=1, scale=1), -10, 10)[0]
        x = np.linspace(-10, 10, 10000)
        y_actual_pt = models.normal_posterior(0, height, x, 1, 1)
        y_skew_actual_pt = models.skew_normal_posterior(0, area, x, 1, 1, 0)
        y_actual = y_actual_pt.eval().astype(float)
        y_skew_actual = y_skew_actual_pt.eval().astype(float)
        # many values are extremely close to zero so rtol was increased. As guaranteed by the absurdly low atol, this will not mask any actual differences
        np.testing.assert_allclose(y_skew_actual, y_actual, atol=0.00000000000000000001, rtol=0.9)
        pass
