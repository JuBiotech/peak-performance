import numpy as np
import scipy.stats as st

import pytest

from peak_performance import models

def test_initial_guesses():
    # define time and intensity for example with known result
    time = [2+0.1*x for x in range(17)]
    intensity = [1,5,3] + 11 * [1000] + [7,9,11]
    # define expected results
    expected_noise_width = np.ptp([1,5,3,7,9,11])
    expected_baseline_fit = st.linregress(time, intensity)
    # get the values from the initial guesses function
    slope, intercept, noise_width = models.initial_guesses(time, intensity)
    # compare the outcome with the expected values
    assert expected_baseline_fit.slope == slope
    assert expected_baseline_fit.intercept == intercept
    assert expected_noise_width == noise_width
    pass
