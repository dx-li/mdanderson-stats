import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_information, asypow_regression_information


@pytest.mark.parametrize("family", ["logistic", "cloglog", "poisson", "exponential"])
def test_original_r_two_group_quadratic_design(family):
    reference = json.loads((Path(__file__).parent / "fixtures/asypow-regression.json").read_text())
    theta = [[-0.7, 0.3, -0.1], [0.2, -0.4, 0.15]]
    x = asypow_regression_information(
        theta,
        [[-2, -1, 0, 1, 2], [-1, 0, 1, 2, 3]],
        family=family,
        observations=[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        group_size=[1, 3],
        **({"duration": [2, 5]} if family == "exponential" else {}),
    )
    assert_allclose(x, reference[family], rtol=1e-13, atol=1e-15)
    design = asypow_information(np.ravel(theta), x, [0, 1, 0, 0, -1, 0])
    assert_allclose(design.power(design.sample_size()), 0.8, atol=1e-12)
    assert not x.flags.writeable


def test_extreme_predictor_and_observation_scales():
    # Predictor tails remain usable when x^2 offsets a very small information weight.
    for family in ("logistic", "cloglog", "poisson"):
        x = asypow_regression_information([-1000, 0], [1e200], family=family)
        assert_allclose(x[1, 1], np.exp(400 * np.log(10) - 1000), rtol=1e-12)
    expected = asypow_regression_information([0.1, 0.2], [-1, 0, 1], observations=[1, 2, 3])
    scaled = asypow_regression_information(
        [0.1, 0.2], [-1, 0, 1], observations=[1e300, 2e300, 3e300], group_size=1e300
    )
    assert_allclose(scaled, expected, rtol=1e-12)
    omitted = asypow_regression_information(
        [0.1, 0.2], [-1, 0, 1, 1e308], observations=[1, 2, 3, 0]
    )
    assert_allclose(omitted, expected)
    with pytest.raises(ValueError, match="positive observations"):
        asypow_regression_information([0.1, 0.2], [0, 1], observations=0)


def test_survival_time_units_and_rare_event_limit():
    theta = np.array([-0.3, 0.2])
    base = asypow_regression_information(theta, [-1, 0, 1], family="exponential", duration=5)
    for scale in (1e-200, 1e200):
        other = theta.copy()
        other[0] -= np.log(scale)
        result = asypow_regression_information(
            other, [-1, 0, 1], family="exponential", duration=5 * scale
        )
        assert_allclose(result, base, rtol=1e-12, atol=1e-15)
    rare = asypow_regression_information([-500, 0], [0, 1], family="exponential", duration=1)
    assert_allclose(rare, np.exp(-500) / 2 * np.array([[1, 0.5], [0.5, 0.5]]), rtol=1e-13)
