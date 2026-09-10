import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.windows import window_cross_validation, window_smooth


def test_ten_estimators_match_original_fortran():
    fixture = json.loads((Path(__file__).parent / "fixtures/windows-native.json").read_text())
    for row in fixture:
        result = window_smooth(
            np.arange(7),
            [2, 0, 4, 1, 8, 3, 7],
            width=4,
            centers=[0, 2.5, 5],
            estimator=row["estimator"],
            weighting=row["weighting"],
            derivative_degree=2,
            polynomial_weights="native_inverse",
        )
        assert result.status == ("ok",) * 3
        assert_allclose(result.estimates, row["expected"], rtol=1e-5, atol=2e-6)


def test_polynomial_reproduction_derivatives_and_invalid_windows():
    x = np.arange(-4.0, 5.0)
    y = 2 + 3 * x - 0.5 * x * x
    for weighting in ("boxcar", "biquadratic"):
        fit = window_smooth(
            x, y, width=6, centers=[-2, 0, 2], estimator="quadratic", weighting=weighting
        )
        derivative = window_smooth(
            x,
            y,
            width=6,
            centers=[-2, 0, 2],
            estimator="derivative",
            derivative_degree=2,
            weighting=weighting,
        )
        assert_allclose(fit.estimates, [-6, 2, 6], atol=1e-12)
        assert_allclose(derivative.estimates, [5, 3, 1], atol=1e-12)
    bad = window_smooth([0, 0, 1], [2, 3, 4], width=0.1, centers=[0, 5], estimator="linear")
    assert bad.status == ("rank_deficient", "empty")
    assert np.isnan(bad.estimates).all()


def test_cross_validation_excludes_one_row_and_quantile_extrapolation():
    x = np.arange(6.0)
    result = window_cross_validation(x, 2 + 3 * x, [0.1, 4, 10], estimator="linear")
    assert np.isnan(result.sum_squared_errors[0])
    assert_allclose(result.sum_squared_errors[1:], 0, atol=1e-25)
    repeated = window_smooth([0, 0, 1], [1, 3, 9], width=0.1, leave_one_out=True)
    assert_allclose(repeated.estimates[:2], [3, 1])
    assert repeated.status[2] == "empty"
    low = window_smooth(
        [0, 1, 2], [9, 1, 5], width=4, centers=[1], estimator="quantile", quantile=0
    )
    high = window_smooth(
        [0, 1, 2], [9, 1, 5], width=4, centers=[1], estimator="quantile", quantile=1
    )
    assert_allclose(low.estimates, [-1])
    assert_allclose(high.estimates, [11])
