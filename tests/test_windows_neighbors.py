import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.windows import window_neighbor_cross_validation, window_smooth
from mdanderson_stats.windows_neighbors import _neighbor_bounds, _neighbor_weights


def test_original_fortran_estimators_and_boundary_weights():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/windows-neighbor-native.json").read_text()
    )
    for row in fixture:
        result = window_smooth(
            np.arange(9),
            [2, 0, 4, 1, 8, 3, 7, 2, 9],
            neighbors=5,
            centers=[2.5, 4.5, 5.5],
            estimator=row["estimator"],
            weighting=row["weighting"],
            derivative_degree=2,
            polynomial_weights="native_inverse",
        )
        assert_allclose(result.estimates, row["expected"], rtol=1e-5, atol=2e-6)
    for weighting, expected in [
        ("boxcar", [3 / 16, 3 / 16, 1 / 4, 3 / 16, 3 / 16]),
        ("biquadratic", [27 / 172, 27 / 172, 16 / 43, 27 / 172, 27 / 172]),
    ]:
        weights = _neighbor_weights(np.array([0.0, 0, 1, 2, 2]), 1, 4, weighting)
        assert_allclose(weights / weights.sum(), expected, rtol=1e-14)


def test_membership_matches_exhaustive_distances_and_includes_ties():
    rng = np.random.default_rng(54)
    for n in [7, 40, 101]:
        x = np.sort(rng.normal(size=n))
        for center in [-20, -0.1, 0, 1, 20]:
            for k in [2, 5, n]:
                low, high = _neighbor_bounds(x, center, k)
                distance = np.abs(x - center)
                radius = np.partition(distance, k - 1)[k - 1]
                assert_array_equal(np.arange(low, high), np.flatnonzero(distance <= radius))
    result = window_smooth([0, 1, 2], [1, 5, 9], neighbors=2, centers=[1], estimator="count")
    assert_array_equal(result.counts, [3])
    irregular = window_smooth([0, 1, 1.1, 100], [0, 1, 2, 100], neighbors=2, centers=[1.1])
    assert_allclose(irregular.estimates, [1.5])


def test_polynomials_zero_weights_and_neighbor_cross_validation():
    x = np.arange(9.0)
    y = 2 + 3 * x - 0.5 * x * x
    fit = window_smooth(
        x, y, neighbors=5, centers=[0, 4, 8], estimator="quadratic", weighting="biquadratic"
    )
    assert_allclose(fit.estimates, [2, 6, -6], atol=1e-12)
    cv = window_neighbor_cross_validation(
        x, y, [2, 5, 7], estimator="quadratic", weighting="biquadratic"
    )
    assert np.isnan(cv.sum_squared_errors[0])
    assert_allclose(cv.sum_squared_errors[1:], 0, atol=1e-23)
    assert cv.best_neighbors in (5, 7)
    result = window_smooth(
        [0, 0, 0], [1, 3, 8], neighbors=2, centers=[0, 1], weighting="biquadratic"
    )
    assert_allclose(result.estimates[0], 4)
    assert result.status[1] == "zero_weight"


def test_neighbor_cross_validation_matches_independent_r_fits():
    fixture = json.loads((Path(__file__).parent / "fixtures/windows-neighbor-cv.json").read_text())
    y = np.array([2.0, 0, 4, 1, 8, 3, 7, 2, 9])
    for row in fixture:
        kwargs = {name: row[name] for name in ("estimator", "weighting", "polynomial_weights")}
        fit = window_smooth(np.arange(9), y, neighbors=5, leave_one_out=True, **kwargs)
        assert_allclose(fit.estimates, row["expected"], rtol=1e-12, atol=1e-12)
        cv = window_neighbor_cross_validation(np.arange(9), y, [5], **kwargs)
        assert_allclose(cv.sum_squared_errors, [np.sum((y - row["expected"]) ** 2)], rtol=1e-12)
