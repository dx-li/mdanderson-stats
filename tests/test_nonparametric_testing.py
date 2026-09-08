import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.nonparametric import NonparametricFitError
from mdanderson_stats.nonparametric_testing import nonparametric_testing
from mdanderson_stats.schweder import SchwederFitError, schweder_fit

REFERENCE = json.loads((Path(__file__).parent / "fixtures/nonparametric.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["desktop_cases"])
def test_original_desktop_np1p(case):
    result = nonparametric_testing(case["pvalues"], case["null_estimate"], alpha=case["alpha"])
    assert result.fitted_points == case["fitted_points"]
    assert case["status"] == 0
    if not result.fitted_points:
        # NP1P's early return leaves output arrays untouched in the original.
        np.testing.assert_array_equal(case["sorted_scores"], -999)
        np.testing.assert_array_equal(result.scores, 1)
        assert not np.any(result.reject)
        return
    if "high_precision_density" in case:
        np.testing.assert_allclose(result.density, case["high_precision_density"], rtol=1e-11)
        if case["name"] == "published":
            np.testing.assert_array_equal(result.bandwidths, case["bandwidths"])
    if case["name"] == "published" and result.fitted_points == 150:
        # Original normal equations lose accuracy; independent 70-digit solves
        # at the traced original windows above validate the corrected densities.
        assert np.max(np.abs(result.density - case["sorted_density"])) > 0.001
    else:
        np.testing.assert_allclose(result.density, case["sorted_density"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        result.scores[result.order], case["sorted_scores"], rtol=1e-7, atol=1e-9
    )
    np.testing.assert_array_equal(result.reject[result.order], case["sorted_reject"])


@pytest.mark.parametrize("fitted", [2, 5, 6, 10, 11, 25])
def test_all_fit_branches_use_full_family_empirical_ranks(fitted):
    n = 40
    values = np.r_[np.arange(1, fitted + 1) / (100 * n), np.linspace(0.5, 0.9, n - fitted)]
    result = nonparametric_testing(values, n - fitted + 0.9, alpha=0.02)
    assert result.fitted_points == fitted
    np.testing.assert_allclose(result.density, 100, rtol=1e-12)
    np.testing.assert_allclose(result.scores[:fitted], 0.01, rtol=1e-12)
    np.testing.assert_array_equal(result.reject, np.arange(n) < fitted)
    np.testing.assert_array_equal(result.scores[fitted:], 1)
    assert (result.bandwidths is None) == (fitted <= 10)


@pytest.mark.parametrize("fitted", [6, 10, 25])
def test_quadratic_empirical_cdf_and_narrow_coordinates(fitted):
    n = 40
    x = np.r_[np.sqrt(np.arange(1, fitted + 1) / n), np.linspace(0.9, 1, n - fitted)]
    result = nonparametric_testing(x, n - fitted)
    np.testing.assert_allclose(result.density, 2 * x[:fitted], rtol=1e-12)
    x = 0.75 + np.arange(1, n + 1) * 2.0**-30
    result = nonparametric_testing(x, n - fitted)
    np.testing.assert_allclose(result.density, 2.0**30 / n, rtol=1e-12)


def test_auto_schweder_and_failure_propagation():
    values = REFERENCE["cases"][0]["pvalues"]
    automatic = nonparametric_testing(values, alpha=0.2)
    explicit = nonparametric_testing(values, schweder_fit(values).null_estimate, alpha=0.2)
    np.testing.assert_array_equal(automatic.scores, explicit.scores)
    np.testing.assert_array_equal(automatic.reject, explicit.reject)
    with pytest.raises(SchwederFitError):
        nonparametric_testing([0.1] * 10)


@pytest.mark.parametrize("estimate", [9, 9.9, 10, 100])
def test_retain_all_branch_is_defined_even_at_alpha_one(estimate):
    result = nonparametric_testing(np.linspace(0.1, 0.9, 10), estimate, alpha=1)
    assert result.fitted_points == 0 and result.density.size == 0
    assert result.bandwidths is None
    np.testing.assert_array_equal(result.scores, 1)
    assert not np.any(result.reject)


def test_permutations_and_inclusive_threshold():
    values = np.array(REFERENCE["cases"][0]["pvalues"])
    saved = values.copy()
    first = nonparametric_testing(values, len(values) - 25)
    cutoff = first.scores[first.order[5]]
    result = nonparametric_testing(values, len(values) - 25, alpha=cutoff)
    assert result.reject[result.order[5]]
    permutation = np.random.default_rng(56).permutation(values.size)
    changed = nonparametric_testing(values[permutation], len(values) - 25, alpha=cutoff)
    np.testing.assert_allclose(changed.scores, result.scores[permutation], rtol=1e-13)
    np.testing.assert_array_equal(changed.reject, result.reject[permutation])
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize("estimate", [-1, np.nan, np.inf, [1, 2]])
def test_invalid_null_estimate(estimate):
    with pytest.raises(ValueError):
        nonparametric_testing([0.1, 0.2, 0.3, 0.4], estimate)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        nonparametric_testing([0.1, 0.2, 0.3, 0.4], 0, alpha=alpha)


@pytest.mark.parametrize("values", [[0.1] * 4, [0.1] * 3 + [0.2] * 3, [0.1] * 12])
def test_singular_or_undefined_fits_fail(values):
    with pytest.raises(NonparametricFitError):
        nonparametric_testing(values, 0)
