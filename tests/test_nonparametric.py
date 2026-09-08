import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.nonparametric import NonparametricFitError, nonparametric_pvalues

REFERENCE = json.loads((Path(__file__).parent / "fixtures/nonparametric.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_original_nonparametric_fit_and_explicit_failure_cases(case):
    if case["name"] == "random10":
        assert min(case["sorted_density"]) <= -1
        with pytest.raises(NonparametricFitError, match="beta-tail parameters"):
            nonparametric_pvalues(case["pvalues"])
        return
    result = nonparametric_pvalues(case["pvalues"])
    assert result.bandwidth == case["bandwidth"]
    if case["name"] == "narrow":
        expected = 2.0**30 / 20
        np.testing.assert_allclose(result.density, expected, rtol=2e-14)
        assert np.max(np.abs(np.array(case["sorted_density"]) - expected)) > expected / 10
        np.testing.assert_array_equal(result.scores, 0)
        return
    assert all(status == 0 for status in case["regression_status"])
    # The reference inverts uncentered normal equations, losing several digits.
    np.testing.assert_allclose(
        result.density[result.order], case["sorted_density"], rtol=1e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        result.scores[result.order], case["sorted_scores"], rtol=1e-7, atol=2e-9
    )


def test_uniform_grid_known_density_and_integer_binomial_tail():
    n = 20
    result = nonparametric_pvalues(np.arange(1, n + 1) / n)
    np.testing.assert_allclose(result.density, 1, rtol=2e-14)
    # Density exactly one gives Pr(Binomial(n,1/n) > 1).
    p = 1 / n
    expected = 1 - (1 - p) ** n - n * p * (1 - p) ** (n - 1)
    np.testing.assert_allclose(result.scores, expected, rtol=2e-14)


def test_quadratic_empirical_cdf_has_exact_derivative():
    # Rank/n = x**2: every nonsingular local quadratic has derivative 2*x.
    x = np.sqrt(np.arange(1, 31) / 30)
    result = nonparametric_pvalues(x)
    np.testing.assert_allclose(result.density, 2 * x, rtol=3e-14)
    assert np.all(np.diff(result.scores) >= 0)


def test_ties_permutations_and_input_preservation():
    values = np.array([0.01] * 10 + [0.1, 0.2, 0.3, 0.5, 0.8, 0.9])
    original = values.copy()
    result = nonparametric_pvalues(values)
    permutation = np.random.default_rng(12).permutation(values.size)
    changed = nonparametric_pvalues(values[permutation])
    np.testing.assert_allclose(changed.density, result.density[permutation], atol=1e-14)
    np.testing.assert_allclose(changed.scores, result.scores[permutation], atol=1e-14)
    np.testing.assert_array_equal(values, original)
    assert np.all(np.diff(result.scores[result.order]) >= 0)
    assert np.all((result.scores >= 0) & (result.scores <= 1))


@pytest.mark.parametrize(
    "values",
    [
        [],
        0.1,
        [0.1, 0.2, 0.3],
        [0.1] * 100,
        [0, 0, 0.5, 1, 1],
        [0, 0.1, np.nan, 0.5],
        [0, 0.1, 0.5, np.inf],
        [-0.1, 0.1, 0.2, 0.5],
        [0, 0.1, 0.2, 1.1],
        [[0.1, 0.2, 0.3, 0.4]],
    ],
)
def test_invalid_or_undefined_data_rejected(values):
    with pytest.raises(ValueError):
        nonparametric_pvalues(values)
