"""Published examples, six-family inversion and analytically known distributions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    ParameterDistribution,
    solve_distribution_moments,
    solve_distribution_quantiles,
)


@pytest.mark.parametrize(
    "family,a,b",
    [
        ("beta", 2, 5),
        ("gamma", 3, 4),
        ("inverse_gamma", 5, 3),
        ("normal", -2, 4),
        ("lognormal", 0.4, 0.7),
        ("weibull", 2, 3),
    ],
)
def test_both_input_modes_recover_distribution(family, a, b):
    original = ParameterDistribution(family, a, b)
    by_moments = solve_distribution_moments(family, original.mean, original.variance)
    x = original.quantile([0.1, 0.9])
    by_quantiles = solve_distribution_quantiles(family, x, [0.1, 0.9])
    assert_allclose([by_moments.parameter1, by_moments.parameter2], [a, b], rtol=2e-9, atol=1e-12)
    assert_allclose(
        [by_quantiles.parameter1, by_quantiles.parameter2], [a, b], rtol=2e-8, atol=1e-12
    )
    assert_allclose(original.cdf(x) + original.sf(x), 1, atol=1e-14)


def test_guide_beta_examples_and_moment_existence():
    fit = solve_distribution_quantiles("beta", [0.1, 0.4], [0.025, 0.975])
    assert_allclose([fit.parameter1, fit.parameter2], [6.672, 22.036], atol=0.0006)
    moments = solve_distribution_moments("beta", 0.4, 0.01)
    assert_allclose([moments.parameter1, moments.parameter2], [9.2, 13.8], rtol=1e-14)
    forward = ParameterDistribution("beta", 7, 3)
    assert forward.mean == 0.7
    assert_allclose(forward.variance, 0.01909090909090909, rtol=1e-14)
    assert ParameterDistribution("inverse_gamma", 1, 3).mean == np.inf
    assert ParameterDistribution("inverse_gamma", 2, 3).variance == np.inf
    with pytest.raises(ValueError, match="beta requires"):
        solve_distribution_moments("beta", 0.4, 0.24)
    with pytest.raises(ValueError, match="exactly one"):
        solve_distribution_moments("normal", 0, 1, standard_deviation=1)


def test_analytic_quantiles_and_scale_invariance():
    p = np.array([0.2, 0.8])
    for family in ["gamma", "weibull"]:
        for scale in [1e-100, 1, 1e100]:
            fit = solve_distribution_quantiles(family, -np.log1p(-p) * scale, p)
            assert_allclose(fit.parameter1, 1, rtol=1e-9)
            assert_allclose(fit.parameter2 / scale, 1, rtol=1e-9)
    inverse = solve_distribution_quantiles("inverse_gamma", -3 / np.log(p), p)
    assert_allclose([inverse.parameter1, inverse.parameter2], [1, 3], rtol=1e-9)
    narrow = solve_distribution_moments("weibull", 1, 1e-12)
    assert_allclose(narrow.mean, 1, rtol=1e-10)
    assert_allclose(narrow.variance, 1e-12, rtol=1e-8)
    normal = solve_distribution_moments("normal", 3, standard_deviation=2)
    assert normal.parameter_names == ("mean", "variance")
    assert_allclose(normal.parameter2, 4, rtol=1e-15)
