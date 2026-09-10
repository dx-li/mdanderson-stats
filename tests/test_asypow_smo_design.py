import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_design, asypow_smo_regression


def test_original_multivariable_logistic_likelihood():
    design = np.array([[1, -1, 0], [1, 0, 1], [1, 1, 0], [1, 2, 1]])
    p = np.array([-0.3, 0.4, 0.2])
    weights = np.arange(1, 5) / 10
    x = asypow_smo_design(p, design, constraints=[1, 3, 0], lower=-2, upper=2, observations=weights)
    assert_allclose(x.null_parameters, [-0.21867054462624594, 0.43734108925249193, 0], atol=1e-8)
    assert_allclose(x.divergence_per_observation, 0.0019670078069664587, rtol=1e-11)
    assert_allclose(x.sample_size(), 4498.6402585625783, rtol=1e-11)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    alt = 1 / (1 + np.exp(-design @ p))
    null = 1 / (1 + np.exp(-design @ x.null_parameters))
    assert_allclose(design[:, :2].T @ (weights * (alt - null)), 0, atol=1e-8)


def test_explicit_polynomial_designs_match_all_families():
    covariates = np.array([-1, 0, 1, 2])
    matrix = np.vander(covariates, 3, increasing=True)
    for family in ["logistic", "cloglog", "poisson", "exponential"]:
        kwargs = dict(
            family=family, constraints=[1, 3, 0], lower=-1, upper=1, observations=[1, 2, 3, 4]
        )
        if family == "exponential":
            kwargs["duration"] = 5
        a = asypow_smo_regression([-0.3, 0.2, 0.1], covariates, **kwargs)
        b = asypow_smo_design([-0.3, 0.2, 0.1], matrix, **kwargs)
        assert_allclose(a.null_parameters, b.null_parameters, atol=1e-12)
        assert_allclose(a.divergence_per_observation, b.divergence_per_observation, rtol=1e-12)


def test_design_without_intercept_and_unused_rows():
    p = 1 / (1 + np.exp(-0.2 * np.array([1, 2, 3])))
    expected = 2 * np.mean(p * np.log(2 * p) + (1 - p) * np.log(2 * (1 - p)))
    x = asypow_smo_design(
        [0.2],
        [[1], [2], [3], [1e300]],
        constraints=[1, 1, 0],
        lower=-1,
        upper=1,
        observations=[1e300, 1e300, 1e300, 0],
    )
    assert_allclose(x.divergence_per_observation, expected, rtol=1e-12)
    assert x.degrees_of_freedom == 1
    with pytest.raises(ValueError, match="identify"):
        asypow_smo_design([0.2, 0.3], [[1, 1], [2, 2]], constraints=[1, 1, 0], lower=-1, upper=1)
