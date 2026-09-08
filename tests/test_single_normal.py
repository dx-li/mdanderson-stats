"""Normal-prior source comparisons and independent distribution moments."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_design_precision, single_normal_criterion

CASES = json.loads((Path(__file__).parent / "fixtures/single_normal.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_hermite_criterion(case):
    options = {k: v for k, v in case.items() if k not in ("variance", "value")}
    x, n = [-1, 0, 1, 2], [10, 20, 30, 40]
    if case["comparison"] is not None:
        x, n = (x, x), (n, n)
    result = single_normal_criterion(
        x, n, covariance=np.diag(case["variance"]), legacy_scale=True, **options
    )
    # Source pi/sqrt2 constants are rounded, unlike NumPy's constants.
    assert_allclose(result.value, case["value"], rtol=2e-8)


@pytest.mark.parametrize("legacy,fraction", [(False, 1), (True, 0.25)])
def test_correlated_prior_has_requested_moments(legacy, fraction):
    mean = np.array([0.2, 1.2])
    covariance = np.array([[0.02, 0.01], [0.01, 0.03]])
    result = single_normal_criterion([-1, 1], [50, 50], mean, covariance, legacy_scale=legacy)
    assert_allclose(result.weights @ result.parameters, mean, atol=1e-14)
    centered = result.parameters - mean
    assert_allclose(centered.T @ (result.weights[:, None] * centered), covariance * fraction)
    assert_allclose(result.weights.sum(), 1)
    assert_allclose(result.value, 1 / (result.weights @ (1 / result.local_criterion)))


def test_mixed_normal_lognormal_moments():
    mean = np.array([0.2, 0.0])
    covariance = np.array([[0.02, 0.01], [0.01, 0.03]])
    result = single_normal_criterion(
        [-1, 1], [50, 50], mean, covariance, lognormal=[False, True], order=12
    )
    expected_mean = [0.2, np.exp(0.03 / 2)]
    assert_allclose(result.weights @ result.parameters, expected_mean, rtol=1e-13)
    centered = result.parameters - expected_mean
    observed = centered.T @ (result.weights[:, None] * centered)
    expected = [[0.02, 0.01 * np.exp(0.015)], [0.01 * np.exp(0.015), np.expm1(0.03) * np.exp(0.03)]]
    assert_allclose(observed, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "criterion", ["slope_sd", "slope_variance", "quantile_sd", "quantile_variance"]
)
def test_point_prior_reduction(criterion):
    result = single_normal_criterion(
        [-1, 1], [50, 50], [0, 0], np.zeros((2, 2)), lognormal=[False, True], criterion=criterion
    )
    fixed = single_design_precision([-1, 1], [50, 50], [0, 1])
    assert_allclose(result.value, getattr(fixed, criterion))
    assert result.parameters.shape == (1, 2)


def test_rank_one_prior_and_allocation_scaling():
    covariance = np.full((2, 2), 0.001)
    args = ([-1, 1], [50, 50], [0.2, 1], covariance)
    r = single_normal_criterion(*args)
    higher = single_normal_criterion(*args, order=12)
    assert r.parameters.shape == (6, 2)
    assert_allclose(r.value, higher.value, rtol=1e-8)
    double = single_normal_criterion([-1, 1], [100, 100], [0.2, 1], covariance)
    assert_allclose(double.value, r.value / np.sqrt(2))


@pytest.mark.parametrize(
    "changes",
    [
        {"covariance": [[1, 2], [2, 1]]},
        {"covariance": [[1, 0.1], [0, 1]]},
        {"covariance": [[-1e-20, 0], [0, 0]]},
        {"covariance": [[1]]},
        {"mean": [0]},
        {"mean": [np.nan, 1]},
        {"lognormal": [0, 1]},
        {"lognormal": [True]},
        {"mean": [0, 1000], "lognormal": [False, True]},
        {"mean": [0, -1000], "lognormal": [False, True]},
        {"criterion": "arithmetic"},
        {"order": 1},
        {"order": 33},
        {"order": True},
        {"order": 6.5},
        {"legacy_scale": "yes"},
        {"quantile": [0.1, 0.2]},
    ],
)
def test_invalid_prior(changes):
    args = dict(doses=[-1, 1], subjects=[50, 50], mean=[0.2, 1], covariance=np.eye(2) * 0.001)
    args.update(changes)
    with pytest.raises(ValueError):
        single_normal_criterion(**args)
