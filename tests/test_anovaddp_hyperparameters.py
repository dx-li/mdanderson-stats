import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import anovaddp_hyperparameter_update
from mdanderson_stats.anovaddp_hyperparameters import _inverse_wishart


def test_hyperparameter_conditionals_against_r():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/anovaddp-hyperparameters.json").read_text()
    )
    result = anovaddp_hyperparameter_update(
        [[2, 1, 2, 3, 4, 0.5], [2.1, 2, 1, 2, 3, 0.6], [1.8, 1, 1, 4, 5, 0.7]],
        [[1], [1], [1]],
        [0, 1, 0],
        [[1, 2, 3, 4, 0.5], [2, 1, 2, 3, 0.6]],
        base_covariance=np.eye(5),
        base_prior=np.zeros(5),
        covariance_prior=np.eye(6),
        covariance_df=8,
        concentration=1,
        concentration_shape=2,
        concentration_rate=1,
        seed=6712,
    )
    for actual, key in [
        (result.base_mean_posterior.mean, "mean"),
        (result.base_mean_posterior.covariance, "covariance"),
        (result.base_intercept_scale, "base_scale"),
        (result.residual_scale, "residual_scale"),
    ]:
        assert_allclose(actual, reference[key], atol=1e-13, rtol=0)
    assert_allclose(
        [result.concentration_mixture_probability, result.concentration_gamma_rate],
        reference["mixture"],
        atol=1e-14,
        rtol=0,
    )
    assert result.concentration_gamma_shape in (3, 4)
    assert result.concentration > 0
    assert result.base_intercept_df == 12 and result.residual_df == 11
    assert np.all(np.linalg.eigvalsh(result.residual_covariance) > 0)
    assert not np.array_equal(result.residual_covariance, np.eye(6))
    with pytest.raises(ValueError):
        result.base_covariance[0, 0] = 0
    fixed = anovaddp_hyperparameter_update(
        [[2, 1, 2, 3, 4, 0.5]],
        [[1, 0]],
        [0],
        [np.zeros(10)],
        base_covariance=np.eye(10),
        base_prior=np.zeros(10),
        covariance_prior=np.eye(6),
        covariance_df=8,
        concentration=1,
        concentration_shape=2,
        concentration_rate=1,
        seed=12,
    )
    assert np.array_equal(fixed.base_covariance[5:, 5:], np.eye(5))
    assert np.count_nonzero(fixed.base_covariance[:5, 5:]) == 0
    c = np.eye(10)
    c[0, 5] = c[5, 0] = 0.1
    with pytest.raises(ValueError, match="independent"):
        anovaddp_hyperparameter_update(
            [[2, 1, 2, 3, 4, 0.5]],
            [[1, 0]],
            [0],
            [np.zeros(10)],
            base_covariance=c,
            base_prior=np.zeros(10),
            covariance_prior=np.eye(6),
            covariance_df=8,
            concentration=1,
            concentration_shape=2,
            concentration_rate=1,
        )


def test_inverse_wishart_orientation_and_scale():
    scale = np.array([[2, 0.5], [0.5, 1]])
    rng = np.random.default_rng(6711)
    draws = np.array([_inverse_wishart(8, scale, rng) for _ in range(5000)])
    expected = scale / (8 - 2 - 1)
    se = draws.std(axis=0, ddof=1) / np.sqrt(5000)
    assert np.all(np.abs(draws.mean(axis=0) - expected) < 5 * se)
    assert np.all(np.linalg.eigvalsh(draws) > 0)
    # One-dimensional inverse Wishart is exactly inverse gamma(df/2,scale/2).
    value = _inverse_wishart(8, np.array([[4.0]]), np.random.default_rng(81))[0, 0]
    assert value == pytest.approx(4 / np.random.default_rng(81).chisquare(8), rel=1e-14)
