"""Focused numerical checks for the single-outcome bCRM core."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.bcrm_model import (
    BCRMCurve,
    bcrm_log_likelihood,
    bcrm_log_probabilities,
    bcrm_probabilities,
    fit_bcrm,
)


def test_curve_maps_skeleton_and_respects_bounded_asymptotes_and_log_tails():
    skeleton = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.70])
    curve = BCRMCurve(skeleton, alpha=3)
    assert_allclose(curve.probabilities([1])[0], skeleton, atol=2e-16, rtol=0)
    assert not curve.standardized_doses.flags.writeable

    bounded = BCRMCurve([0.12, 0.28, 0.62], alpha=1.2, lower=0.1, upper=0.8)
    assert_allclose(
        bounded.probabilities([1])[0], [0.12, 0.28, 0.62], atol=2e-16, rtol=0
    )
    logs = bcrm_log_probabilities([-1e6, 1e6], [0, 1], alpha=0)
    assert np.isfinite(logs).all()
    assert_allclose(np.exp(logs).sum(axis=-1), 1, atol=2e-16)
    assert_allclose(bcrm_log_likelihood([0], [1], [1], [0]), [-0.04858735157374206])


def test_prior_only_summaries_match_uniform_slope_moments():
    curve = BCRMCurve([0.05, 0.10, 0.20])
    result = fit_bcrm(curve, [0, 0, 0], [0, 0, 0], prior_bounds=(0, 3))
    assert result.beta_mean == pytest.approx(1.5, abs=2e-11)
    assert result.beta_sd == pytest.approx(3 / np.sqrt(12), abs=2e-11)
    assert_allclose(result.beta_interval, [0.075, 2.925], atol=2e-10)
    assert np.max(np.abs(result.dose_mean - result.plugin_dose_probability)) > 0.1
    assert result.integration_error < 1e-8


def test_grouped_likelihood_is_invariant_to_splitting_binomial_rows():
    curve = BCRMCurve([0.05, 0.10, 0.20, 0.35])
    grouped = bcrm_log_likelihood(curve.standardized_doses, [0, 1, 2, 2], [4, 4, 4, 4], [0.4, 1.2])
    split = bcrm_log_likelihood(
        [curve.standardized_doses[0], curve.standardized_doses[0], curve.standardized_doses[1],
         curve.standardized_doses[2], curve.standardized_doses[3]],
        [0, 0, 1, 2, 2], [2, 2, 4, 4, 4], [0.4, 1.2],
    )
    assert_allclose(grouped, split, atol=2e-14, rtol=0)
    result = fit_bcrm(curve, [0, 1, 2, 2], [4, 4, 4, 4])
    assert 0 <= result.beta_mean <= 3


def test_invalid_shapes_and_limits_fail_at_the_boundary():
    with pytest.raises(ValueError, match="strictly increasing"):
        BCRMCurve([0.1, 0.1])
    with pytest.raises(ValueError, match="beta-dose"):
        bcrm_log_probabilities(np.zeros(100), np.zeros(2001))
    with pytest.raises(ValueError, match="beta-dose"):
        bcrm_probabilities([], np.zeros(200_001))
    with pytest.raises(ValueError, match="total subjects"):
        bcrm_log_likelihood([0], [0], [10001], [1])
    with pytest.raises(ValueError, match="alpha"):
        BCRMCurve([0.1], alpha=51)


def test_multidimensional_beta_shape_and_negative_dose_interval_order():
    doses = np.array([-2.0, -1.0, 0.0, 1.0])
    slopes = np.arange(6, dtype=float).reshape(2, 3)
    assert bcrm_probabilities(doses, slopes).shape == (2, 3, 4)
    assert bcrm_log_probabilities(doses, 1.0).shape == (4, 2)
    assert bcrm_log_likelihood(doses, [0, 1, 2, 3], [3, 3, 3, 3], slopes).shape == (2, 3)

    curve = BCRMCurve([0.05, 0.10, 0.20])
    result = fit_bcrm(curve, [0, 1, 2], [4, 4, 4])
    assert np.all(np.diff(result.dose_interval, axis=1) >= 0)


def test_sharp_endpoint_posterior_is_resolved():
    curve = BCRMCurve([0.05])
    result = fit_bcrm(curve, [10_000], [10_000], prior_bounds=(0, 3))
    assert result.beta_mean < 0.001
    assert result.beta_interval[1] < 0.002
