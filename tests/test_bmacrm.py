"""Focused posterior checks for the Bayesian model-averaged CRM core."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from mdanderson_stats.bmacrm import fit_bmacrm


def test_no_data_retains_model_prior_and_analytic_overdose_probability():
    skeletons = np.array([[0.05, 0.15, 0.3], [0.08, 0.2, 0.4]])
    prior = np.array([0.25, 0.75])
    result = fit_bmacrm(skeletons, [0, 0, 0], [0, 0, 0], target=0.2, model_prior=prior)
    threshold = np.log(-np.log(0.2)) - np.log(-np.log(skeletons))
    assert_allclose(result.posterior_model_weights, prior)
    assert_allclose(result.model_log_evidence, 0, atol=0)
    assert_allclose(result.model_overdose_probability, norm.cdf(threshold / np.sqrt(2)), atol=2e-10)
    assert_allclose(result.alpha_mean, 0, atol=2e-9)
    assert_allclose(result.alpha_sd, np.sqrt(2), rtol=2e-8)
    assert_allclose(result.overdose_probability, prior @ result.model_overdose_probability)

    narrow = fit_bmacrm([0.05, 0.2], [0, 0], [0, 0], target=1e-100, prior_sd=1e-3)
    threshold = np.log(-np.log(1e-100)) - np.log(-np.log([0.05, 0.2]))
    assert_allclose(narrow.model_overdose_probability[0], norm.cdf(threshold / 1e-3), atol=1e-14)
    assert_allclose(narrow.alpha_mean, 0, atol=2e-12)
    assert_allclose(narrow.alpha_sd, 1e-3, rtol=2e-8)


def test_identical_models_preserve_prior_weight_ratios_and_single_model_fits():
    skeleton = np.array([0.05, 0.15, 0.3])
    duplicated = fit_bmacrm(
        np.stack([skeleton, skeleton]), [0, 1, 0], [3, 4, 2], target=0.25, model_prior=[1, 3]
    )
    assert_allclose(duplicated.posterior_model_weights, [0.25, 0.75], atol=2e-12)
    assert_allclose(duplicated.model_dose_mean[0], duplicated.model_dose_mean[1])

    single = fit_bmacrm(skeleton, [0, 1, 0], [3, 4, 2], target=0.25)
    assert single.skeletons.shape == (1, 3)
    assert single.model_dose_mean.shape == (1, 3)
    assert np.all(np.diff(single.dose_mean) > 0)

    repeated = fit_bmacrm([0.05, 0.15, 0.15, 0.3], [0, 1, 0, 0], [3, 4, 2, 1], target=0.25)
    assert_allclose(repeated.dose_mean[1], repeated.dose_mean[2], atol=2e-12)


def test_extreme_all_toxic_and_all_nontoxic_data_remain_finite_and_immutable():
    skeleton = [0.01, 0.05, 0.2, 0.5]
    for events in ([100, 100, 100, 100], [0, 0, 0, 0]):
        result = fit_bmacrm(skeleton, events, [100, 100, 100, 100], target=0.25)
        assert np.all(np.isfinite(result.model_log_evidence))
        assert np.all(np.isfinite(result.model_dose_mean))
        assert np.all(np.isfinite(result.model_overdose_probability))
        with pytest.raises(ValueError):
            result.dose_mean.setflags(write=True)


def test_invalid_counts_skeletons_and_integration_budget_are_rejected():
    with pytest.raises(ValueError, match="nondecreasing"):
        fit_bmacrm([0.1, 0.09], [0, 0], [1, 1], target=0.2)
    with pytest.raises(ValueError, match="integer counts"):
        fit_bmacrm([0.1], [0.5], [1], target=0.2)
    with pytest.raises(RuntimeError, match="max_evaluations"):
        fit_bmacrm([0.1, 0.2], [0, 1], [2, 2], target=0.2, max_evaluations=1)
