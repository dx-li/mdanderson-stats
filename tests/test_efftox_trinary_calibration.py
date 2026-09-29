"""Focused tests for trinary EffTox prior moments and calibration."""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from mdanderson_stats.efftox_trinary_calibration import (
    calibrate_efftox_trinary_prior,
    efftox_trinary_prior_moments,
)
from mdanderson_stats.efftox_trinary_model import EffToxTrinaryPrior


def _reference_rows():
    path = Path(__file__).parent / "fixtures" / "efftox-trinary-prior-moments.csv"
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_fixed_trinary_prior_uses_product_second_moment_for_marginal_ess():
    prior = EffToxTrinaryPrior(mean=[-1.1, 0.8, 0.25, 1.2], sd=[0, 0, 0, 0])
    result = efftox_trinary_prior_moments([1, 2, 4], prior)
    x = np.log([1, 2, 4]) - np.mean(np.log([1, 2, 4]))
    toxicity = expit(-1.1 + 0.8 * x)
    conditional_efficacy = expit(0.25 + 1.2 * x)
    efficacy = (1 - toxicity) * conditional_efficacy
    np.testing.assert_allclose(result.toxicity_mean, toxicity, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        result.conditional_efficacy_mean, conditional_efficacy, rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(result.efficacy_mean, efficacy, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(result.efficacy_variance, 0)
    assert np.isposinf(result.efficacy_effective_sample_size).all()
    assert not result.efficacy_mean.flags.writeable


def test_nearly_fixed_coefficients_retain_positive_marginal_variance():
    prior = EffToxTrinaryPrior(mean=[-1.0, 0.8, 0.2, 1.1], sd=[1e-4, 0, 1e-4, 0])
    result = efftox_trinary_prior_moments([1, 2, 4], prior)
    assert np.all(result.toxicity_variance > 0)
    assert np.all(result.conditional_efficacy_variance > 0)
    assert np.all(result.efficacy_variance > 0)
    assert np.all(np.isfinite(result.efficacy_effective_sample_size))
    expected = [row for row in _reference_rows() if row["case"] == "near_fixed"]
    for field, expected_field in (
        ("toxicity_mean", "toxicity_mean"),
        ("toxicity_variance", "toxicity_variance"),
        ("conditional_efficacy_mean", "conditional_efficacy_mean"),
        ("conditional_efficacy_variance", "conditional_efficacy_variance"),
        ("efficacy_mean", "efficacy_mean"),
        ("efficacy_variance", "efficacy_variance"),
        ("efficacy_effective_sample_size", "efficacy_ess"),
        ("toxicity_effective_sample_size", "toxicity_ess"),
    ):
        np.testing.assert_allclose(
            getattr(result, field),
            [float(row[expected_field]) for row in expected],
            rtol=2e-5,
            atol=2e-15,
        )


@pytest.mark.parametrize("toxicity_intercept", [50.0, -50.0])
def test_extreme_logit_complements_preserve_marginal_probability_tails(toxicity_intercept):
    prior = EffToxTrinaryPrior(mean=[toxicity_intercept, 1.0, 0.0, 1.0], sd=np.zeros(4))
    result = efftox_trinary_prior_moments([1, 2], prior)
    x = np.log([1, 2]) - np.mean(np.log([1, 2]))
    expected_t = expit(toxicity_intercept + x)
    expected_q = expit(x)
    expected_e = expit(-(toxicity_intercept + x)) * expected_q
    np.testing.assert_allclose(result.toxicity_mean, expected_t, rtol=2e-15, atol=0)
    np.testing.assert_allclose(result.efficacy_mean, expected_e, rtol=2e-14, atol=0)


def test_calibration_matches_marginal_means_and_separate_ess_targets():
    result = calibrate_efftox_trinary_prior(
        [1, 2, 4],
        [0.15, 0.24, 0.34],
        [0.08, 0.14, 0.22],
        efficacy_target_ess=0.8,
        toxicity_target_ess=1.1,
        quadrature_order=32,
        max_iterations=400,
        max_evaluations=400,
    )
    assert result.prior.sd[1] > 0 and result.prior.sd[3] > 0
    assert np.all(np.diff(result.moments.toxicity_mean) >= 0)
    assert np.all(np.diff(result.moments.conditional_efficacy_mean) >= 0)
    assert result.efficacy_target_ess == 0.8
    assert result.toxicity_target_ess == 1.1
    np.testing.assert_allclose(
        result.efficacy_mean_residual,
        result.moments.efficacy_mean - result.efficacy_elicited_mean,
        rtol=0,
        atol=1e-15,
    )
    assert np.all(result.moments.efficacy_mean < 1 - result.moments.toxicity_mean)
    assert np.isfinite(result.efficacy_objective)
    expected_objective = (
        np.sum(result.efficacy_mean_residual**2)
        + 0.1 * (np.mean(result.moments.efficacy_effective_sample_size) - 0.8) ** 2
        + 0.02 * (result.prior.sd[2] - result.prior.sd[3]) ** 2
    )
    assert result.efficacy_objective == pytest.approx(expected_objective, rel=2e-12, abs=1e-14)


def test_marginal_efficacy_incompatible_with_fitted_toxicity_is_rejected():
    with pytest.raises(ValueError, match="below one minus fitted mean toxicity"):
        calibrate_efftox_trinary_prior(
            [1, 2, 4],
            [0.92, 0.92, 0.92],
            [0.2, 0.3, 0.4],
            quadrature_order=16,
            max_iterations=300,
            max_evaluations=300,
        )


def test_independent_base_r_fixture_for_random_intercept_product_moments():
    fixture = _reference_rows()
    specs = {
        "moderate": (-1.1, 0.45, 0.8, 0.25, 0.6, 1.2),
        "high_toxicity": (-0.2, 0.3, 0.4, 0.9, 0.5, 0.7),
        "low_efficacy": (-1.8, 0.7, 1.1, -0.6, 0.35, 0.9),
        "near_fixed": (-1.0, 1e-4, 0.8, 0.2, 1e-4, 1.1),
    }
    for name, (mu_t, sd_t, beta_t, mu_q, sd_q, beta_q) in specs.items():
        rows = [row for row in fixture if row["case"] == name]
        prior = EffToxTrinaryPrior([mu_t, beta_t, mu_q, beta_q], [sd_t, 0, sd_q, 0])
        actual = efftox_trinary_prior_moments([1, 2, 4], prior)
        for field in (
            "toxicity_mean",
            "toxicity_variance",
            "conditional_efficacy_mean",
            "conditional_efficacy_variance",
            "efficacy_mean",
            "efficacy_variance",
            "efficacy_effective_sample_size",
            "toxicity_effective_sample_size",
        ):
            expected_field = {
                "efficacy_effective_sample_size": "efficacy_ess",
                "toxicity_effective_sample_size": "toxicity_ess",
            }.get(field, field)
            np.testing.assert_allclose(
                getattr(actual, field),
                [float(row[expected_field]) for row in rows],
                rtol=2e-10,
                atol=2e-12,
            )
