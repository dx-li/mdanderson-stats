import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from mdanderson_stats.mtadf import MTADFPrior, mtadf_toxicity_prior
from mdanderson_stats.mtadf_logistic import (
    mtadf_local_logistic_decision,
    mtadf_local_logistic_posterior,
    mtadf_logistic_decision,
    mtadf_logistic_posterior,
)


def _local_slope_quadrature(n, y, doses):
    """Independent tensor Gauss-Legendre integral in transformed Cauchy coordinates."""
    nodes, weights = leggauss(120)
    angle = nodes * (np.pi / 2)
    weight = weights * (np.pi / 2)
    alpha = 10 * np.tan(angle)
    beta = 2.5 * np.tan(angle)
    eta = alpha[:, None, None] + beta[None, :, None] * np.asarray(doses)[None, None, :]
    log_likelihood = (
        np.asarray(y)[None, None, :] * -np.logaddexp(0.0, -eta)
        + (np.asarray(n) - np.asarray(y))[None, None, :] * -np.logaddexp(0.0, eta)
    ).sum(axis=2)
    kernel = np.exp(log_likelihood) * weight[:, None] * weight[None, :]
    return float(kernel[:, beta > 0].sum() / kernel.sum())


def test_global_quadratic_posterior_is_seed_replayable_and_bounded():
    args = ([4, 5, 4, 2], [0, 2, 3, 1], [1.0, 2.0, 3.0, 4.0])
    options = dict(draws=64, warmup=64, chains=2)
    first = mtadf_logistic_posterior(*args, rng=np.random.default_rng(71), **options)
    second = mtadf_logistic_posterior(*args, rng=np.random.default_rng(71), **options)
    assert first.parameter_draws.shape == (2, 64, 3)
    assert first.efficacy_draws.shape == (2, 64, 4)
    assert np.all((first.efficacy_draws >= 0) & (first.efficacy_draws <= 1))
    np.testing.assert_array_equal(first.parameter_draws, second.parameter_draws)
    np.testing.assert_array_equal(first.efficacy_draws, second.efficacy_draws)
    assert np.all((first.acceptance_rate >= 0) & (first.acceptance_rate <= 1))
    assert first.efficacy_summary.split_rhat.shape == (4,)


def test_global_initial_action_uses_the_safest_available_prior_dose():
    decision = mtadf_logistic_decision(
        [0, 0],
        [0, 0],
        [0, 0],
        [1.0, 2.0],
        current_dose=None,
    )
    assert decision.action == "start"
    assert decision.dose == 0
    assert decision.posterior is None
    assert np.all(np.isnan(decision.efficacy_mean))


def test_local_slope_probability_obeys_symmetric_design_reference():
    fit = mtadf_local_logistic_posterior(
        [12, 12],
        [6, 6],
        [-1.0, 1.0],
        current_dose=1,
        draws=512,
        warmup=512,
        chains=4,
        rng=np.random.default_rng(18),
    )
    assert fit.parameter_draws.shape == (4, 512, 2)
    assert fit.efficacy_draws.shape == (4, 512, 2)
    assert fit.probability_positive_slope == pytest.approx(0.5, abs=0.12)
    assert fit.probability_positive_slope_mcse < 0.05


def test_local_slope_probability_matches_independent_quadrature():
    n, y, doses = [20, 20], [8, 12], [-1.0, 1.0]
    reference = _local_slope_quadrature(n, y, doses)
    fit = mtadf_local_logistic_posterior(
        n,
        y,
        doses,
        current_dose=1,
        draws=3000,
        warmup=2000,
        chains=4,
        rng=np.random.default_rng(114),
    )
    tolerance = 5 * fit.probability_positive_slope_mcse + 0.015
    assert abs(fit.probability_positive_slope - reference) < tolerance


def test_global_decision_checks_fit_data_and_drops_unsafe_current_dose():
    n = [12, 12, 12]
    y = [1, 2, 3]
    doses = [1.0, 2.0, 3.0]
    fit = mtadf_logistic_posterior(
        n, y, doses, draws=64, warmup=64, chains=2, rng=np.random.default_rng(5)
    )
    decision = mtadf_logistic_decision(
        n,
        [0, 12, 12],
        y,
        doses,
        current_dose=2,
        posterior=fit,
        prior=mtadf_toxicity_prior(),
    )
    assert not decision.toxicity.admissible[2]
    assert decision.action == "treat"
    assert decision.dose == 0
    with pytest.raises(ValueError, match="match dose coding and efficacy counts"):
        mtadf_logistic_decision(n, [0, 0, 0], [1, 2, 2], doses, current_dose=1, posterior=fit)


def test_local_final_decision_uses_existing_isotonic_safety_and_selection():
    fit = mtadf_local_logistic_posterior(
        [10, 10, 10],
        [1, 6, 8],
        [1.0, 2.0, 3.0],
        current_dose=2,
        draws=32,
        warmup=32,
        chains=2,
        rng=np.random.default_rng(2),
    )
    decision = mtadf_local_logistic_decision(
        [10, 10, 10],
        [0, 0, 0],
        [1, 6, 8],
        [1.0, 2.0, 3.0],
        current_dose=None,
        posterior=fit,
        final=True,
    )
    assert decision.reason == "final_double_sided_isotonic"
    assert decision.final_decision is not None
    assert decision.dose == decision.final_decision.dose
    assert decision.action == decision.final_decision.action


def test_local_initial_action_stops_when_prior_makes_every_dose_unsafe():
    decision = mtadf_local_logistic_decision(
        [0, 0],
        [0, 0],
        [0, 0],
        [1.0, 2.0],
        current_dose=None,
        prior=MTADFPrior(alpha=100.0, beta=0.1),
    )
    assert decision.action == "stop"
    assert decision.dose is None
    assert not np.any(decision.toxicity.admissible)


def test_local_boundary_window_uses_first_dose_levels():
    fit = mtadf_local_logistic_posterior(
        [12, 12, 0],
        [3, 8, 0],
        [1.0, 2.0, 3.0],
        current_dose=0,
        window_length=2,
        draws=32,
        warmup=32,
        chains=2,
        rng=np.random.default_rng(29),
    )
    np.testing.assert_array_equal(fit.window_doses, [1.0, 2.0])
    np.testing.assert_array_equal(fit.window_subjects, [12, 12])
