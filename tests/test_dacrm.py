"""Focused input, augmentation, reproducibility, and budget checks for DA-CRM."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.dacrm import DACRMPrior, dacrm_pending_probability, fit_dacrm


def _prior():
    return DACRMPrior([0, 1, 3], [2, 3], [4, 5])


def test_gamma_prior_and_pending_probability_formula_are_stable():
    prior = _prior()
    assert prior.breaks[-1] == 3
    assert_allclose(
        dacrm_pending_probability([0.2, 0.7], [0, 2]),
        [0.2, 0.7 * np.exp(-2) / (0.3 + 0.7 * np.exp(-2))],
    )
    assert_allclose(dacrm_pending_probability(0.5, 1000), 0, atol=0)
    with pytest.raises(ValueError, match="broadcast"):
        dacrm_pending_probability(np.ones((20, 1)) * 0.5, np.ones((1, 20)))
    with pytest.raises(ValueError):
        prior.shape.setflags(write=True)


def test_prior_only_and_zero_time_pending_paths_match_analytic_conditionals():
    result = fit_dacrm(
        [0.1, 0.25, 0.45],
        [1, 2],
        [-1, -1],
        [0, 0],
        prior=_prior(),
        target=0.3,
        rng=np.random.default_rng(123),
        draws=16,
        warmup=0,
        chains=2,
    )
    assert result.pending_probability_draws.shape == (2, 16, 2)
    assert_allclose(result.pending_probability_draws, result.dose_probability[:, :, [1, 2]])
    assert result.evaluations == 0
    assert result.pending_summary is not None


def test_mixed_pending_and_observed_outcomes_are_reproducible_and_immutable():
    args = ([0.05, 0.15, 0.3], [0, 1, 2, 1], [0, 1, -1, 0], [3, 1, 0.4, 3])
    options = dict(prior=_prior(), target=0.2, draws=16, warmup=8, chains=2)
    first = fit_dacrm(*args, rng=np.random.default_rng(812), **options)
    second = fit_dacrm(*args, rng=np.random.default_rng(812), **options)
    assert_allclose(first.alpha_draws, second.alpha_draws, atol=0)
    assert_allclose(first.hazard_draws, second.hazard_draws, atol=0)
    assert_allclose(first.dose_mean, second.dose_mean, atol=0)
    assert first.pending_indices.tolist() == [2]
    assert first.pending_probability.shape == (1,)
    assert first.dose_probability.shape == (2, 16, 3)
    assert np.all((first.dose_probability >= 0) & (first.dose_probability <= 1))
    assert np.all((first.pending_probability_draws >= 0) & (first.pending_probability_draws <= 1))
    with pytest.raises(ValueError):
        first.hazard_draws.setflags(write=True)


def test_interval_boundary_event_and_input_work_bounds():
    result = fit_dacrm(
        [0.1, 0.2],
        [0, 1],
        [1, 1],
        [0, 1],
        prior=_prior(),
        target=0.2,
        rng=np.random.default_rng(22),
        draws=8,
        warmup=0,
        chains=2,
    )
    assert np.all(result.hazard_draws[:, :, 1] > 0)
    with pytest.raises(ValueError, match="full window"):
        fit_dacrm(
            [0.1],
            [0],
            [0],
            [1],
            prior=_prior(),
            target=0.2,
            rng=np.random.default_rng(1),
            draws=8,
            warmup=0,
        )
    with pytest.raises(RuntimeError, match="max_evaluations"):
        fit_dacrm(
            [0.1, 0.2],
            [0],
            [0],
            [3],
            prior=_prior(),
            target=0.2,
            rng=np.random.default_rng(1),
            draws=8,
            warmup=0,
            max_evaluations=1,
        )


def test_wide_finite_hazard_draws_keep_parameter_summaries_representable():
    prior = DACRMPrior([0, 1], [2], [1e-300])
    result = fit_dacrm(
        [0.2],
        [0],
        [-1],
        [0],
        prior=prior,
        target=0.3,
        rng=np.random.default_rng(6),
        draws=8,
        warmup=0,
        chains=2,
    )
    assert np.all(np.isfinite(result.parameter_summary.standard_deviation))
    assert np.max(result.parameter_summary.mean) > 1e299
