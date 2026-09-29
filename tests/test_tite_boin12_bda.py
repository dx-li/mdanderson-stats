import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12_bda import tite_boin12_bda_posterior


def test_complete_data_bda_summaries_reduce_to_boin12():
    design = BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))
    prior = np.asarray([1.2, 0.8, 0.3, 0.7])
    result = tite_boin12_bda_posterior(
        design,
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1.0, 1.0, 0.2, 0.3],
        [0.1, 2.0, 0.25, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
        prior_concentrations=prior,
        rng=np.random.default_rng(941),
        draws=24,
        warmup=8,
        chains=2,
    )
    expected = design.posterior([4], [2], [2], efficacy_without_toxicity=[1])
    np.testing.assert_allclose(result.mean_completed_joint_counts, [[1, 1, 1, 1]])
    np.testing.assert_allclose(
        result.posterior.toxicity_overdose_probability,
        expected.toxicity_overdose_probability,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.posterior.efficacy_futility_probability,
        expected.efficacy_futility_probability,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.posterior.utility_mean, expected.utility_mean, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        result.posterior.utility_probability,
        expected.utility_probability,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.posterior.utility_events, expected.utility_events, rtol=0.0, atol=1e-14
    )
    assert result.joint_probability_draws.shape == (2, 24, 1, 4)
    assert np.all(result.joint_probability_draws > 0)
    np.testing.assert_allclose(result.joint_probability_draws.sum(axis=-1), 1.0)


def test_pending_data_are_imputed_and_sampling_is_reproducible():
    design = BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))
    arguments = dict(
        design=design,
        doses=[1, 1, 1, 1],
        toxicity=[0, -1, 1, -1],
        efficacy=[1, 1, -1, -1],
        toxicity_followup=[1.0, 0.6, 0.1, 0.2],
        efficacy_followup=[1.0, 0.1, 0.25, 0.4],
        toxicity_window=1.0,
        efficacy_window=1.0,
        n_doses=1,
        prior_concentrations=[1.2, 0.8, 0.3, 0.7],
        draws=80,
        warmup=40,
        chains=2,
    )
    first = tite_boin12_bda_posterior(rng=np.random.default_rng(13), **arguments)
    second = tite_boin12_bda_posterior(rng=np.random.default_rng(13), **arguments)
    np.testing.assert_array_equal(first.joint_probability_draws, second.joint_probability_draws)
    np.testing.assert_array_equal(
        first.mean_completed_joint_counts, second.mean_completed_joint_counts
    )
    assert first.mean_completed_joint_counts.shape == (1, 4)
    assert first.mean_completed_joint_counts.sum() == pytest.approx(4.0)
    assert first.diagnostics.joint_probabilities.batch_mean_mcse.shape == (1, 4)
    assert first.diagnostics.boin12_metrics.split_rhat.shape == (1, 5)


def test_prior_must_be_explicit_and_work_is_preflighted():
    design = BOIN12Design(0.35, 0.25)
    common = dict(
        design=design,
        doses=[1],
        toxicity=[0],
        efficacy=[1],
        toxicity_followup=[1.0],
        efficacy_followup=[1.0],
        toxicity_window=1.0,
        efficacy_window=1.0,
        n_doses=1,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=np.random.default_rng(7),
        draws=20,
        warmup=0,
        chains=2,
    )
    with pytest.raises(ValueError, match="max_work"):
        tite_boin12_bda_posterior(**common, max_work=1)
    invalid_prior = dict(common)
    invalid_prior["prior_concentrations"] = [0.5, 0.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="prior_concentrations"):
        tite_boin12_bda_posterior(**invalid_prior)
