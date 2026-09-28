import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from mdanderson_stats.bard_blrm import (
    BARDLogisticPrior,
    bard_blrm_probability,
    fit_bard_blrm,
)


def test_blrm_uses_the_paper_raw_dose_ratio():
    doses = np.array([10.0, 20.0, 50.0, 100.0])
    actual = bard_blrm_probability(doses, 50.0, np.log(0.2), np.log(0.7))
    expected = expit(np.log(0.2) + 0.7 * doses / 50.0)
    assert_allclose(actual, expected, rtol=2e-15, atol=0)
    assert np.all(np.diff(actual) > 0)


def test_fixed_log_slope_fit_returns_coherent_target_and_overdose_summaries():
    doses = np.array([10.0, 20.0, 50.0])
    prior = BARDLogisticPrior([np.log(0.2), np.log(0.7)], [0.8, 0.0])
    result = fit_bard_blrm(
        doses,
        [3, 5, 8],
        [0, 1, 4],
        50.0,
        prior,
        target_interval=[0.2, 0.4],
        draws=24,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(17),
    )
    assert result.coefficient_draws.shape == (2, 24, 2)
    assert result.probability_draws.shape == (2, 24, 3)
    assert np.all(result.coefficient_draws[..., 1] == np.log(0.7))
    assert np.all(np.diff(result.probability_draws, axis=-1) > 0)
    assert np.all(
        (result.posterior_target_probability >= 0) & (result.posterior_target_probability <= 1)
    )
    assert np.all(
        (result.posterior_overdose_probability >= 0) & (result.posterior_overdose_probability <= 1)
    )
    assert np.all(result.posterior_target_probability + result.posterior_overdose_probability <= 1)
    assert result.work_units >= result.likelihood_evaluations * doses.size
    for array in (
        result.coefficient_draws,
        result.probability_draws,
        result.target_indicator_draws,
    ):
        assert not array.flags.writeable


def test_fixed_prior_is_exact_and_sampler_limits_are_checked():
    doses = np.array([10.0, 20.0])
    prior = BARDLogisticPrior([-2.0, 0.0], [0.0, 0.0])
    args = dict(
        doses=doses,
        patients=[2, 2],
        toxicities=[0, 1],
        reference_dose=10.0,
        prior=prior,
        target_interval=[0.1, 0.5],
        draws=8,
        warmup=0,
        chains=2,
    )
    result = fit_bard_blrm(**args, rng=np.random.default_rng(1))
    expected = expit(np.array([-2.0 + 1.0, -2.0 + 2.0]))
    assert_allclose(result.probability_draws, np.broadcast_to(expected, (2, 8, 2)))
    assert result.likelihood_evaluations == 2
    assert np.all(result.coefficient_draws == [-2.0, 0.0])
    assert_allclose(result.posterior_target_probability, [1.0, 0.0])
    assert_allclose(result.posterior_overdose_probability, [0.0, 1.0])

    rng = np.random.default_rng(23)
    state = rng.bit_generator.state.copy()
    with pytest.raises(ValueError, match="minimum likelihood-call count"):
        fit_bard_blrm(
            doses,
            [2, 2],
            [0, 1],
            10.0,
            BARDLogisticPrior([-2.0, -0.2], [0.5, 0.2]),
            target_interval=[0.1, 0.5],
            draws=8,
            warmup=0,
            chains=2,
            rng=rng,
            max_evaluations=1,
        )
    assert rng.bit_generator.state == state
