import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet_adaptive_precision import _diagnostics
from mdanderson_stats.u2oet_gao_adaptive_precision import fit_u2oet_gao_adaptive_precision
from mdanderson_stats.u2oet_gao_fit import fit_u2oet_gao, u2oet_gao_parameter_names


def _inputs():
    names = u2oet_gao_parameter_names(2, 2)
    mean = np.array([-1.0, -0.2, 0.4, 0.7, 0.0, -0.2, 0.3, 0.2, 0.3, 0.0, np.log(1.2), 0.0])
    assert len(names) == mean.size
    sd = np.zeros(mean.size)
    sd[0] = 0.7
    counts = np.zeros((2, 2, 2, 2))
    counts[0, 0, 1, 0] = 3
    counts[0, 0, 0, 0] = 1
    toxicity_only = np.zeros((2, 2, 2))
    toxicity_only[1, 1, 1] = 2
    return mean, sd, counts, toxicity_only


def test_adaptive_gao_resumes_complete_chain_state_and_aggregates_draws():
    mean, sd, counts, toxicity_only = _inputs()
    utility = np.array([[0.0, 1.0], [2.0, 0.0]])
    adaptive_rng = np.random.default_rng(771)
    result = fit_u2oet_gao_adaptive_precision(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        toxicity_only=toxicity_only,
        utility=utility,
        target_mcse_ratio=0.001,
        max_draws_per_chain=23,
        initial_draws=8,
        batch_draws=8,
        warmup=4,
        chains=2,
        rng=adaptive_rng,
    )

    reference_rng = np.random.default_rng(771)
    first = fit_u2oet_gao(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        toxicity_only=toxicity_only,
        draws=8,
        warmup=4,
        chains=2,
        rng=reference_rng,
    )
    second = fit_u2oet_gao(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        toxicity_only=toxicity_only,
        draws=15,
        warmup=0,
        chains=2,
        initial=first.parameters[:, -1, :],
        rng=reference_rng,
    )
    assert result.draws_per_chain == 23
    assert result.termination == "draw_cap"
    assert not result.target_met
    assert result.fit.warmup == 4
    assert result.fit.likelihood_evaluations == (
        first.likelihood_evaluations + second.likelihood_evaluations + 2
    )
    assert result.fit.likelihood_work_units == (
        first.likelihood_work_units + second.likelihood_work_units + 2 * (2 * 2 * 2 * 2)
    )
    assert_allclose(
        result.fit.parameters, np.concatenate((first.parameters, second.parameters), axis=1)
    )
    assert_allclose(result.fit.joint, np.concatenate((first.joint, second.joint), axis=1))
    assert result.corner_indices == ((0, 0), (0, 1), (1, 0), (1, 1))
    corner_values = np.stack(
        [
            np.sum(result.fit.joint[:, :, i, j] * utility, axis=(-2, -1))
            for i, j in result.corner_indices
        ],
        axis=-1,
    )
    expected_sd, expected_mcse, expected_ratio = _diagnostics(corner_values)
    assert_allclose(result.posterior_sd, expected_sd)
    assert_allclose(result.mcse, expected_mcse)
    assert_allclose(result.mcse_ratio, expected_ratio)
    assert not result.fit.joint.flags.writeable


def test_aggregate_work_rejection_preserves_rng_state():
    mean, sd, counts, toxicity_only = _inputs()
    rng = np.random.default_rng(772)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_work"):
        fit_u2oet_gao_adaptive_precision(
            [1, 2],
            [1, 2],
            counts,
            prior_mean=mean,
            prior_sd=sd,
            toxicity_only=toxicity_only,
            utility=np.ones((2, 2)),
            target_mcse_ratio=0.01,
            max_draws_per_chain=16,
            initial_draws=8,
            warmup=2,
            chains=2,
            max_work=1,
            rng=rng,
        )
    assert rng.bit_generator.state == state


def test_zero_variance_corners_do_not_pass_precision_target():
    mean, sd, counts, _ = _inputs()
    sd[:] = 0
    result = fit_u2oet_gao_adaptive_precision(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        utility=np.zeros((2, 2)),
        target_mcse_ratio=0.05,
        max_draws_per_chain=15,
        initial_draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(773),
    )
    assert result.draws_per_chain == 8
    assert result.termination == "draw_cap"
    assert not result.target_met
    assert np.all(result.posterior_sd == 0)
    assert np.all(np.isnan(result.mcse_ratio))
