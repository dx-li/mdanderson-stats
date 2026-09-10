import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import (
    calibrate_u2oet_prior,
    sample_u2oet_prior,
    u2oet_parameter_names,
    u2oet_prior_ess,
)


def test_independent_prior_draws_recover_underlying_truncated_normal_moments():
    names = u2oet_parameter_names(2, 2)
    mean, sd = np.zeros(12), np.full(12, 0.2)
    indices = [i for i, name in enumerate(names[:-1]) if ".slope." in name]
    mean[indices], sd[indices] = [-2, 0, 1, -10], 1
    prior = sample_u2oet_prior(
        [1, 2],
        [1, 2],
        efficacy_levels=2,
        toxicity_levels=2,
        prior_mean=mean,
        prior_sd=sd,
        draws=12000,
        rng=np.random.default_rng(7705),
    )
    # Independently evaluated in R using log dnorm/pnorm Mills ratios.
    reference = np.array(
        [0.373215532822841, 0.797884560802865, 1.28759997093918, 0.098093233962564]
    )
    x = prior.parameters[:, indices]
    assert np.all(x > 0)
    assert np.all(
        np.abs(x.mean(axis=0) - reference) < 5 * x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    )
    assert_allclose(prior.joint.sum(axis=(-2, -1)), 1, atol=2e-14)
    ess = u2oet_prior_ess(prior.joint)
    assert np.all(np.isfinite(ess.efficacy)) and np.all(ess.efficacy > 0)
    assert np.all(np.isfinite(ess.toxicity)) and np.all(ess.toxicity > 0)
    assert not prior.parameters.flags.writeable


def test_beta_moment_information_and_degenerate_cells():
    e, t = np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.1, 0.2, 0.3, 0.4])
    joint = np.stack((1 - e, e), -1)[:, :, None] * np.stack((1 - t, t), -1)[:, None, :]
    draws = np.broadcast_to(joint[:, None, None], (4, 2, 2, 2, 2)).copy()
    ess = u2oet_prior_ess(draws)
    assert_allclose(ess.efficacy, 4, atol=1e-13)
    assert_allclose(ess.toxicity, 14, atol=1e-13)
    assert_allclose([ess.mean, ess.maximum], [9, 14])
    constant = u2oet_prior_ess(np.full((4, 2, 2, 2, 2), 0.25))
    assert np.isinf(constant.mean)
    endpoint = np.zeros((4, 2, 2, 2, 2))
    endpoint[..., 0, 0] = 1
    assert np.isnan(u2oet_prior_ess(endpoint).mean)
    alternating = endpoint.copy()
    alternating[2:, ..., 0, 0] = 0
    alternating[2:, ..., 1, 1] = 1
    limit = u2oet_prior_ess(alternating)
    assert_allclose([limit.mean, limit.maximum], 0)


def test_pseudo_trial_averaging_tracks_elicited_efficacy_and_retains_diagnostics():
    means = []
    for level in (0, 1):
        scenario = np.zeros((2, 2, 2, 2))
        scenario[..., level, 0] = 1
        result = calibrate_u2oet_prior(
            [1, 2],
            [1, 2],
            scenario,
            repetitions=2,
            patients_per_pair=10,
            pseudo_prior_sd=0.5,
            draws=500,
            warmup=300,
            chains=2,
            rng=np.random.default_rng(7706),
        )
        assert_allclose(result.counts.sum(axis=(-2, -1)), 10)
        assert_allclose(result.counts[..., level, 0], 10)
        assert result.trial_split_rhat.shape == (2, 12)
        assert np.all(np.isfinite(result.trial_mcse))
        assert np.all(result.standard_error >= 0)
        means.append(result.prior_mean[0])
    assert means[0] < -0.1 and means[1] > 0.1
