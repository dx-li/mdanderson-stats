import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet_adaptive_precision import (
    _diagnostics,
    fit_u2oet_adaptive_precision,
)
from mdanderson_stats.u2oet_fit import fit_u2oet, u2oet_parameter_names


def _inputs(model: str = "cmi"):
    names = u2oet_parameter_names(2, 2, model=model)
    mean = np.zeros(len(names) - 1)
    sd = np.full(mean.size, 0.25)
    for index, name in enumerate(names[:-1]):
        if ".slope." in name:
            mean[index] = 0.8
    counts = np.zeros((2, 2, 2, 2))
    utility = np.array([[0.0, 1.0], [2.0, 0.0]])
    return mean, sd, counts, utility


def test_batch_means_are_computed_per_chain_and_corner():
    values = np.arange(2 * 16 * 4, dtype=float).reshape(2, 16, 4)
    sd, mcse, ratio = _diagnostics(values)
    batch = 4
    means = values.reshape(2, 4, batch, 4).mean(axis=2)
    expected_sd = values.std(axis=1, ddof=1)
    expected_mcse = means.std(axis=1, ddof=1) / np.sqrt(4)
    assert_allclose(sd, expected_sd)
    assert_allclose(mcse, expected_mcse)
    assert_allclose(ratio, expected_mcse / expected_sd)
    large = _diagnostics(values * 1e200)
    small = _diagnostics(values * 1e-200)
    assert_allclose(large[2], ratio, rtol=1e-14)
    assert_allclose(small[2], ratio, rtol=1e-14)
    assert_allclose(large[0] / 1e200, sd)
    assert_allclose(small[1] / 1e-200, mcse)


def test_extensions_resume_each_chain_from_its_last_state_without_reburning():
    mean, sd, counts, utility = _inputs()
    adaptive_rng = np.random.default_rng(671)
    result = fit_u2oet_adaptive_precision(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        utility=utility,
        target_mcse_ratio=0.001,
        max_draws_per_chain=23,
        initial_draws=8,
        batch_draws=8,
        warmup=4,
        chains=2,
        model="cmi",
        rng=adaptive_rng,
    )
    reference_rng = np.random.default_rng(671)
    first = fit_u2oet(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        draws=8,
        warmup=4,
        chains=2,
        model="cmi",
        rng=reference_rng,
    )
    second = fit_u2oet(
        [1, 2],
        [1, 2],
        counts,
        prior_mean=mean,
        prior_sd=sd,
        draws=15,
        warmup=0,
        chains=2,
        model="cmi",
        initial=first.parameters[:, -1, :],
        rng=reference_rng,
    )
    assert result.draws_per_chain == 23
    assert not result.target_met
    assert result.termination == "draw_cap"
    assert result.fit.warmup == 4
    assert_allclose(
        result.fit.parameters, np.concatenate((first.parameters, second.parameters), axis=1)
    )
    assert_allclose(result.fit.joint, np.concatenate((first.joint, second.joint), axis=1))
    assert result.corner_indices == ((0, 0), (0, 1), (1, 0), (1, 1))


def test_flat_corner_utilities_do_not_silently_pass_precision():
    mean, sd, counts, _ = _inputs()
    result = fit_u2oet_adaptive_precision(
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
        model="cmi",
        rng=np.random.default_rng(672),
    )
    assert not result.target_met
    assert result.termination == "draw_cap"
    assert result.draws_per_chain == 8
    assert result.draws_per_chain < result.max_draws_per_chain
    assert np.all(result.posterior_sd == 0)
    assert np.all(np.isnan(result.mcse_ratio))


def test_work_rejection_precedes_randomness():
    mean, sd, counts, utility = _inputs()
    rng = np.random.default_rng(673)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_work"):
        fit_u2oet_adaptive_precision(
            [1, 2],
            [1, 2],
            counts,
            prior_mean=mean,
            prior_sd=sd,
            utility=utility,
            target_mcse_ratio=0.01,
            max_draws_per_chain=8,
            initial_draws=8,
            warmup=0,
            chains=2,
            model="cmi",
            max_work=1,
            rng=rng,
        )
    assert rng.bit_generator.state == state
