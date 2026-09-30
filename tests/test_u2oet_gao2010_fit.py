import numpy as np
import pytest

from mdanderson_stats.u2oet_gao2010_fit import (
    fit_u2oet_gao2010,
    u2oet_gao2010_parameter_names,
)


def _fit_arguments() -> dict[str, object]:
    return {
        "doses1": [0.0, 1.0],
        "doses2": [0.0, 2.0],
        "counts": np.zeros((2, 2, 2, 2), dtype=int),
        "prior_mean": np.zeros(12),
        "prior_sd": np.zeros(12),
    }


def test_named_coordinate_order_is_explicit_and_threshold_major() -> None:
    names = u2oet_gao2010_parameter_names(3, 2)
    assert names[:6] == (
        "efficacy.intercept.1.agent1",
        "efficacy.intercept.1.agent2",
        "efficacy.slope.1.agent1",
        "efficacy.slope.1.agent2",
        "efficacy.intercept.2.agent1",
        "efficacy.intercept.2.agent2",
    )
    assert names[-1] == "association"
    assert len(names) == 17


def test_fixed_gaussian_and_association_fit_retains_readonly_draws() -> None:
    result = fit_u2oet_gao2010(
        **_fit_arguments(),
        draws=8,
        warmup=0,
        chains=2,
        fixed_association=0.25,
        rng=np.random.default_rng(42),
    )
    assert result.parameters.shape == (2, 8, 13)
    assert result.joint.shape == (2, 8, 2, 2, 2, 2)
    assert result.association_acceptance is None
    assert np.all(result.parameters[..., -1] == 0.25)
    assert np.allclose(result.joint.sum(axis=(-2, -1)), 1.0)
    assert not result.parameters.flags.writeable
    assert not result.joint.flags.writeable
    assert not result.log_likelihood.flags.writeable
    assert np.array_equal(result.doses1, [0.0, 1.0])
    assert np.array_equal(result.doses2, [0.0, 2.0])
    assert not result.doses1.flags.writeable
    assert result.likelihood_evaluations == 2
    assert result.likelihood_work_units == 2 * 16


def test_free_association_uses_uniform_independence_update() -> None:
    result = fit_u2oet_gao2010(
        **_fit_arguments(),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(11),
    )
    assert result.association_acceptance is not None
    assert np.array_equal(result.association_acceptance, np.ones(2))
    assert np.all(np.abs(result.parameters[..., -1]) <= 1.0)
    assert "Uniform(-1,1)" in result.prior_support


def test_invalid_initial_state_and_budget_fail_before_rng_advances() -> None:
    args = _fit_arguments()
    reference = np.random.default_rng(17)
    invalid = np.random.default_rng(17)
    args["prior_sd"] = np.eye(1, 12, 5).ravel() * 0.1
    starts = np.zeros((2, 13))
    starts[1, 5] = -2.0  # Invalid only in the second chain.
    with pytest.raises(ValueError, match="initial state is invalid"):
        fit_u2oet_gao2010(
            **args,
            initial=starts,
            fixed_association=0.0,
            draws=8,
            warmup=0,
            chains=2,
            rng=invalid,
        )
    assert invalid.random() == reference.random()

    invalid = np.random.default_rng(18)
    reference = np.random.default_rng(18)
    with pytest.raises(ValueError, match="minimum evaluations"):
        fit_u2oet_gao2010(
            **args,
            fixed_association=0.0,
            draws=8,
            warmup=0,
            chains=2,
            max_likelihood_evaluations=1,
            rng=invalid,
        )
    assert invalid.random() == reference.random()


def test_complete_and_toxicity_only_counts_enter_the_observed_likelihood() -> None:
    args = _fit_arguments()
    counts = np.zeros((2, 2, 2, 2), dtype=int)
    counts[0, 0, 1, 0] = 2
    toxicity_only = np.zeros((2, 2, 2), dtype=int)
    toxicity_only[1, 0, 1] = 3
    result = fit_u2oet_gao2010(
        **{**args, "counts": counts},
        toxicity_only=toxicity_only,
        draws=8,
        warmup=0,
        chains=2,
        fixed_association=0.0,
        rng=np.random.default_rng(22),
    )
    toxicity_marginal = result.joint.sum(axis=-2)
    expected = counts[0, 0, 1, 0] * np.log(result.joint[:, :, 0, 0, 1, 0])
    expected += toxicity_only[1, 0, 1] * np.log(toxicity_marginal[:, :, 1, 0, 1])
    assert np.allclose(result.log_likelihood, expected)


def test_negative_gamma_support_is_checked_jointly_over_the_supplied_grid() -> None:
    args = _fit_arguments()
    means = np.zeros(12)
    sds = np.zeros(12)
    # Efficacy alpha coordinates and gamma are jointly variable. Starts at
    # the center are valid, and retained draws must remain grid-valid.
    sds[[0, 5]] = [0.3, 0.2]
    result = fit_u2oet_gao2010(
        **{**args, "prior_mean": means, "prior_sd": sds},
        draws=8,
        warmup=2,
        chains=2,
        fixed_association=0.0,
        rng=np.random.default_rng(19),
    )
    assert np.all(np.isfinite(result.parameters))
    assert np.all(np.isfinite(result.joint))
    assert result.likelihood_evaluations > 2
