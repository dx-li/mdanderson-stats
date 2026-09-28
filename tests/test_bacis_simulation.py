import numpy as np
import pytest
from scipy.special import betaincc, gammaln

from mdanderson_stats.bacis_simulation import simulate_bacis_oc


def test_single_group_exact_binomial_oracle_and_simulated_decisions() -> None:
    n = 25
    counts = np.arange(n + 1)
    decisions = betaincc(counts + 1, n - counts + 1, 0.1) > 0.92
    log_mass = (
        gammaln(n + 1)
        - gammaln(counts + 1)
        - gammaln(n - counts + 1)
        + counts * np.log(0.1)
        + (n - counts) * np.log(0.9)
    )
    exact_false_positive = float(np.exp(log_mass[decisions]).sum())
    assert np.flatnonzero(decisions).tolist() == list(range(5, 26))
    assert exact_false_positive == pytest.approx(0.0979936211954647, abs=2e-15)

    result = simulate_bacis_oc(
        [0.1],
        replications=40,
        draws=8,
        warmup=0,
        chains=2,
        outcome_rng=123,
        sampler_rng=456,
    )
    expected = (
        betaincc(
            result.successes[:, 0] + 1,
            n - result.successes[:, 0] + 1,
            0.1,
        )
        > 0.92
    )
    np.testing.assert_array_equal(result.efficacious[:, 0], expected)
    assert result.false_positive_rate[0] == np.mean(expected)
    assert result.familywise_false_positive_probability == np.mean(expected)
    assert result.power[0] != result.power[0]
    assert not result.successes.flags.writeable


def test_seeded_multigroup_replay_metrics_and_stream_separation() -> None:
    args = dict(
        true_response_rates=[0.1, 0.3, 0.2],
        trials_per_group=[10, 12, 8],
        replications=3,
        draws=8,
        warmup=0,
        chains=2,
    )
    first = simulate_bacis_oc(**args, outcome_rng=101, sampler_rng=202)
    replay = simulate_bacis_oc(**args, outcome_rng=101, sampler_rng=202)
    changed_sampler = simulate_bacis_oc(**args, outcome_rng=101, sampler_rng=303)
    np.testing.assert_array_equal(first.successes, replay.successes)
    np.testing.assert_array_equal(first.efficacious, replay.efficacious)
    np.testing.assert_array_equal(first.successes, changed_sampler.successes)
    np.testing.assert_array_equal(first.null_groups, [True, False, False])
    assert np.isnan(first.false_positive_rate[1:]).all()
    assert np.isnan(first.power[0])
    assert first.familywise_false_positive_probability in (0.0, 1 / 3, 2 / 3, 1.0)
    assert first.classification_high_probability.shape == (3,)
    assert first.max_split_rhat_by_replication.shape == (3,)
    assert first.nonfinite_diagnostic_by_replication.dtype == np.bool_


def test_invalid_work_and_shared_rngs_fail_before_consumption() -> None:
    outcome = np.random.default_rng(8)
    sampler = np.random.default_rng(8)
    outcome_before = outcome.bit_generator.state
    sampler_before = sampler.bit_generator.state
    with pytest.raises(ValueError, match="independent"):
        simulate_bacis_oc(
            [0.1],
            replications=1,
            draws=8,
            warmup=0,
            chains=2,
            outcome_rng=outcome,
            sampler_rng=outcome,
        )
    assert outcome.bit_generator.state == outcome_before
    assert sampler.bit_generator.state == sampler_before
    with pytest.raises(ValueError, match="max_work"):
        simulate_bacis_oc(
            [0.1],
            replications=2,
            draws=8,
            warmup=0,
            chains=2,
            max_work=10,
            outcome_rng=outcome,
            sampler_rng=sampler,
        )
    assert outcome.bit_generator.state == outcome_before
    assert sampler.bit_generator.state == sampler_before
