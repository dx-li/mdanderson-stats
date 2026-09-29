import numpy as np
import pytest
from scipy.special import betaincc, gammaln

import mdanderson_stats.bacis_simulation as simulation_module
from mdanderson_stats.bacis import bacis_fit
from mdanderson_stats.bacis_ess import bacis_equivalent_sample_size
from mdanderson_stats.bacis_simulation import (
    _summarize_equivalent_sample_sizes,
    simulate_bacis_oc,
)


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
    assert result.equivalent_sample_size_by_replication is None
    assert not result.successes.flags.writeable


def test_ess_summary_uses_scaled_mean_and_reports_one_trial_mcse_as_none() -> None:
    mean, mcse = _summarize_equivalent_sample_sizes(np.array([[1e308, 0.0], [1e308, 1e308]]))
    np.testing.assert_array_equal(mean, [1e308, 5e307])
    np.testing.assert_array_equal(mcse, [0.0, 5e307])
    one_mean, one_mcse = _summarize_equivalent_sample_sizes(np.array([[7.0]]))
    np.testing.assert_array_equal(one_mean, [7.0])
    assert one_mcse is None


def test_seeded_multigroup_replay_metrics_and_stream_separation() -> None:
    first = simulate_bacis_oc(
        [0.1, 0.3, 0.2],
        trials_per_group=[10, 12, 8],
        replications=3,
        draws=8,
        warmup=0,
        chains=2,
        outcome_rng=101,
        sampler_rng=202,
    )
    replay = simulate_bacis_oc(
        [0.1, 0.3, 0.2],
        trials_per_group=[10, 12, 8],
        replications=3,
        draws=8,
        warmup=0,
        chains=2,
        outcome_rng=101,
        sampler_rng=202,
    )
    changed_sampler = simulate_bacis_oc(
        [0.1, 0.3, 0.2],
        trials_per_group=[10, 12, 8],
        replications=3,
        draws=8,
        warmup=0,
        chains=2,
        outcome_rng=101,
        sampler_rng=303,
    )
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


def test_optional_ess_replays_each_fit_and_averages_by_subgroup() -> None:
    sampler_seed = 654
    without_ess = simulate_bacis_oc(
        [0.3],
        trials_per_group=25,
        replications=2,
        draws=64,
        warmup=8,
        chains=2,
        outcome_rng=321,
        sampler_rng=sampler_seed,
    )
    with_ess = simulate_bacis_oc(
        [0.3],
        trials_per_group=25,
        replications=2,
        draws=64,
        warmup=8,
        chains=2,
        outcome_rng=321,
        sampler_rng=sampler_seed,
        compute_ess=True,
    )
    np.testing.assert_array_equal(with_ess.successes, without_ess.successes)
    np.testing.assert_array_equal(with_ess.efficacious, without_ess.efficacious)
    assert with_ess.equivalent_sample_size_by_replication is not None
    assert with_ess.mean_equivalent_sample_size is not None
    assert with_ess.equivalent_sample_size_mcse is not None

    sampler = np.random.default_rng(sampler_seed)
    fit_seeds = [int(sampler.integers(0, np.iinfo(np.int64).max)) for _ in range(2)]
    independent = np.empty((2, 1))
    for replication, seed in enumerate(fit_seeds):
        fit = bacis_fit(
            [int(with_ess.successes[replication, 0])],
            [25],
            draws=64,
            warmup=8,
            chains=2,
            seed=seed,
        )
        independent[replication] = bacis_equivalent_sample_size(
            fit.probability_samples,
            [int(with_ess.successes[replication, 0])],
            [25],
        ).equivalent_sample_size
    np.testing.assert_array_equal(with_ess.equivalent_sample_size_by_replication, independent)
    np.testing.assert_allclose(with_ess.mean_equivalent_sample_size, independent.mean(axis=0))
    np.testing.assert_allclose(
        with_ess.equivalent_sample_size_mcse,
        independent.std(axis=0, ddof=1) / np.sqrt(2),
    )
    assert not with_ess.equivalent_sample_size_by_replication.flags.writeable


def test_ess_work_preflight_and_one_replication_mcse_contract() -> None:
    outcome = np.random.default_rng(88)
    sampler = np.random.default_rng(89)
    outcome_before = outcome.bit_generator.state
    sampler_before = sampler.bit_generator.state
    with pytest.raises(ValueError, match="max_work"):
        simulate_bacis_oc(
            [0.3],
            replications=1,
            draws=8,
            warmup=0,
            chains=2,
            compute_ess=True,
            max_work=100,
            outcome_rng=outcome,
            sampler_rng=sampler,
        )
    assert outcome.bit_generator.state == outcome_before
    assert sampler.bit_generator.state == sampler_before


def test_ess_failure_reports_replication_and_subgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def fixed_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            probability_samples=np.full((2, 8, 1), 0.25),
            classification=SimpleNamespace(cluster=np.array([1])),
            efficacious=np.array([False]),
            summary=SimpleNamespace(split_rhat=np.array([1.0]), batch_mean_mcse=np.array([0.0])),
        )

    monkeypatch.setattr(simulation_module, "bacis_fit", fixed_fit)
    with pytest.raises(ValueError, match=r"replication 0, subgroup 0"):
        simulate_bacis_oc(
            [0.3],
            replications=1,
            draws=8,
            warmup=0,
            chains=2,
            compute_ess=True,
            outcome_rng=91,
            sampler_rng=92,
        )
