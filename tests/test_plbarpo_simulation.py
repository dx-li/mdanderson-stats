import numpy as np

from mdanderson_stats.plbarpo_simulation import simulate_plbarpo
from mdanderson_stats.plbarpo_trial import run_plbarpo_trial


def _design():
    return dict(
        true_response=[0.25, 0.7, 0.1],
        prior=np.ones((3, 2)),
        initial_active=[True, True, False],
        candidate_order=[],
        min_n_per_arm=[1, 1, 1],
        max_n_per_arm=[4, 4, 4],
        max_total_n=4,
        look_sizes=[2, 4],
        burn_in_per_arm=1,
        early_monitoring=False,
        theta_fut=None,
        pfut=None,
        theta_eff=None,
        peff=None,
        theta_final=0.5,
        pfinal=0.8,
    )


def test_simulation_aggregates_replayable_trials_and_explicit_null_errors():
    result = simulate_plbarpo(
        **_design(), trials=4, rng=302, null_arms=[True, False, True]
    )
    assert result.trials == 4
    assert result.total_work_units > 0
    assert result.metric_counts.shape == (len(result.metric_names), 3)
    assert result.metric_given_entry.shape == (len(result.metric_names), 3)
    assert np.isnan(result.metric_given_entry[:, 2]).all()
    assert result.metric_probability[0, :2].tolist() == [1.0, 1.0]
    assert result.familywise_false_efficacy_count is not None
    assert result.familywise_false_efficacy_probability == (
        result.familywise_false_efficacy_count / result.trials
    )
    assert not result.trial_seeds.flags.writeable
    assert not result.metric_counts.flags.writeable

    replay = run_plbarpo_trial(**_design(), rng=int(result.trial_seeds[0]))
    np.testing.assert_array_equal(
        result.metric_counts[result.metric_index("entry")], [4, 4, 0]
    )
    assert replay.enrolled <= 4
    assert replay.rng_seeds == tuple(int(value) for value in result.stream_seeds[0])
    one_trial = simulate_plbarpo(**_design(), trials=1, rng=int(result.trial_seeds[0]))
    np.testing.assert_array_equal(one_trial.assigned_mean, replay.assigned)
    np.testing.assert_array_equal(one_trial.responses_mean, replay.successes)
    assert one_trial.metric_counts[one_trial.metric_index("any_efficacy")].tolist() == (
        replay.early_efficacy | replay.final_efficacy
    ).tolist()

    repeated = simulate_plbarpo(
        **_design(), trials=4, rng=302, null_arms=[True, False, True]
    )
    np.testing.assert_array_equal(result.trial_seeds, repeated.trial_seeds)
    np.testing.assert_array_equal(result.metric_counts, repeated.metric_counts)


def test_simulation_preflights_work_and_requires_explicit_null_mask_shape():
    with np.testing.assert_raises(ValueError):
        simulate_plbarpo(**_design(), trials=100, rng=1, max_total_work=1)
    with np.testing.assert_raises(ValueError):
        simulate_plbarpo(**_design(), trials=2, rng=1, null_arms=[True, False])
