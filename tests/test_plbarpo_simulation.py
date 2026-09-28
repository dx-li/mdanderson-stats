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

    np.testing.assert_array_equal(
        result.metric_counts[result.metric_index("entry")], [4, 4, 0]
    )

    null = np.asarray([True, False, True])
    replayed = [
        run_plbarpo_trial(**_design(), rng=int(seed)) for seed in result.trial_seeds
    ]
    replay_metrics = np.zeros_like(result.metric_counts)
    replay_assigned = []
    replay_responses = []
    replay_enrollment = []
    replay_total_responses = []
    replay_false = np.zeros(3, dtype=np.int64)
    replay_familywise = 0
    for trial in replayed:
        any_efficacy = trial.early_efficacy | trial.final_efficacy
        masks = (
            trial.entered,
            trial.early_futility,
            trial.early_efficacy,
            trial.final_assessed,
            trial.final_efficacy,
            any_efficacy,
            trial.cap_closed,
        )
        replay_metrics += np.asarray(masks, dtype=np.int64)
        replay_assigned.append(trial.assigned)
        replay_responses.append(trial.successes)
        replay_enrollment.append(trial.enrolled)
        replay_total_responses.append(int(trial.successes.sum()))
        false = any_efficacy & null
        replay_false += false
        replay_familywise += int(np.any(false))
    np.testing.assert_array_equal(result.metric_counts, replay_metrics)
    replay_probability = replay_metrics / result.trials
    np.testing.assert_allclose(result.metric_probability, replay_probability, rtol=0, atol=0)
    np.testing.assert_allclose(
        result.metric_mcse,
        np.sqrt(replay_probability * (1.0 - replay_probability) / result.trials),
        rtol=0,
        atol=0,
    )
    assert result.false_efficacy_counts is not None
    np.testing.assert_array_equal(result.false_efficacy_counts, replay_false)
    assert result.familywise_false_efficacy_count == replay_familywise
    assert result.familywise_false_efficacy_probability == replay_familywise / result.trials
    assigned = np.asarray(replay_assigned, dtype=float)
    responses = np.asarray(replay_responses, dtype=float)
    np.testing.assert_allclose(result.assigned_mean, assigned.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(result.responses_mean, responses.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(
        result.assigned_mcse, assigned.std(axis=0, ddof=1) / np.sqrt(result.trials)
    )
    np.testing.assert_allclose(
        result.responses_mcse, responses.std(axis=0, ddof=1) / np.sqrt(result.trials)
    )
    total_n = np.asarray(replay_enrollment, dtype=float)
    total_y = np.asarray(replay_total_responses, dtype=float)
    assert result.total_enrollment_mean == total_n.mean()
    assert result.total_responses_mean == total_y.mean()
    assert result.total_enrollment_mcse == total_n.std(ddof=1) / np.sqrt(result.trials)
    assert result.total_responses_mcse == total_y.std(ddof=1) / np.sqrt(result.trials)

    for i, (seed, trial) in enumerate(zip(result.trial_seeds, replayed, strict=True)):
        assert trial.rng_seeds == tuple(int(value) for value in result.stream_seeds[i])

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
