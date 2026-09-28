import numpy as np

from mdanderson_stats.plbarpo_control_simulation import simulate_plbarpo_control
from mdanderson_stats.plbarpo_control_trial import run_plbarpo_control_trial


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
        control_mode="entire",
        method="barcp",
        tau=0.0,
        early_monitoring=False,
        pfinal=0.8,
    )


def test_control_simulation_aggregates_replayable_trials_and_null_errors():
    result = simulate_plbarpo_control(
        **_design(), trials=4, rng=302, null_arms=[False, True, False]
    )
    assert result.trials == 4
    assert result.total_work_units > 0
    assert result.metric_counts.shape == (len(result.metric_names), 3)
    assert result.metric_given_entry.shape == (len(result.metric_names), 3)
    assert result.metric_given_entry_mcse.shape == (len(result.metric_names), 3)
    assert np.isnan(result.metric_given_entry[:, 2]).all()
    assert np.isnan(result.metric_given_entry_mcse[:, 2]).all()
    assert result.metric_probability[0, :2].tolist() == [1.0, 1.0]
    assert result.metric_probability[0, 2] == 0.0
    assert result.familywise_false_efficacy_count is not None
    assert not result.trial_seeds.flags.writeable
    assert not result.metric_counts.flags.writeable

    null = np.asarray([False, True, False])
    replayed = [
        run_plbarpo_control_trial(**_design(), rng=int(seed)) for seed in result.trial_seeds
    ]
    replay_metrics = np.zeros_like(result.metric_counts)
    replay_assigned = []
    replay_responses = []
    replay_enrollment = []
    replay_total_responses = []
    replay_false = np.zeros(3, dtype=np.int64)
    replay_familywise = 0
    replay_no_efficacy = 0
    replay_early_stop = 0
    for trial in replayed:
        any_efficacy = trial.early_efficacy | trial.final_efficacy
        any_efficacy[0] = False
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
        replay_no_efficacy += int(not np.any(any_efficacy[1:]))
        replay_early_stop += int(trial.enrolled < trial.max_total_n)
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
    entry = replay_metrics[0]
    conditional = np.full_like(replay_probability, np.nan)
    np.divide(replay_metrics, entry[None, :], out=conditional, where=entry[None, :] > 0)
    conditional_mcse = np.full_like(conditional, np.nan)
    np.divide(
        np.sqrt(conditional * (1.0 - conditional)),
        np.sqrt(entry[None, :]),
        out=conditional_mcse,
        where=entry[None, :] > 0,
    )
    np.testing.assert_allclose(result.metric_given_entry, conditional, equal_nan=True)
    np.testing.assert_allclose(
        result.metric_given_entry_mcse, conditional_mcse, equal_nan=True
    )
    np.testing.assert_array_equal(result.false_efficacy_counts, replay_false)
    assert result.familywise_false_efficacy_count == replay_familywise
    assert result.no_efficacy_count == replay_no_efficacy
    assert result.early_stop_count == replay_early_stop

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
    for i, trial in enumerate(replayed):
        assert trial.rng_seeds == tuple(int(value) for value in result.stream_seeds[i])


def test_control_simulation_rejects_control_null_and_oversized_work_before_rng():
    random = np.random.default_rng(18)
    state = random.bit_generator.state
    with np.testing.assert_raises_regex(ValueError, "control"):
        simulate_plbarpo_control(**_design(), trials=2, null_arms=[True, False, False], rng=random)
    assert random.bit_generator.state == state
    with np.testing.assert_raises_regex(ValueError, "max_total_work"):
        simulate_plbarpo_control(
            **_design(), trials=4, max_total_work=1, rng=random
        )
    assert random.bit_generator.state == state
