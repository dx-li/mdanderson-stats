import numpy as np
import pytest

from mdanderson_stats.bayes_factor_survival_calendar import (
    bayes_factor_survival_trial,
    simulate_bayes_factor_survival,
)


def test_explicit_calendar_ledger_includes_arrival_and_event_ties():
    trial = bayes_factor_survival_trial(
        [0.0, 1.0, 2.0],
        [1.0, 5.0, 2.0],
        check_times=[1.0, 2.0, 3.0],
        final_time=4.0,
        null_median=4.0,
        alternative_median_mode=5.5,
        censor_durations=[np.inf, 2.0, np.inf],
    )
    assert [float(time) for time, _ in trial.monitor_history] == [1.0, 2.0, 3.0]
    assert trial.monitor_history[0][1].events == 1
    assert trial.monitor_history[0][1].total_time == 1.0
    assert trial.monitor_history[1][1].events == 1
    assert trial.monitor_history[1][1].total_time == 2.0
    np.testing.assert_array_equal(trial.event_observed, [True, False, True])
    np.testing.assert_array_equal(trial.followup_times, [1.0, 2.0, 2.0])
    assert trial.final_monitor.events == 2
    assert trial.final_monitor.total_time == 5.0
    assert trial.stop_reason == "maximum_enrollment"
    assert trial.accrual_stop_time == 2.0


def test_positive_infinite_event_time_represents_unobserved_event():
    trial = bayes_factor_survival_trial(
        [0.0],
        [np.inf],
        check_times=[],
        final_time=5.0,
        null_median=4.0,
        alternative_median_mode=5.5,
    )
    np.testing.assert_array_equal(trial.event_observed, [False])
    np.testing.assert_array_equal(trial.followup_times, [5.0])
    assert trial.final_monitor.events == 0
    assert trial.final_monitor.total_time == 5.0


def test_fractional_absolute_event_time_tie_uses_calendar_comparison():
    trial = bayes_factor_survival_trial(
        [0.3],
        [0.4],
        check_times=[0.7],
        final_time=1.0,
        null_median=4.0,
        alternative_median_mode=5.5,
    )
    assert trial.monitor_history[0][1].events == 1
    assert trial.monitor_history[0][1].total_time == 0.4


def test_fractional_absolute_censor_tie_uses_exact_censor_exposure():
    trial = bayes_factor_survival_trial(
        [0.3],
        [0.5],
        check_times=[],
        final_time=0.7,
        null_median=4.0,
        alternative_median_mode=5.5,
        censor_durations=[0.4],
    )
    assert trial.final_monitor.events == 0
    assert trial.final_monitor.total_time == 0.4


def test_calendar_early_stop_stops_enrollment_but_keeps_prespecified_final_time():
    trial = bayes_factor_survival_trial(
        [0.0, 1.0, 2.0],
        [0.01, 3.0, 4.0],
        check_times=[0.1, 1.0],
        final_time=5.0,
        null_median=4.0,
        alternative_median_mode=5.5,
        inferiority_cutoff=0.499999,
        superiority_cutoff=0.9,
    )
    assert trial.stop_reason == "inferiority"
    assert trial.accrual_stop_time == 0.1
    assert trial.early_monitor is not None
    assert trial.final_time == 5.0
    assert len(trial.enrollment_times) == 1
    assert trial.final_monitor.events == 1

    after_full_tape = bayes_factor_survival_trial(
        [0.0, 1.0, 2.0],
        [0.01, 3.0, 4.0],
        check_times=[3.0],
        final_time=5.0,
        null_median=4.0,
        alternative_median_mode=5.5,
        inferiority_cutoff=0.499999,
        superiority_cutoff=0.9,
    )
    assert after_full_tape.stop_reason == "inferiority"
    assert after_full_tape.accrual_stop_time == 2.0
    assert len(after_full_tape.enrollment_times) == 3


def test_seeded_simulation_is_replayable_and_preflighted():
    args = dict(
        null_median=4.0,
        alternative_median_mode=5.5,
        true_median=5.0,
        accrual_rate=2.0,
        max_patients=3,
        repetitions=12,
        check_times=[1.0, 2.0],
        final_followup=4.0,
    )
    first = simulate_bayes_factor_survival(**args, seed=17)
    repeated = simulate_bayes_factor_survival(**args, seed=17)
    replayed = simulate_bayes_factor_survival(**args, trial_seed_pairs=first.trial_seed_pairs)
    np.testing.assert_array_equal(first.trial_seed_pairs, repeated.trial_seed_pairs)
    np.testing.assert_array_equal(first.patients_enrolled, repeated.patients_enrolled)
    np.testing.assert_array_equal(first.final_superiority, repeated.final_superiority)
    np.testing.assert_array_equal(first.patients_enrolled, replayed.patients_enrolled)
    np.testing.assert_array_equal(first.final_inferiority, replayed.final_inferiority)
    assert np.all(first.patients_enrolled >= 1)
    assert first.patient_count_quantiles.shape == (3,)
    with pytest.raises(ValueError, match="max_total_work"):
        simulate_bayes_factor_survival(**args, seed=17, max_total_work=1)
    with pytest.raises(ValueError, match="trial_seed_pairs"):
        simulate_bayes_factor_survival(
            **args, trial_seed_pairs=np.full((12, 2), -1, dtype=np.int64)
        )


def test_positive_final_followup_must_advance_large_calendar_horizon():
    with pytest.raises(ArithmeticError, match="final time is not representable"):
        simulate_bayes_factor_survival(
            null_median=4.0,
            alternative_median_mode=5.5,
            true_median=5.0,
            accrual_rate=1.0,
            max_patients=1,
            repetitions=1,
            check_times=[1e300],
            final_followup=1.0,
            seed=3,
        )
