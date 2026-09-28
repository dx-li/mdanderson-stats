import numpy as np

from mdanderson_stats.multc_calendar import run_multc_calendar_trial
from mdanderson_stats.multc_core import multc_lean_design


def _design(**changes):
    values = dict(
        max_subjects=6,
        response_prior=(1.0, 1.0),
        toxicity_prior=(1.0, 1.0),
        historical_response=0.5,
        historical_toxicity=0.5,
        response_cutoff=1.0,
        toxicity_cutoff=0.95,
        min_subjects=2,
        cohort_size=2,
        pretrial_check=False,
    )
    values.update(changes)
    return multc_lean_design(**values)


def test_pending_outcomes_do_not_pause_when_stopping_is_impossible_at_look():
    result = run_multc_calendar_trial(
        _design(),
        outcomes=np.zeros((6, 2), dtype=np.int8),
        interarrival_intervals=np.ones(5),
        response_delays=np.zeros(6),
        toxicity_delays=np.full(6, 100.0),
    )
    assert result.looks[0].sample_size == 2
    assert result.looks[0].action == "continue"
    assert result.looks[0].toxicity_stop_min == 3
    np.testing.assert_array_equal(result.arrival_times[:4], [0.0, 1.0, 2.0, 3.0])
    assert result.looks[1].action == "wait"
    assert result.looks[2].action == "continue"
    assert result.arrival_times[4] == 101.0
    assert result.decision == "cap_complete"
    bounds = _design().stopping_bounds()
    assert result.looks[-1].response_stop_max == bounds.response_stop_max[-1]
    assert result.looks[-1].toxicity_stop_min == bounds.toxicity_stop_min[-1]
    assert result.duration == result.last_followup_time
    assert result.paused_duration == 97.0
    assert result.looks[0].toxicity_pending == 2

    changed_pending = np.zeros((6, 2), dtype=np.int8)
    changed_pending[:4, 1] = 1
    alternative = run_multc_calendar_trial(
        _design(),
        outcomes=changed_pending,
        interarrival_intervals=np.ones(5),
        response_delays=np.zeros(6),
        toxicity_delays=np.full(6, 100.0),
    )
    assert alternative.looks[0].action == result.looks[0].action == "continue"
    np.testing.assert_array_equal(alternative.arrival_times[:4], result.arrival_times[:4])


def test_pending_outcomes_pause_only_when_they_can_change_the_next_decision():
    design = _design(toxicity_prior=(1.8, 0.2), toxicity_cutoff=0.8)
    result = run_multc_calendar_trial(
        design,
        outcomes=np.zeros((6, 2), dtype=np.int8),
        interarrival_intervals=np.ones(5),
        response_delays=np.zeros(6),
        toxicity_delays=[10.0, 9.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert result.looks[0].action == "wait"
    assert result.looks[0].toxicity_stop_min == 1
    assert result.looks[1].action == "continue"
    assert result.looks[1].time == 10.0
    np.testing.assert_array_equal(result.arrival_times, [0.0, 1.0, 11.0, 12.0, 13.0, 14.0])
    assert result.paused_duration == 9.0
    assert result.observed_toxicities_at_decision == 0
    assert result.toxicities == 0
    assert result.duration == 14.0
    assert all(
        not row.flags.writeable
        for row in (
            result.outcomes,
            result.arrival_times,
            result.response_available_times,
            result.toxicity_available_times,
            result.response_known_at_decision,
            result.toxicity_known_at_decision,
        )
    )


def test_positive_time_increments_that_round_away_are_rejected():
    design = _design(max_subjects=4, toxicity_cutoff=1.0)
    outcomes = np.zeros((4, 2), dtype=np.int8)
    with np.testing.assert_raises_regex(ArithmeticError, "inter-arrival"):
        run_multc_calendar_trial(
            design,
            outcomes,
            interarrival_intervals=[1e20, 1.0, 1.0],
            response_delays=np.zeros(4),
            toxicity_delays=np.zeros(4),
        )

    with np.testing.assert_raises_regex(ArithmeticError, "endpoint delay"):
        run_multc_calendar_trial(
            design,
            outcomes,
            interarrival_intervals=[1e20, 1e20, 1e20],
            response_delays=[0.0, 1.0, 0.0, 0.0],
            toxicity_delays=np.zeros(4),
        )
