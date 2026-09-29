import numpy as np

from mdanderson_stats.arand_calendar import (
    ArandControllerPolicy,
    arand_calendar_replay,
)


def _policy(**updates):
    values = {
        "floor_transform": "mixture",
        "ranking_scope": "all_arms",
        "trigger_order": ("futility", "suspension", "early_winner"),
        "duration_minimum_precedence": "duration_wins",
        "multiple_winner": "highest_probability",
        "all_suspended_action": "stop",
        "same_time_order": "arrival_before_analysis",
    }
    values.update(updates)
    return ArandControllerPolicy(**values)


def test_initial_equal_randomization_and_binary_followup_completeness():
    result = arand_calendar_replay(
        [0.0, 1.0],
        [0.1, 0.8],
        prior=[[1.0, 1.0], [1.0, 1.0]],
        family="binary",
        analysis_times=[],
        max_enrollment=2,
        max_duration=None,
        final_followup=0.5,
        minimum_enrollment=2,
        initial_equal_randomization=2,
        binary_outcomes=[[1, 0], [0, 1]],
        binary_delays=[[2.0, 2.0], [2.0, 2.0]],
        policy=_policy(),
    )

    np.testing.assert_array_equal(result.assignments, [0, 1])
    assert result.looks[0].status == "assigned"
    assert result.looks[0].posterior_probability is None
    assert result.final_analysis_time == 1.5
    assert result.final_winner is None
    assert result.reason == "final_followup_incomplete"


def test_survival_event_becomes_visible_at_its_relative_delay():
    result = arand_calendar_replay(
        [0.0],
        [0.0],
        prior=[[2.0, 1.0]],
        family="exponential",
        analysis_times=[0.25, 0.5],
        max_enrollment=1,
        max_duration=None,
        final_followup=1.0,
        minimum_enrollment=1,
        event_times=[[0.5]],
        policy=_policy(),
    )

    by_time = {look.time: look for look in result.looks}
    assert by_time[0.25].observed == 0
    assert by_time[0.5].observed == 1
    assert by_time[1.0].observed == 1


def test_duration_minimum_policy_reports_tape_exhaustion_without_winner():
    result = arand_calendar_replay(
        [0.0, 1.0],
        [0.0, 0.0],
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[0.5],
        max_enrollment=None,
        max_duration=0.5,
        final_followup=1.0,
        minimum_enrollment=3,
        binary_outcomes=[[0], [1]],
        binary_delays=[[0.0], [0.0]],
        policy=_policy(duration_minimum_precedence="minimum_enrollment_wins"),
    )

    assert result.assignments.size == 2
    assert result.reason == "minimum_not_reached_input_exhausted"
    assert result.final_analysis_time is None
    assert result.final_winner is None
    assert result.completed_duration == 1.0


def test_suspension_reverses_when_a_delayed_binary_outcome_arrives():
    result = arand_calendar_replay(
        [0.0, 1.0],
        [0.1, 0.8],
        prior=[[1.0, 1.0], [1.0, 1.0]],
        family="binary",
        analysis_times=[2.0, 4.0],
        max_enrollment=2,
        max_duration=None,
        final_followup=5.0,
        minimum_enrollment=1,
        initial_equal_randomization=2,
        binary_outcomes=[[1, 0], [0, 1]],
        binary_delays=[[0.0, 0.0], [0.0, 2.0]],
        loser_cutoff=0.4,
        policy=_policy(),
    )

    by_time = {look.time: look for look in result.looks}
    assert by_time[2.0].suspended[1]
    assert not by_time[4.0].suspended[1]


def test_early_winner_waits_for_minimum_enrollment_and_uses_strict_cutoff():
    result = arand_calendar_replay(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[],
        max_enrollment=3,
        max_duration=None,
        final_followup=1.0,
        minimum_enrollment=2,
        binary_outcomes=[[1], [1], [1]],
        binary_window=0.1,
        early_winner_cutoff=1.0,
        final_winner_cutoff=1.0,
        policy=_policy(),
    )

    assert not result.stopped_early
    assert result.assignments.size == 3
    assert result.early_winner is None
    assert result.final_winner is None


def test_future_scheduled_look_is_ignored_after_terminal_followup():
    result = arand_calendar_replay(
        [0.0],
        [0.0],
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[2.0],
        max_enrollment=1,
        max_duration=None,
        final_followup=1.0,
        minimum_enrollment=1,
        binary_outcomes=[[1]],
        binary_window=0.5,
        policy=_policy(),
    )

    assert result.final_analysis_time == 1.0
    assert all(look.time <= 1.0 for look in result.looks)


def test_minimum_gate_defers_futility_and_all_futile_has_no_winner():
    result = arand_calendar_replay(
        [0.0, 1.0],
        [0.0, 0.0],
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[0.5],
        max_enrollment=2,
        max_duration=None,
        final_followup=0.2,
        minimum_enrollment=2,
        binary_outcomes=[[0], [0]],
        binary_window=0.1,
        futility_threshold=0.99,
        futility_cutoff=0.5,
        final_winner_cutoff=0.0,
        policy=_policy(),
    )

    by_time = {look.time: look for look in result.looks}
    assert not by_time[0.5].futile[0]
    assert result.futile[0]
    assert result.reason == "all_arms_futile"
    assert result.final_winner is None


def test_final_winner_can_be_reversibly_suspended():
    result = arand_calendar_replay(
        [0.0],
        [0.0],
        prior=[[1.0, 1.0], [1.0, 1.0]],
        family="binary",
        analysis_times=[],
        max_enrollment=1,
        max_duration=None,
        final_followup=1.0,
        minimum_enrollment=1,
        binary_outcomes=[[1, 0]],
        binary_window=0.5,
        loser_cutoff=0.8,
        final_winner_cutoff=0.4,
        policy=_policy(),
    )

    assert np.all(result.suspended)
    assert result.final_winner == 0


def test_survival_event_is_visible_at_absolute_deadline_equality():
    result = arand_calendar_replay(
        [0.3],
        [0.0],
        prior=[[2.0, 1.0]],
        family="exponential",
        analysis_times=[],
        max_enrollment=1,
        max_duration=None,
        final_followup=0.6,
        minimum_enrollment=1,
        event_times=[[0.6]],
        policy=_policy(),
    )

    assert result.final_analysis_time == 0.3 + 0.6
    assert result.looks[-1].observed == 1


def test_scheduled_look_after_last_arrival_runs_before_duration_terminal():
    result = arand_calendar_replay(
        [0.0, 1.0],
        [0.0, 0.0],
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[2.0],
        max_enrollment=None,
        max_duration=10.0,
        final_followup=0.5,
        minimum_enrollment=2,
        binary_outcomes=[[1], [1]],
        binary_window=0.1,
        early_winner_cutoff=0.9,
        policy=_policy(),
    )

    assert result.reason == "early_winner"
    assert result.stop_time == 2.0
    assert result.early_winner == 0
