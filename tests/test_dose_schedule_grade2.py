"""Source-defined grade-2 toxicity adjudication checks."""

import pytest

from mdanderson_stats.dose_schedule_observation import adjudicate_dose_schedule_grade2


def test_paper_persistence_example_backdates_event_to_onset() -> None:
    before = adjudicate_dose_schedule_grade2(10, as_of=23)
    assert before.qualifies is None and before.adjudication_time is None

    at_deadline = adjudicate_dose_schedule_grade2(10, as_of=24)
    assert at_deadline.qualifies is True
    assert at_deadline.adjudication_time == 24
    assert at_deadline.onset_time == 10


def test_resolution_by_deadline_is_nonqualifying_but_reduction_is_an_or_rule() -> None:
    resolved = adjudicate_dose_schedule_grade2(10, as_of=24, resolution_time=24)
    assert resolved.qualifies is False
    assert resolved.adjudication_time == 24

    reduced = adjudicate_dose_schedule_grade2(
        10, as_of=18, resolution_time=15, dose_reduction_time=18
    )
    assert reduced.qualifies is True
    assert reduced.adjudication_time == 18
    assert reduced.onset_time == 10


def test_future_resolution_and_reduction_do_not_leak_across_snapshots() -> None:
    early = adjudicate_dose_schedule_grade2(
        10, as_of=20, resolution_time=21, dose_reduction_time=23
    )
    assert early.qualifies is None

    resolved = adjudicate_dose_schedule_grade2(
        10, as_of=21, resolution_time=21, dose_reduction_time=23
    )
    assert resolved.qualifies is False
    assert resolved.adjudication_time == 21

    later = adjudicate_dose_schedule_grade2(
        10, as_of=23, resolution_time=21, dose_reduction_time=23
    )
    assert later.qualifies is True
    assert later.adjudication_time == 23


def test_time_units_match_and_invalid_or_unrepresentable_deadlines_fail() -> None:
    days = adjudicate_dose_schedule_grade2(10, as_of=24)
    hours = adjudicate_dose_schedule_grade2(240, as_of=576, day_length=24)
    assert (days.qualifies, days.adjudication_time, days.onset_time) == (
        hours.qualifies,
        hours.adjudication_time / 24,
        hours.onset_time / 24,
    )
    with pytest.raises(ValueError, match="known at as_of"):
        adjudicate_dose_schedule_grade2(5, as_of=4)
    with pytest.raises(ValueError, match="deadline .*floating-point"):
        adjudicate_dose_schedule_grade2(1e308, as_of=1e308)
