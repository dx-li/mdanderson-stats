import numpy as np
import pytest

from mdanderson_stats.prt_calendar import _interval_counts, run_prt_calendar
from mdanderson_stats.prt_fit import fit_prt_model


def test_interval_ledger_uses_half_open_intervals_and_followup_completion():
    patients = [
        {
            "entry": 0.0,
            "dose": 0,
            "delay": 1.0,
            "event_seen": False,
            "event_interval": -1,
            "completed": 0,
        },
        {
            "entry": 0.0,
            "dose": 1,
            "delay": 2.0,
            "event_seen": False,
            "event_interval": -1,
            "completed": 0,
        },
    ]
    survived, events, pending, pending_rows = _interval_counts(
        patients, 1.0, np.array([1.0, 2.0]), 2
    )
    np.testing.assert_array_equal(survived, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(events, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(pending, [[0, 0], [0, 1]])
    assert pending_rows == [(1, 1)]


def test_calendar_replays_cohorts_and_waits_for_final_followup():
    result = run_prt_calendar(
        [10.0, 10.1, 10.2, 10.3],
        [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
        interval_endpoints=[1.0, 2.0],
        starting_dose=0,
        cohort_size=2,
        max_patients=4,
        waiting_policy="queue",
        rng=np.random.default_rng(9101),
        target=0.99,
        prior_mean=-2.0,
        prior_variance=0.3,
        draws=32,
        warmup=8,
        chains=2,
    )
    assert result.status in {"completed", "no_acceptable_dose"}
    assert result.enrolled_count == 4
    assert result.duration == pytest.approx(2.3)
    assert result.likelihood_evaluations > 0
    assert result.fit_count >= 1
    assert [a.reason for a in result.analyses[:2]] == ["cohort_complete", "cohort_complete"]
    assert result.analyses[-1].reason == "final"
    assert np.all(result.patient_completed_intervals == 2)
    assert not result.patient_event_observed.any()
    assert all(not a.survived.flags.writeable for a in result.analyses)
    assert not result.patient_dose.flags.writeable


def test_calendar_rejects_unresolvable_times_and_work_before_randomness():
    args = dict(
        arrival_times=[0.0, 0.1],
        potential_toxicity_delays=[[np.inf, np.inf], [np.inf, np.inf]],
        interval_endpoints=[1.0],
        starting_dose=0,
        cohort_size=2,
        max_patients=2,
        waiting_policy="queue",
        draws=8,
        warmup=0,
        chains=2,
    )
    rng = np.random.default_rng(9102)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="work"):
        run_prt_calendar(**args, rng=rng, max_work=10)
    assert rng.bit_generator.state == before

    with pytest.raises(ArithmeticError, match="calendar-time resolution"):
        run_prt_calendar(
            **{**args, "arrival_times": [1e16, 1e16 + 2]},
            rng=np.random.default_rng(9103),
        )


def test_suspended_arrivals_queue_and_resume_as_a_new_cohort():
    result = run_prt_calendar(
        [0.0, 0.1, 0.2, 0.3],
        [[1.0, 1.0]] * 4,
        interval_endpoints=[2.0],
        starting_dose=0,
        cohort_size=2,
        max_patients=4,
        waiting_policy="queue",
        rng=np.random.default_rng(123),
        target=0.3,
        negligible_cutoff=0.1,
        excessive_cutoff=0.6,
        epsilon=0.01,
        prior_mean=-1.2,
        prior_variance=0.2,
        draws=32,
        warmup=8,
        chains=2,
    )
    assert [analysis.action for analysis in result.analyses[:4]] == [
        "suspend",
        "suspend",
        "suspend",
        "stay",
    ]
    np.testing.assert_allclose(result.patient_enrollment_time, [0.0, 0.1, 1.0, 1.0])
    assert result.queued_not_enrolled_count == 0
    assert result.enrollment_end_reason == "maximum_enrollment"
    assert result.stopping_time is None


def test_fit_likelihood_cap_rejects_invalid_values_before_randomness():
    rng = np.random.default_rng(9104)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_likelihood_evaluations"):
        fit_prt_model(
            [[1, 0]],
            [[0, 1]],
            draws=8,
            warmup=0,
            chains=2,
            rng=rng,
            max_likelihood_evaluations=np.array([1, 2]),
        )
    assert rng.bit_generator.state == before


def test_safety_stop_survives_roundoff_in_predictive_category_mass():
    result = run_prt_calendar(
        [0.0, 0.1, 0.2, 0.3],
        [[np.inf, np.inf]] * 4,
        interval_endpoints=[2.0],
        starting_dose=0,
        cohort_size=2,
        max_patients=2,
        waiting_policy="queue",
        rng=np.random.default_rng(9101),
        target=0.001,
        prior_mean=0.0,
        prior_variance=0.01,
        draws=32,
        warmup=8,
        chains=2,
    )
    assert result.status == "early_stop"
    assert result.stopping_time == pytest.approx(0.1)
    assert result.enrollment_end_reason == "early_stop"
    assert result.enrolled_count == 2


def test_tape_exhaustion_time_is_not_overwritten_by_followup_updates():
    result = run_prt_calendar(
        [10.0],
        [[np.inf, np.inf]],
        interval_endpoints=[2.0, 3.0],
        starting_dose=0,
        cohort_size=1,
        max_patients=4,
        waiting_policy="queue",
        rng=np.random.default_rng(9105),
        target=0.99,
        prior_mean=-1.3,
        prior_variance=0.02,
        draws=32,
        warmup=8,
        chains=2,
    )
    assert result.enrollment_end_time == 0.0
    assert result.duration == 3.0
