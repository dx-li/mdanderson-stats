import numpy as np
import pytest

from mdanderson_stats.bard_blrm import BARDLogisticPrior
from mdanderson_stats.bard_blrm_trial import run_bard_blrm_trial


def _run(**overrides):
    n = 17
    defaults = dict(
        doses=[1.0, 2.0, 3.0],
        reference_dose=1.0,
        prior=BARDLogisticPrior([-3.0, 0.0], [0.0, 0.0]),
        target_interval=[0.16, 0.6],
        eta=0.3,
        arrival_times=np.arange(n, dtype=float) * 0.5,
        potential_toxicities=np.zeros((n, 3), dtype=bool),
        potential_responses=np.ones((n, 3), dtype=bool),
        dlt_assessment_delays=np.full((n, 3), 2.0),
        response_assessment_delays=np.full((n, 3), 0.25),
        dlt_window=2.0,
        cohort_size=2,
        max_escalation_patients=8,
        backfill_evaluable_cap=3,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(10),
        boundary_policy="stop",
    )
    defaults.update(overrides)
    return run_bard_blrm_trial(**defaults)


def test_explicit_timeline_refits_only_dlt_and_backfills_while_cohort_waits():
    result = _run()
    assert result.accepted_arrival_indices == (0, 1, 5, 6, 7, 8, 9, 10, 11, 15, 16)
    assert tuple(patient.dose for patient in result.patients) == (
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    )
    assert result.backfill_patients == 3
    assert result.stop_reason == "escalation_patient_cap"
    assert result.final_time == 10.0
    assert result.selected_mtd == 2
    assert result.fit_count == 12  # prior fit plus eleven distinct DLT assessment times
    assert np.array_equal(result.final_assigned, [5, 6, 0])
    assert np.array_equal(result.final_evaluated, result.final_assigned)
    assert result.response_observed_patients == len(result.patients)
    assert all(not step.snapshot.ptt.flags.writeable for step in result.steps)


def test_initial_unsafe_boundary_is_explicit_and_does_not_assign():
    result = _run(
        prior=BARDLogisticPrior([-3.0, 0.0], [0.0, 0.0]),
        eta=0.0,
    )
    assert result.stop_reason == "boundary_policy_stop"
    assert result.boundary_event == "unsafe_initial_dose"
    assert result.patients == ()
    assert result.selected_mtd is None

    with pytest.raises(RuntimeError, match="unsafe_initial_dose"):
        _run(eta=0.0, boundary_policy="raise")


def test_all_overdose_at_prior_state_stops_before_enrollment():
    result = _run(prior=BARDLogisticPrior([1.0, 0.0], [0.0, 0.0]))
    assert result.stop_reason == "stop_all_overdose"
    assert result.all_overdose_detected
    assert result.all_overdose_time == result.start_time
    assert result.patients == ()
    assert result.selected_mtd is None


def test_assessment_delays_cannot_be_observed_before_dlt_window():
    delays = np.full((17, 3), 1.0)
    with pytest.raises(ValueError, match="cannot precede"):
        _run(dlt_assessment_delays=delays)


def _titration_run(**overrides):
    n = 7
    defaults = dict(
        doses=[1.0, 2.0, 3.0],
        reference_dose=1.0,
        prior=BARDLogisticPrior([-3.0, 0.0], [0.0, 0.0]),
        target_interval=[0.16, 0.6],
        eta=0.3,
        arrival_times=np.arange(n, dtype=float) * 0.5,
        potential_toxicities=np.zeros((n, 3), dtype=bool),
        potential_responses=np.ones((n, 3), dtype=bool),
        dlt_assessment_delays=np.ones((n, 3)),
        response_assessment_delays=np.zeros((n, 3)),
        dlt_window=1.0,
        cohort_size=2,
        max_escalation_patients=7,
        backfill_evaluable_cap=3,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(22),
        boundary_policy="stop",
        accelerated_titration=True,
        potential_grade2_toxicities=np.zeros((n, 3), dtype=bool),
        grade2_assessment_delays=np.full((n, 3), 0.1),
    )
    defaults.update(overrides)
    return run_bard_blrm_trial(**defaults)


def test_first_dlt_exits_titration_and_tops_up_current_cohort():
    toxicity = np.zeros((7, 3), dtype=bool)
    toxicity[0, 0] = True
    dlt_delay = np.ones((7, 3))
    dlt_delay[0, 0] = 0.2
    result = _titration_run(
        potential_toxicities=toxicity,
        dlt_assessment_delays=dlt_delay,
        max_escalation_patients=2,
    )
    assert [(p.role, p.dose) for p in result.patients] == [
        ("titration", 1),
        ("titration_topup", 1),
    ]
    assert result.titration_exit_reason == "first_dlt"
    assert result.titration_patients == 1
    assert result.titration_grade2_observed == 0
    assert result.stop_reason == "escalation_patient_cap"


def test_second_grade2_exits_and_lower_cap_waits_then_starts_full_next_cohort():
    grade2 = np.zeros((7, 3), dtype=bool)
    grade2[0, 0] = True
    grade2[2, 1] = True
    grade2_delay = np.full((7, 3), 0.1)
    result = _titration_run(
        potential_grade2_toxicities=grade2,
        grade2_assessment_delays=grade2_delay,
        max_escalation_patients=3,
    )
    assert [p.role for p in result.patients] == ["titration", "titration", "titration_topup"]
    assert result.titration_exit_reason == "second_grade2"
    assert result.titration_grade2_observed == 2

    lower_cap = _titration_run(
        titration_cap=2,
        max_escalation_patients=4,
        arrival_times=np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    )
    assert [p.role for p in lower_cap.patients[:3]] == [
        "titration",
        "titration",
        "escalation",
    ]
    assert [p.dose for p in lower_cap.patients[:3]] == [1, 2, 3]
    assert lower_cap.titration_exit_reason == "dose_cap"
    assert any(step.reason == "titration_assessment_pending" for step in lower_cap.steps)


def test_highest_dose_reach_topups_without_waiting_and_disabled_mode_rejects_extra_inputs():
    highest = _titration_run(
        doses=[1.0, 2.0],
        potential_toxicities=np.zeros((7, 2), dtype=bool),
        potential_responses=np.ones((7, 2), dtype=bool),
        dlt_assessment_delays=np.ones((7, 2)),
        response_assessment_delays=np.zeros((7, 2)),
        potential_grade2_toxicities=np.zeros((7, 2), dtype=bool),
        grade2_assessment_delays=np.full((7, 2), 0.1),
        max_escalation_patients=3,
        arrival_times=np.array([0.0, 1.0, 1.1, 2.0, 3.0, 4.0, 5.0]),
    )
    assert [(patient.role, patient.dose) for patient in highest.patients] == [
        ("titration", 1),
        ("titration", 2),
        ("titration_topup", 2),
    ]
    assert [patient.arrival_time for patient in highest.patients] == [0.0, 1.0, 1.1]
    assert highest.titration_exit_reason == "highest_dose"

    with pytest.raises(ValueError, match="require accelerated_titration"):
        _run(potential_grade2_toxicities=np.zeros((17, 3)))


def test_titration_keeps_blrm_safety_boundary_and_waits_for_grade2_with_single_patient_cohorts():
    single = _titration_run(
        cohort_size=1,
        max_escalation_patients=2,
        arrival_times=np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
        dlt_assessment_delays=np.full((7, 3), 1.0),
        grade2_assessment_delays=np.full((7, 3), 2.0),
    )
    assert [p.dose for p in single.patients] == [1, 2]
    assert not any(step.kind == "cohort_decision" for step in single.steps[:6])

    unsafe = _titration_run(
        doses=[1.0, 2.0, 3.0],
        prior=BARDLogisticPrior([-60.0, 4.0], [0.0, 0.0]),
        max_escalation_patients=4,
    )
    assert [p.dose for p in unsafe.patients] == [1]
    assert unsafe.stop_reason == "boundary_policy_stop"
    assert unsafe.boundary_event == "unsafe_current_dose"
