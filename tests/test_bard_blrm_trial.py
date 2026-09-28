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
