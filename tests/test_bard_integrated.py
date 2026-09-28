import numpy as np
import pytest

from mdanderson_stats.bard import bard_minimization
from mdanderson_stats.bard_blrm_decision import bard_blrm_select_mtd
from mdanderson_stats.bard_blrm_trial import BARDBLRMPatient, BARDBLRMTrial
from mdanderson_stats.bard_integrated import continue_bard_trial


def _stage_one(*, selected_mtd=2, all_overdose=False):
    patients = (
        BARDBLRMPatient(0, 0.0, 1, "escalation", False, True, 2.0, 0.25),
        BARDBLRMPatient(1, 1.0, 2, "escalation", True, True, 1.5, 1.25),
        BARDBLRMPatient(2, 2.0, 3, "escalation", False, False, 4.0, 2.5),
        BARDBLRMPatient(3, 3.0, 2, "backfill", False, False, 5.0, 3.5),
    )
    assigned = np.array([1, 2, 1], dtype=np.int64)
    selection = bard_blrm_select_mtd(
        [0.2, 0.4, 0.6], [0.1, 0.2, 0.4], assigned, eta=0.3, minimum_treated=1
    )
    return BARDBLRMTrial(
        patients=patients,
        steps=(),
        arrival_times=np.array([0.0, 1.0, 2.0, 3.0]),
        final_evaluated=assigned,
        final_toxicities=np.array([0, 1, 0], dtype=np.int64),
        final_assigned=assigned,
        final_responses_observed=assigned,
        final_responses=np.array([1, 1, 0], dtype=np.int64),
        final_ptt=np.array([0.2, 0.4, 0.6]),
        final_pod=np.array([0.1, 0.2, 0.4]),
        final_selection=selection,
        selected_mtd=selected_mtd,
        stop_reason="stop_all_overdose" if all_overdose else "escalation_patient_cap",
        boundary_policy="stop",
        boundary_event=None,
        accepted_arrival_indices=(0, 1, 2, 3),
        declined_arrival_indices=(),
        escalation_patients=3,
        backfill_patients=1,
        response_observed_patients=4,
        dlt_evaluable_patients=4,
        start_time=0.0,
        enrollment_stop_time=3.0,
        final_time=5.0,
        duration=5.0,
        fit_count=4,
        likelihood_evaluations=0,
        work_units=0,
        all_overdose_detected=all_overdose,
        all_overdose_time=0.0 if all_overdose else None,
    )


def _run(stage_one=None, **overrides):
    defaults = dict(
        stage_one=_stage_one() if stage_one is None else stage_one,
        dose_pair=[1, 2],
        stage_one_eligible=[True, True, True, False],
        stage_one_factors=[[1, 1], [1, 2], [2, 2], [2, 1]],
        candidate_factors=[[1, 1], [2, 2], [1, 2]],
        potential_toxicities=[[0, 0], [0, 1], [1, 0]],
        potential_responses=[[1, 1], [0, 1], [1, 0]],
        total_target=5,
        prior=[0.25] * 4,
        safety_weights=[1, 1],
        allocation_probability=1.0,
        tie_probability=0.5,
        toxicity_limit=0.3,
        efficacy_limit=0.2,
        safety_cutoff=0.99,
        efficacy_cutoff=0.99,
        method="utility",
        utilities=[0, 30, 50, 100],
        margin=0.05,
        tie_arm=1,
        rng=np.random.default_rng(42),
    )
    defaults.update(overrides)
    return continue_bard_trial(**defaults)


def test_inclusive_total_target_carries_only_eligible_selected_doses():
    result = _run(total_target=4)
    assert result.status == "completed"
    assert result.dose_pair == (1, 2)
    assert result.stage_one_included_indices == (0, 1)
    assert result.stage_one_excluded_other_dose_indices == (2,)
    assert result.stage_one_excluded_ineligible_indices == (3,)
    assert result.stage_one_carryover == 2
    assert result.required_new_enrollment == 2
    assert result.stage_two_enrollment == 2
    assert result.shortfall == 0
    assert result.outcome_counts.sum() == 4
    assert result.final_selection is not None
    assert not result.outcome_counts.flags.writeable


def test_assignment_replays_and_candidate_exhaustion_is_not_final_selection():
    first = _run(total_target=5, rng=np.random.default_rng(7))
    replay = _run(total_target=5, rng=np.random.default_rng(7))
    assert [p.dose for p in first.patients] == [p.dose for p in replay.patients]
    assert [p.allocation_seed for p in first.patients] == [
        p.allocation_seed for p in replay.patients
    ]
    assert first.status == "completed"
    assert first.candidates_examined == 3

    short = _run(
        total_target=6,
        candidate_factors=[[1, 1]],
        potential_toxicities=[[0, 0]],
        potential_responses=[[0, 1]],
    )
    assert short.status == "candidate_tape_exhausted"
    assert short.required_new_enrollment == 4
    assert short.stage_two_enrollment == 1
    assert short.shortfall == 3
    assert short.final_selection is None


def test_target_cannot_drop_carryover_and_permanent_stage_one_stop_is_not_reopened():
    with pytest.raises(ValueError, match="smaller than mandatory"):
        _run(total_target=1)

    generator = np.random.default_rng(9)
    before = generator.bit_generator.state
    result = _run(stage_one=_stage_one(selected_mtd=None, all_overdose=True), rng=generator)
    assert result.status == "stage_one_no_mtd"
    assert result.stage_one_carryover == 2
    assert result.stage_two_enrollment == 0
    assert result.shortfall == result.required_new_enrollment
    assert result.final_selection is None
    assert generator.bit_generator.state == before


def test_saved_allocation_seed_reconstructs_minimization_decision():
    result = _run(total_target=3)
    patient = result.patients[0]
    history = [1, 2]
    factors = np.array([[1, 1], [1, 2]])
    allocation = bard_minimization(
        history,
        factors,
        [1, 1],
        probability=1,
        tie_probability=0.5,
        seed=patient.allocation_seed,
    )
    assert allocation.assigned_arm == 2
    assert patient.dose == result.dose_pair[allocation.assigned_arm - 1]
    assert np.array_equal(patient.scores, allocation.scores)
    assert np.array_equal(patient.probabilities, allocation.probabilities)
