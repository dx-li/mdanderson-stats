import numpy as np

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.boin12_two_stage import (
    boin12_two_stage_next_dose,
    simulate_boin12_two_stage,
)


def test_stage_one_uses_toxicity_only_and_does_not_apply_three_patient_guard() -> None:
    design = BOIN12Design(0.35, 0.25, toxicity_cutoff=0.85)
    low_efficacy = boin12_two_stage_next_dose(design, [1, 0], [1, 0], [0, 0], 1, stage1_threshold=6)
    high_efficacy = boin12_two_stage_next_dose(
        design, [1, 0], [1, 0], [1, 0], 1, stage1_threshold=6
    )
    assert low_efficacy.stage == high_efficacy.stage == 1
    assert low_efficacy.action == high_efficacy.action == "stop_safety"
    assert np.array_equal(low_efficacy.eliminated, high_efficacy.eliminated)
    assert low_efficacy.posterior is high_efficacy.posterior is None


def test_stage_one_safety_cutoff_equality_is_inadmissible() -> None:
    # For zero DLTs in one patient, P(p_T > .25 | data) = .75**2 exactly.
    design = BOIN12Design(0.25, 0.25, toxicity_cutoff=0.5625)
    result = boin12_two_stage_next_dose(design, [1], [0], [0], 1, stage1_threshold=6)
    assert result.action == "stop_safety"
    assert result.eliminated.tolist() == [True]


def test_stage_one_never_jumps_over_an_unsafe_current_dose() -> None:
    design = BOIN12Design(0.35, 0.25)
    result = boin12_two_stage_next_dose(
        design,
        [3, 3, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        2,
        stage1_threshold=6,
    )
    assert result.stage == 1
    assert result.action == "deescalate"
    assert result.next_dose == 1
    assert result.eliminated.tolist() == [False, True, False, False]


def test_lowest_dose_safety_stop_precedes_precision_but_middle_dose_does_not() -> None:
    lowest = BOIN12Design(0.35, 0.25, toxicity_cutoff=0.85, early_stop_patients=1)
    stopped = boin12_two_stage_next_dose(lowest, [1], [1], [0], 1, stage1_threshold=6)
    assert stopped.action == "stop_safety"

    middle = boin12_two_stage_next_dose(
        lowest,
        [3, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        2,
        stage1_threshold=6,
    )
    assert middle.action == "stop_precision"


def test_stage_one_stops_if_boundary_deescalation_would_assign_excluded_neighbor() -> None:
    design = BOIN12Design(0.35, 0.25, toxicity_cutoff=0.999999)
    result = boin12_two_stage_next_dose(
        design,
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 0],
        2,
        stage1_threshold=6,
        eliminated=[True, False, False],
    )
    assert result.action == "stop_no_admissible_neighbor"
    assert result.next_dose is None


def test_stage_switch_is_after_the_triggering_cohort_and_counts_are_retained() -> None:
    design = BOIN12Design(0.35, 0.25)
    no_toxicity_efficacy = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    result = simulate_boin12_two_stage(
        design,
        no_toxicity_efficacy,
        stage1_threshold=6,
        cohorts=4,
        cohort_size=3,
        trials=1,
        rng=12,
    )
    assert result.transition_cohort.tolist() == [3]
    assert result.stage1_cohorts.tolist() == [3]
    assert result.stage2_cohorts.tolist() == [1]
    assert result.patients.tolist() == [[3, 9]]
    assert result.toxicities.tolist() == [[0, 0]]
    assert result.efficacies.tolist() == [[3, 9]]


def test_no_transition_is_reported_and_final_obd_uses_stage_one_joint_data() -> None:
    design = BOIN12Design(0.35, 0.25)
    result = simulate_boin12_two_stage(
        design,
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        stage1_threshold=6,
        cohorts=1,
        cohort_size=3,
        trials=1,
        rng=4,
    )
    assert result.transition_cohort.tolist() == [0]
    assert result.stage1_cohorts.tolist() == [1]
    assert result.stage2_cohorts.tolist() == [0]
    assert result.patients.sum() == 3
    assert result.selected_obd.tolist() == [1]
