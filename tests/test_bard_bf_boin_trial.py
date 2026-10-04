import numpy as np

from mdanderson_stats.bard_bf_boin_trial import (
    BARDStageTwoDesign,
    run_bard_bf_boin_trial,
)
from mdanderson_stats.bard_response import bard_response_model
from mdanderson_stats.bf_boin import BFBOINDesign


def _stage_two(*, target=14, dose_pair=(1, 2), eligible=(True, True, True, False)):
    return BARDStageTwoDesign(
        total_target=target,
        eligible_profiles=eligible,
        prior=[0.25, 0.25, 0.25, 0.25],
        safety_weights=[1, 1],
        toxicity_limit=0.3,
        efficacy_limit=0.2,
        safety_cutoff=0.99,
        efficacy_cutoff=0.99,
        utilities=[0, 30, 50, 100],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=2,
        dose_pair=dose_pair,
        balanced_factors=(0, 1),
    )


def _run(*, seed=912, stage_two=None, toxicity=(0.08, 0.18, 0.30)):
    profiles = [(1, 1), (1, 2), (2, 1), (2, 2)]
    model = bard_response_model([0.20, 0.32, 0.45], profiles, [0.25] * 4, [[1, 1.5], [1, 0.8]])
    return run_bard_bf_boin_trial(
        BFBOINDesign(target=0.25, n_cap=6, elimination_probability=0.99),
        toxicity,
        model,
        _stage_two() if stage_two is None else stage_two,
        cohorts=3,
        cohort_size=3,
        rng=seed,
    )


def test_two_stage_trial_replays_and_preserves_counted_patient_ledger():
    first = _run()
    replay = _run()
    assert first.status == "completed"
    assert first.dose_pair == (1, 2)
    assert first.stage_two_enrollment == first.required_new_enrollment
    assert first.stage_one_counts.sum() == first.stage_one_carryover
    assert first.stage_two_counts.sum() == first.stage_two_enrollment
    assert first.outcome_counts.sum() == first.stage_one_carryover + first.stage_two_enrollment
    assert first.total_sample_size == first.stage_one_sample_size + first.stage_two_enrollment
    assert first.selected_dose_utility is None or first.selected_dose_utility in first.dose_pair
    assert np.array_equal(first.factor_history, replay.factor_history)
    assert np.array_equal(first.outcome_counts, replay.outcome_counts)
    assert [p.allocation_seed for p in first.stage_two_patients] == [
        p.allocation_seed for p in replay.stage_two_patients
    ]
    assert all(p.factors.flags.writeable is False for p in first.stage_two_patients)
    assert not first.outcome_counts.flags.writeable
    assert first.stage_two_start_time == first.stage_one.trial_duration[0]
    assert first.duration >= first.stage_two_start_time
    assert all(p.profile_index != 3 for p in first.stage_two_patients)
    assert all(
        p.response_probability
        == first.response_probability_history[first.stage_one_sample_size + i]
        for i, p in enumerate(first.stage_two_patients)
    )


def test_mandatory_carryover_over_target_is_retained_without_new_enrollment():
    result = _run(stage_two=_stage_two(target=1))
    assert result.status == "carryover_exceeds_target"
    assert result.stage_one_carryover >= 1
    assert result.stage_two_enrollment == 0
    assert result.stage_one_counts.sum() == result.outcome_counts.sum()
    assert len(result.stage_two_patients) == 0


def test_explicit_pair_cannot_reopen_stage_one_no_mtd_stop():
    result = _run(toxicity=(1.0, 1.0, 1.0))
    assert result.status == "stage_one_no_mtd"
    assert result.dose_pair is None
    assert result.stage_two_enrollment == 0
    assert result.final_utility is None
    assert result.final_noninferiority is None
