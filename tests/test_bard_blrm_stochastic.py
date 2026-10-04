from itertools import product

import numpy as np
import pytest

from mdanderson_stats.bard import bard_select_obd
from mdanderson_stats.bard_bf_boin_trial import BARDStageTwoDesign
from mdanderson_stats.bard_blrm import BARDLogisticPrior
from mdanderson_stats.bard_blrm_generation import BARDBLRMSimulationDesign
from mdanderson_stats.bard_blrm_stochastic import run_bard_blrm_stochastic_trial
from mdanderson_stats.bard_response import bard_response_model


def _response_model():
    profiles = list(product((1, 2), repeat=1))
    return bard_response_model([0.30, 0.48, 0.62], profiles, [0.5, 0.5], [[1.0, 1.5]])


def _stage_two(*, dose_pair=(1, 2)):
    return BARDStageTwoDesign(
        total_target=12,
        eligible_profiles=[True, True],
        prior=[0.25, 0.25, 0.25, 0.25],
        safety_weights=[1.0, 1.0],
        toxicity_limit=0.30,
        efficacy_limit=0.20,
        safety_cutoff=0.95,
        efficacy_cutoff=0.95,
        utilities=[0.0, 30.0, 50.0, 100.0],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=3.0,
        dose_pair=dose_pair,
        balanced_factors=(0,),
    )


def _design(*, prior=(-2.0, -0.6931471805599453), max_arrivals=40):
    return BARDBLRMSimulationDesign(
        doses=[1.0, 2.0, 3.0],
        reference_dose=1.0,
        prior=BARDLogisticPrior(mean=prior, standard_deviation=[0.0, 0.0]),
        target_interval=[0.20, 0.30],
        eta=0.45,
        cohort_size=3,
        max_escalation_patients=9,
        backfill_evaluable_cap=3,
        draws=8,
        warmup=0,
        chains=2,
        max_arrivals=max_arrivals,
        accrual_rate=3.0,
        dlt_window=1.0,
    )


def test_generated_two_stage_ledger_replays_and_uses_same_counts_for_both_obd_methods():
    model = _response_model()
    design = _design()
    stage_two = _stage_two()
    first = run_bard_blrm_stochastic_trial(design, [0.10, 0.22, 0.30], model, stage_two, rng=165)
    replay = run_bard_blrm_stochastic_trial(design, [0.10, 0.22, 0.30], model, stage_two, rng=165)

    assert first.seed == 165
    assert first.status in {"completed", "carryover_exceeds_target"}
    assert first.dose_pair == (1, 2)
    assert first.outcome_counts.sum() == first.stage_one_carryover + first.stage_two_enrollment
    assert first.stage_two_enrollment == first.required_new_enrollment - first.shortfall
    assert first.factor_history.shape == (first.total_sample_size, 1)
    assert first.dose_history.size == first.dlt_history.size == first.response_history.size
    assert first.profile_index_history.size == first.arrival_history.size == first.arm_history.size
    assert np.array_equal(first.outcome_counts, replay.outcome_counts)
    assert np.array_equal(first.dose_history, replay.dose_history)
    assert np.array_equal(first.profile_index_history, replay.profile_index_history)
    assert first.duration == replay.duration
    assert all(
        patient.dlt_assessment <= patient.response_assessment
        for patient in first.stage_two_patients
    )

    ni = bard_select_obd(
        first.outcome_counts,
        prior=stage_two.prior,
        safety_weights=stage_two.safety_weights,
        toxicity_limit=stage_two.toxicity_limit,
        efficacy_limit=stage_two.efficacy_limit,
        safety_cutoff=stage_two.safety_cutoff,
        efficacy_cutoff=stage_two.efficacy_cutoff,
        method="noninferiority",
        utilities=stage_two.utilities,
        margin=stage_two.margin,
        tie_arm=stage_two.tie_arm,
    )
    utility = bard_select_obd(
        first.outcome_counts,
        prior=stage_two.prior,
        safety_weights=stage_two.safety_weights,
        toxicity_limit=stage_two.toxicity_limit,
        efficacy_limit=stage_two.efficacy_limit,
        safety_cutoff=stage_two.safety_cutoff,
        efficacy_cutoff=stage_two.efficacy_cutoff,
        method="utility",
        utilities=stage_two.utilities,
        margin=stage_two.margin,
        tie_arm=stage_two.tie_arm,
    )
    assert first.final_noninferiority is not None
    assert first.final_utility is not None
    assert np.array_equal(first.final_noninferiority.posterior_shape, ni.posterior_shape)
    assert np.array_equal(first.final_utility.posterior_shape, utility.posterior_shape)


def test_explicit_pair_does_not_reopen_a_no_mtd_stage_one():
    result = run_bard_blrm_stochastic_trial(
        _design(prior=(1.0, 0.0)),
        [0.5, 0.7, 0.9],
        _response_model(),
        _stage_two(dose_pair=(1, 2)),
        rng=165,
    )

    assert result.status == "stage_one_no_mtd"
    assert result.dose_pair is None
    assert result.stage_two is None
    assert result.final_noninferiority is None
    assert result.final_utility is None
    assert result.stage_two_enrollment == 0


def test_incomplete_stage_one_arrival_schedule_is_not_a_valid_trial():
    with pytest.raises(RuntimeError, match="increase max_arrivals"):
        run_bard_blrm_stochastic_trial(
            _design(max_arrivals=1),
            [0.10, 0.22, 0.30],
            _response_model(),
            _stage_two(),
            rng=165,
        )


def test_stage_two_assessment_delay_must_survive_its_calendar_origin():
    stage_two = BARDStageTwoDesign(
        total_target=12,
        eligible_profiles=[True, True],
        prior=[0.25, 0.25, 0.25, 0.25],
        safety_weights=[1.0, 1.0],
        toxicity_limit=0.30,
        efficacy_limit=0.20,
        safety_cutoff=0.95,
        efficacy_cutoff=0.95,
        utilities=[0.0, 30.0, 50.0, 100.0],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=1e-20,
        dose_pair=(1, 2),
        balanced_factors=(0,),
    )
    with pytest.raises(ArithmeticError, match="assessment delay is not representable"):
        run_bard_blrm_stochastic_trial(
            _design(), [0.10, 0.22, 0.30], _response_model(), stage_two, rng=165
        )
