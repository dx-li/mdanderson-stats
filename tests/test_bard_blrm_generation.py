import numpy as np
import pytest

from mdanderson_stats.bard_blrm import BARDLogisticPrior
from mdanderson_stats.bard_blrm_generation import (
    BARDBLRMSimulationDesign,
    _generate_bard_blrm_outcome_tapes,
    simulate_bard_blrm_stage_one,
)
from mdanderson_stats.bard_response import bard_response_model


def _design(*, draws: int = 8) -> BARDBLRMSimulationDesign:
    return BARDBLRMSimulationDesign(
        doses=[1.0, 2.0],
        reference_dose=1.0,
        prior=BARDLogisticPrior([-3.0, 0.0], [0.0, 0.0]),
        target_interval=[0.15, 0.35],
        eta=0.3,
        cohort_size=1,
        max_escalation_patients=2,
        backfill_evaluable_cap=2,
        draws=draws,
        warmup=0,
        chains=2,
        max_arrivals=4,
        arrival_distribution="exponential",
    )


def _model():
    return bard_response_model([0.4, 0.6], [[1], [2]], [0.7, 0.3], [[1.0, 2.0]])


def test_seeded_generation_replays_tapes_and_maps_accepted_patients_by_arrival() -> None:
    model = _model()
    design = _design()
    first = simulate_bard_blrm_stage_one(design, [0.1, 0.25], model, rng=281)
    second = simulate_bard_blrm_stage_one(design, [0.1, 0.25], model, rng=281)
    assert first.seed == second.seed == 281
    assert np.array_equal(first.outcome_tapes.arrival_times, second.outcome_tapes.arrival_times)
    assert np.array_equal(first.outcome_tapes.profile_indices, second.outcome_tapes.profile_indices)
    assert np.array_equal(
        first.outcome_tapes.potential_toxicities, second.outcome_tapes.potential_toxicities
    )
    assert np.array_equal(
        first.outcome_tapes.potential_responses, second.outcome_tapes.potential_responses
    )
    for patient in first.trial.patients:
        row, column = patient.arrival_index, patient.dose - 1
        assert patient.dlt == first.outcome_tapes.potential_toxicities[row, column]
        assert patient.response == first.outcome_tapes.potential_responses[row, column]
        assert np.array_equal(
            first.outcome_tapes.factor_profiles[row],
            model.factor_profiles[first.outcome_tapes.profile_indices[row]],
        )
    assert not first.outcome_tapes.potential_responses.flags.writeable


def test_joint_endpoint_tapes_respect_frechet_conditional_probabilities() -> None:
    model = bard_response_model([0.5], [[1], [2]], [0.6, 0.4], [[1.0, 2.0]])
    p_tox = np.array([0.2])
    p_resp = model.conditional_probabilities[0]
    q = np.minimum(p_resp, p_tox[0]) * 0.5
    tapes = _generate_bard_blrm_outcome_tapes(
        p_tox,
        model,
        arrivals=300,
        accrual_rate=10,
        arrival_distribution="uniform",
        dlt_window=1,
        joint_toxicity_response_probability=[[q[0], q[1]]],
        rng=np.random.default_rng(44),
    )
    for i, profile in enumerate(tapes.profile_indices):
        expected = (
            q[profile] / p_tox[0]
            if tapes.potential_toxicities[i, 0]
            else (p_resp[profile] - q[profile]) / (1 - p_tox[0])
        )
        assert tapes.sampled_response_probabilities[i, 0] == pytest.approx(expected)


def test_invalid_resource_or_truth_inputs_fail_before_generator_advances() -> None:
    model = _model()
    generator = np.random.default_rng(9)
    before = generator.bit_generator.state
    with pytest.raises(ValueError, match="true_toxicity"):
        _generate_bard_blrm_outcome_tapes(
            [0.1],
            model,
            arrivals=20,
            accrual_rate=1,
            arrival_distribution="uniform",
            dlt_window=1,
            rng=generator,
        )
    assert generator.bit_generator.state == before
    with pytest.raises(ValueError, match="max_arrivals"):
        BARDBLRMSimulationDesign(
            doses=[1.0, 2.0],
            reference_dose=1.0,
            prior=BARDLogisticPrior([-3, 0], [0, 0]),
            target_interval=[0.1, 0.3],
            eta=0.3,
            cohort_size=1,
            max_escalation_patients=2,
            backfill_evaluable_cap=2,
            draws=8,
            warmup=0,
            chains=2,
            max_arrivals=2001,
        )


def test_small_nondegenerate_sampler_path_runs() -> None:
    design = BARDBLRMSimulationDesign(
        doses=[1.0],
        reference_dose=1.0,
        prior=BARDLogisticPrior([-3.0, 0.0], [0.05, 0.05]),
        target_interval=[0.05, 0.25],
        eta=0.9,
        cohort_size=1,
        max_escalation_patients=1,
        backfill_evaluable_cap=1,
        draws=8,
        warmup=0,
        chains=2,
        max_arrivals=2,
    )
    result = simulate_bard_blrm_stage_one(
        design, [0.1], bard_response_model([0.4], [[1]], [1], [[1]]), rng=8
    )
    assert result.trial.fit_count >= 1
    assert result.trial.likelihood_evaluations >= 0
