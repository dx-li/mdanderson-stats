"""Focused checks for scenario-supplied CiBolus truth cells."""

from copy import deepcopy

import numpy as np
import pytest

from mdanderson_stats.cibolus import CiBolusPrior, cibolus_predict
from mdanderson_stats.cibolus_simulation import simulate_cibolus_operating_characteristics
from mdanderson_stats.cibolus_trial import simulate_cibolus_trial

_TRUTH = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.03, 0.9, 0.12, 0.25, 0.2])
_CONCENTRATIONS = [0.2, 0.4]
_BOLUS = [0.1, 0.6]
_ENDPOINTS = [0.5, 1.0]
_UTILITY = [[100, 0]] * 3 + [[0, 0]]


def _trial(truth: np.ndarray | None, *, joint: np.ndarray | None, seed: int = 910) -> object:
    prior = CiBolusPrior(_TRUTH, np.full(11, 0.1))
    return simulate_cibolus_trial(
        truth,
        prior,
        _CONCENTRATIONS,
        _BOLUS,
        _ENDPOINTS,
        utility=_UTILITY,
        n_patients=2,
        cohort_size=2,
        toxicity_limit=0.8,
        toxicity_cutoff=0.9,
        efficacy_limit=0.01,
        efficacy_cutoff=0.9,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(seed),
        outcome_uniforms=[0.123, 0.789],
        truth_joint_probabilities=joint,
    )


def test_model_derived_joint_truth_reproduces_parameter_truth_path() -> None:
    joint = list(
        cibolus_predict(_TRUTH, _CONCENTRATIONS, _BOLUS, _ENDPOINTS, utility=_UTILITY).joint
    )

    model_path = _trial(_TRUTH, joint=None)
    supplied_path = _trial(None, joint=joint)

    assert [patient.observation for patient in supplied_path.patients] == [
        patient.observation for patient in model_path.patients
    ]
    assert [patient.regimen for patient in supplied_path.patients] == [
        patient.regimen for patient in model_path.patients
    ]
    assert supplied_path.final_pair == model_path.final_pair
    assert supplied_path.work_units > model_path.work_units


def test_deterministic_nonparametric_cells_drive_generated_observations() -> None:
    # Bolus response without toxicity is category 0, toxicity index 0.
    joint = np.zeros((2, 2, 4, 2))
    joint[..., 0, 0] = 1.0
    result = _trial(None, joint=joint)

    assert all(patient.observation.kind == "bolus" for patient in result.patients)
    assert all(not patient.observation.toxicity for patient in result.patients)


def test_invalid_joint_truth_fails_before_rng_advances() -> None:
    rng = np.random.default_rng(51)
    state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="shape"):
        simulate_cibolus_trial(
            None,
            CiBolusPrior(_TRUTH, np.full(11, 0.1)),
            _CONCENTRATIONS,
            _BOLUS,
            _ENDPOINTS,
            utility=_UTILITY,
            n_patients=2,
            cohort_size=2,
            toxicity_limit=0.8,
            toxicity_cutoff=0.9,
            efficacy_limit=0.01,
            efficacy_cutoff=0.9,
            draws=8,
            warmup=0,
            chains=2,
            rng=rng,
            truth_joint_probabilities=np.ones((1,)),
        )
    assert rng.bit_generator.state == state


def test_joint_truth_aggregate_seeds_replay_trial_results() -> None:
    joint = np.zeros((2, 2, 4, 2))
    joint[..., 0, 0] = 1.0
    prior = CiBolusPrior(_TRUTH, np.full(11, 0.1))
    aggregate = simulate_cibolus_operating_characteristics(
        None,
        prior,
        _CONCENTRATIONS,
        _BOLUS,
        _ENDPOINTS,
        utility=_UTILITY,
        n_patients=2,
        cohort_size=2,
        toxicity_limit=0.8,
        toxicity_cutoff=0.9,
        efficacy_limit=0.01,
        efficacy_cutoff=0.9,
        trials=2,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(718),
        truth_joint_probabilities=joint,
    )
    replay = [
        simulate_cibolus_trial(
            None,
            prior,
            _CONCENTRATIONS,
            _BOLUS,
            _ENDPOINTS,
            utility=_UTILITY,
            n_patients=2,
            cohort_size=2,
            toxicity_limit=0.8,
            toxicity_cutoff=0.9,
            efficacy_limit=0.01,
            efficacy_cutoff=0.9,
            draws=8,
            warmup=0,
            chains=2,
            rng=np.random.default_rng(int(seed)),
            truth_joint_probabilities=joint,
        )
        for seed in aggregate.trial_seeds
    ]
    assert np.array_equal(aggregate.assigned_patients, sum(trial.treated for trial in replay))
    replay_selection = np.zeros(aggregate.grid_shape, dtype=np.int64)
    for trial in replay:
        if trial.final_pair is not None:
            replay_selection[trial.final_pair] += 1
    assert np.array_equal(aggregate.selection_count, replay_selection)
    assert aggregate.response_category_count[..., 0].sum() == 4
    assert all(
        patient.observation.kind == "bolus" for trial in replay for patient in trial.patients
    )
