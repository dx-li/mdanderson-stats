"""Focused aggregate CiBolus operating-characteristic checks."""

from copy import deepcopy

import numpy as np
import pytest

from mdanderson_stats.cibolus import CiBolusPrior
from mdanderson_stats.cibolus_simulation import (
    _STOP_REASONS,
    simulate_cibolus_operating_characteristics,
)
from mdanderson_stats.cibolus_trial import _summary_max_defined, simulate_cibolus_trial

_TRUTH = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.03, 0.9, 0.12, 0.25, 0.2])
_CONCENTRATIONS = [0.2, 0.4, 0.8]
_BOLUS = [0.1, 0.6]
_ENDPOINTS = [0.25, 0.5, 0.75, 1.0]
_UTILITY = [[100, 0]] * 5 + [[0, 0]]


def _aggregate(rng: np.random.Generator) -> object:
    return simulate_cibolus_operating_characteristics(
        _TRUTH,
        CiBolusPrior(_TRUTH, np.zeros(11)),
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
        rng=rng,
    )


def test_aggregate_summaries_replay_from_returned_trial_seeds() -> None:
    result = _aggregate(np.random.default_rng(712))
    assert result.selection_probability.sum() + result.no_selection_probability == 1
    assert result.stop_reason_count.sum() == result.trials
    assert result.mean_allocation.sum() == 2
    assert result.assigned_patients.sum() == 4
    assert np.array_equal(result.response_category_count.sum(axis=-1), result.assigned_patients)
    assigned_mask = result.assigned_patients > 0
    assert np.allclose(result.response_category_probability.sum(axis=-1)[assigned_mask], 1)
    assert len(result.trial_seeds) == result.trials
    assert result.work_units > result.likelihood_evaluations
    assert tuple(result.stop_reasons) == _STOP_REASONS

    replayed = [
        simulate_cibolus_trial(
            _TRUTH,
            CiBolusPrior(_TRUTH, np.zeros(11)),
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
        )
        for seed in result.trial_seeds
    ]
    replayed_selection = np.zeros(result.grid_shape, dtype=np.int64)
    for trial in replayed:
        if trial.final_pair is not None:
            replayed_selection[trial.final_pair] += 1
    assert np.array_equal(result.selection_count, replayed_selection)
    assert result.mean_enrollment == np.mean([len(trial.patients) for trial in replayed])
    assert np.array_equal(result.assigned_patients, sum(trial.treated for trial in replayed))
    trial_toxicity = np.zeros((len(replayed), *result.grid_shape))
    trial_responses = np.zeros_like(trial_toxicity)
    for trial_index, trial in enumerate(replayed):
        for patient in trial.patients:
            trial_toxicity[(trial_index, *patient.regimen)] += int(patient.observation.toxicity)
            trial_responses[(trial_index, *patient.regimen)] += int(
                patient.observation.kind != "failure"
            )
    assignment = np.stack([trial.treated for trial in replayed])
    denominator = assignment.sum(axis=0)
    p_tox = np.full(result.grid_shape, np.nan)
    p_response = np.full(result.grid_shape, np.nan)
    np.divide(trial_toxicity.sum(axis=0), denominator, out=p_tox, where=denominator > 0)
    np.divide(trial_responses.sum(axis=0), denominator, out=p_response, where=denominator > 0)
    tox_residual = trial_toxicity - p_tox * assignment
    response_residual = trial_responses - p_response * assignment
    expected_tox_mcse = np.sqrt(
        len(replayed)
        / (len(replayed) - 1)
        * np.sum(tox_residual**2, axis=0)
        / np.where(denominator > 0, denominator**2, 1)
    )
    expected_response_mcse = np.sqrt(
        len(replayed)
        / (len(replayed) - 1)
        * np.sum(response_residual**2, axis=0)
        / np.where(denominator > 0, denominator**2, 1)
    )
    expected_tox_mcse[denominator == 0] = np.nan
    expected_response_mcse[denominator == 0] = np.nan
    assert np.allclose(result.toxicity_probability, p_tox, equal_nan=True)
    assert np.allclose(result.toxicity_mcse, expected_tox_mcse, equal_nan=True)
    assert np.allclose(result.response_probability, p_response, equal_nan=True)
    assert np.allclose(result.response_mcse, expected_response_mcse, equal_nan=True)


def test_cumulative_budget_lower_bound_fails_before_rng_advances() -> None:
    rng = np.random.default_rng(45)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="first cohort of every trial"):
        simulate_cibolus_operating_characteristics(
            _TRUTH,
            CiBolusPrior(_TRUTH, np.zeros(11)),
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
            rng=rng,
            max_total_evaluations=35,
        )
    assert rng.bit_generator.state == before


def test_infinite_chain_diagnostic_is_not_discarded() -> None:
    class Summary:
        split_rhat = np.array([1.01, np.inf, np.nan])

    assert _summary_max_defined(Summary(), "split_rhat") == np.inf

    class UndefinedSummary:
        split_rhat = np.array([np.nan, np.nan])

    assert _summary_max_defined(UndefinedSummary(), "split_rhat") is None
