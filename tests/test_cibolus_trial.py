"""Focused complete-outcome CiBolus conduct checks."""

from copy import deepcopy

import numpy as np
import pytest

from mdanderson_stats.cibolus import CiBolusPrior
from mdanderson_stats.cibolus_trial import simulate_cibolus_trial

_TRUTH = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.03, 0.9, 0.12, 0.25, 0.2])
_CONCENTRATIONS = [0.2, 0.4, 0.8]
_BOLUS = [0.1, 0.6]
_ENDPOINTS = [0.25, 0.5, 0.75, 1.0]
_UTILITY = [[100, 0]] * 5 + [[0, 0]]


def _run(*, parameters: np.ndarray = _TRUTH, uniforms: list[float]) -> object:
    return simulate_cibolus_trial(
        parameters,
        CiBolusPrior(parameters, np.zeros(11)),
        _CONCENTRATIONS,
        _BOLUS,
        _ENDPOINTS,
        utility=_UTILITY,
        n_patients=4,
        cohort_size=2,
        toxicity_limit=0.8,
        toxicity_cutoff=0.9,
        efficacy_limit=0.01,
        efficacy_cutoff=0.9,
        starting=(0, 0),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(910),
        outcome_uniforms=uniforms,
    )


def test_complete_joint_cells_drive_cohorts_and_unrestricted_final_selection() -> None:
    result = _run(uniforms=[0, 0.2, 0.6, 0.95])

    assert [patient.regimen for patient in result.patients] == [(0, 0), (0, 0), (1, 1), (1, 1)]
    assert [patient.observation.kind for patient in result.patients] == [
        "bolus",
        "interval",
        "interval",
        "failure",
    ]
    assert [patient.observation.toxicity for patient in result.patients] == [
        False,
        False,
        False,
        True,
    ]
    assert result.final_pair == (2, 1)
    assert result.final_decision is not None
    assert result.final_decision.action == "select"
    assert result.treated.sum() == 4
    assert result.likelihood_evaluations > 0
    assert result.work_units >= result.likelihood_evaluations


def test_unacceptable_first_cohort_stops_without_later_final_selection() -> None:
    parameters = _TRUTH.copy()
    parameters[6] = 3.0
    result = _run(parameters=parameters, uniforms=[0, 0.2, 0.6, 0.95])

    assert result.early_stopped
    assert result.stop_reason == "all_regimens_unacceptable"
    assert len(result.patients) == 2
    assert result.final_pair is None
    assert result.final_decision is None
    assert result.steps[-1].decision.action == "stop"


def test_cumulative_budget_rejection_does_not_consume_rng() -> None:
    rng = np.random.default_rng(45)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="below one required cohort fit"):
        simulate_cibolus_trial(
            _TRUTH,
            CiBolusPrior(_TRUTH, np.full(11, 0.1)),
            _CONCENTRATIONS,
            _BOLUS,
            _ENDPOINTS,
            utility=_UTILITY,
            n_patients=4,
            cohort_size=2,
            toxicity_limit=0.8,
            toxicity_cutoff=0.9,
            efficacy_limit=0.01,
            efficacy_cutoff=0.9,
            draws=8,
            warmup=0,
            chains=2,
            rng=rng,
            max_total_evaluations=1,
        )
    assert rng.bit_generator.state == before
