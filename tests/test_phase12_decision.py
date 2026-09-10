from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import (
    fit_phase12_model,
    phase12_source_decision,
    phase12_source_final_selection,
)


def test_source_allocation_suspension_closure_and_patient_gate():
    base = fit_phase12_model(
        np.zeros((6, 4)), draws=8, warmup=0, chains=2, rng=np.random.default_rng(8532)
    )
    fit = replace(
        base,
        reference_superiority=np.array([0.5, 0.5, 0.5, 0.5, 0.001, 0.5]),
        efficacy_probability=np.full(6, 0.01),
        toxicity_probability=np.array([0.95, 0.951, 0, 0, 0, 0]),
    )
    decision = phase12_source_decision(fit, [4] * 6, phase_one_admissible=[1] * 6)
    assert not decision.enough_patients and not decision.terminated
    assert np.array_equal(decision.closed, [0, 1, 0, 0, 0, 0])
    assert np.array_equal(decision.suspended, [0, 0, 0, 0, 1, 0])
    assert_allclose(decision.probability, [0.25, 0, 0.25, 0.25, 0, 0.25])
    recovery = replace(fit, reference_superiority=np.full(6, 0.5), toxicity_probability=np.zeros(6))
    again = phase12_source_decision(
        recovery,
        [4] * 6,
        phase_one_admissible=[1] * 6,
        closed=decision.closed,
        suspended=decision.suspended,
    )
    assert again.closed[1] and not again.suspended[4]
    stopped = phase12_source_decision(fit, [5, 5, 5, 5, 0, 0], phase_one_admissible=[1] * 6)
    assert stopped.terminated and stopped.reason == "futility"
    assert_allclose(stopped.probability, 0)
    # The source maximum for futility includes previously eligible, now closed doses.
    efficacy = np.full(6, 0.01)
    efficacy[1] = 0.2
    keep = phase12_source_decision(
        replace(fit, efficacy_probability=efficacy), [5] * 6, phase_one_admissible=[1] * 6
    )
    assert not keep.terminated


def test_source_ineligible_early_winner_and_final_confidence_threshold():
    base = fit_phase12_model(
        np.zeros((6, 4)), draws=8, warmup=0, chains=2, rng=np.random.default_rng(8533)
    )
    pair = np.zeros((6, 6))
    pair[1, :] = 0.9
    pair[1, 1] = 0
    fit = replace(
        base,
        reference_superiority=np.full(6, 0.5),
        pairwise_superiority=pair,
        efficacy_probability=np.array([0.2, 0.95, 0.2, 0.2, 0.2, 0.2]),
        toxicity_probability=np.array([0, 1, 0, 0, 0, 0]),
        future_probability=np.array([0.85, 0.99, 0.9, 0.91, 0.91, 0.1]),
    )
    result = phase12_source_decision(fit, [5] * 6, phase_one_admissible=[1] * 6)
    assert result.terminated and result.selected == 1 and result.selected_eligible is False
    assert_allclose(result.probability, 0)
    assert phase12_source_final_selection(fit, closed=result.closed, suspended=[0] * 6) == 3
    assert (
        phase12_source_final_selection(fit, closed=result.closed, suspended=[0, 0, 0, 1, 0, 0]) == 4
    )
    assert (
        phase12_source_final_selection(
            replace(fit, future_probability=np.full(6, 0.9)), closed=[0] * 6, suspended=[0] * 6
        )
        is None
    )
    closed = phase12_source_decision(
        replace(fit, efficacy_probability=np.full(6, 0.2)),
        [5] * 6,
        phase_one_admissible=[1] * 6,
        closed=[1] * 6,
    )
    assert closed.arm_closed and not closed.terminated and closed.selected is None
    assert_allclose(closed.probability, 0)
