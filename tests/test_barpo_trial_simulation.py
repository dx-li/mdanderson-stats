"""Focused completed-outcome BARPO trial and OC behavior."""

import numpy as np
import pytest

from mdanderson_stats.barpo_simulation import simulate_barpo
from mdanderson_stats.barpo_trial import run_barpo_trial


def _design():
    return {
        "prior": [[1.0, 1.0], [1.0, 1.0]],
        "max_n": 5,
        "burn_in": 4,
        "er_block_size": 4,
        "cohort_size": 2,
        "looks": (),
    }


def test_aggregate_replay_matches_individual_trial_and_preserves_accounting():
    assignment = np.array([[0.1, 0.7, 0.2, 0.8, 0.3]])
    outcome = np.array([[0.2, 0.9, 0.4, 0.7, 0.6]])
    kwargs = _design()
    trial = run_barpo_trial(
        [0.4, 0.6],
        **kwargs,
        assignment_uniforms=assignment[0],
        outcome_uniforms=outcome[0],
    )
    simulation = simulate_barpo(
        [0.4, 0.6],
        **kwargs,
        trials=1,
        assignment_uniforms=assignment,
        outcome_uniforms=outcome,
        null_arms=np.array([True, False]),
    )
    np.testing.assert_array_equal(simulation.patients_by_arm[0], trial.assigned)
    np.testing.assert_array_equal(simulation.successes_by_arm[0], trial.successes)
    np.testing.assert_array_equal(simulation.failures_by_arm[0], trial.failures)
    np.testing.assert_array_equal(simulation.enrolled, [trial.enrolled])
    assert simulation.early_efficacy_probability[0] == float(trial.early_efficacy[0])
    assert simulation.cumulative_efficacy_probability[0] == float(
        trial.early_efficacy[0] or trial.final_efficacy[0]
    )
    assert simulation.false_cumulative_efficacy_probability == float(
        np.any((trial.early_efficacy | trial.final_efficacy)[:1])
    )
    np.testing.assert_allclose(
        simulation.allocation_probability_by_enrollment[: trial.enrolled],
        trial.assignment_probability,
    )
    assert not simulation.patients_by_arm.flags.writeable


def test_early_declaration_is_included_in_cumulative_probability():
    kwargs = {
        "prior": [[1.0, 1.0]],
        "max_n": 4,
        "burn_in": 0,
        "er_block_size": 4,
        "cohort_size": 1,
        "looks": [2],
        "min_n": 2,
        "theta_eff": 0.5,
        "peff": 0.8,
        "theta_final": 0.5,
        "pfinal": 0.8,
    }
    result = simulate_barpo(
        [0.8],
        **kwargs,
        trials=1,
        assignment_uniforms=np.zeros((1, 4)),
        outcome_uniforms=np.zeros((1, 4)),
        null_arms=np.array([True]),
    )
    assert result.enrolled[0] == 2
    assert result.early_efficacy[0, 0]
    assert not result.final_efficacy[0, 0]
    assert result.cumulative_efficacy_probability[0] == 1.0
    assert result.any_cumulative_efficacy_probability == 1.0
    assert result.false_cumulative_efficacy_probability == 1.0


def test_stream_and_dbcd_preflight_fail_before_consuming_rng():
    assignment_rng = np.random.default_rng(23)
    state_before = assignment_rng.bit_generator.state
    with pytest.raises(ValueError, match="complete balanced ER block"):
        run_barpo_trial(
            [0.2, 0.4],
            prior=[[1.0, 1.0]] * 2,
            max_n=12,
            burn_in=2,
            er_block_size=10,
            cohort_size=2,
            method="dbcd",
            target_probability=[0.3, 0.7],
            assignment_rng=assignment_rng,
            outcome_rng=12,
        )
    assert assignment_rng.bit_generator.state == state_before
    with pytest.raises(ValueError, match="seeds must differ"):
        run_barpo_trial(
            [0.2],
            prior=[[1.0, 1.0]],
            max_n=4,
            burn_in=4,
            er_block_size=4,
            cohort_size=1,
            assignment_rng=17,
            outcome_rng=17,
        )


def test_oversized_replay_tape_is_rejected_before_array_conversion():
    class NoArrayConversion:
        def __array__(self, *args, **kwargs):
            raise AssertionError("oversized tape should be rejected before conversion")

    with pytest.raises(ValueError, match="two-million-cell limit"):
        simulate_barpo(
            [0.2, 0.4],
            prior=[[1.0, 1.0]] * 2,
            max_n=1_200,
            trials=2_000,
            burn_in=4,
            er_block_size=4,
            cohort_size=2,
            assignment_uniforms=NoArrayConversion(),
        )
