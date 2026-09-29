from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_binary_optimization import (
    optimize_bop2_dc_randomized_binary,
)


def _calibration(**kwargs):
    defaults = dict(
        max_subjects=4,
        theta_lrv=0.0,
        theta_cmv=0.2,
        futile_truth=(0.2, 0.2),
        effective_truth=(0.2, 0.7),
        control_prior=(1.0, 1.0),
        treatment_prior=(1.0, 1.0),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        lambda_lrv_grid=[0.3, 0.7],
        lambda_cmv_grid=[0.3, 0.7],
        gamma_lrv_grid=[0.5],
        gamma_cmv_grid=[0.5],
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        false_consider_limit=1.0,
        graduate_at_interim=True,
    )
    defaults.update(kwargs)
    return optimize_bop2_dc_randomized_binary(**defaults)


def _enumerated_oc(design, control_rate, treatment_rate):
    decisions = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
    look_index = {int(value): i for i, value in enumerate(design.looks)}
    decision_mass = np.zeros((design.looks.size, len(decisions)))
    sample_mass = np.zeros(design.looks.size)
    for tape in product((0, 1), repeat=design.max_subjects):
        probability = 1.0
        for arm, response in zip(design.arm_assignments, tape, strict=True):
            rate = control_rate if arm == 0 else treatment_rate
            probability *= rate if response else 1 - rate
        replay = design.replay(tape)
        terminal_n = int(replay.states[-1].total_n)
        terminal_look = look_index[terminal_n]
        decision_mass[terminal_look, decisions.index(replay.terminal_decision)] += probability
        sample_mass[terminal_look] += probability
    return decision_mass, sample_mass, float(sample_mass @ design.looks)


def test_randomized_binary_grid_matches_exhaustive_fixed_tape_paths() -> None:
    result = _calibration()
    assert result.candidates.parameters.shape == (4, 4)
    assert result.candidates.decision_probability.shape == (2, 4, 2, 5)
    for candidate_index, row in enumerate(result.candidates.parameters):
        design = replace(
            result.design,
            lambda_lrv=float(row[0]),
            lambda_cmv=float(row[1]),
            gamma_lrv=float(row[2]),
            gamma_cmv=float(row[3]),
        )
        for scenario, truth in enumerate((result.futile_truth, result.effective_truth)):
            expected_decisions, expected_sizes, expected_n = _enumerated_oc(
                design, float(truth[0]), float(truth[1])
            )
            np.testing.assert_allclose(
                result.candidates.decision_probability[scenario, candidate_index],
                expected_decisions,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result.candidates.sample_size_probability[scenario, candidate_index],
                expected_sizes,
                rtol=0,
                atol=1e-12,
            )
            assert result.candidates.expected_sample_size[
                scenario, candidate_index
            ] == pytest.approx(expected_n, abs=1e-12)

    futile_decisions = result.candidates.decision_probability[0]
    effective_decisions = result.candidates.decision_probability[1]
    np.testing.assert_allclose(
        result.candidates.false_go_rate,
        futile_decisions[:, :, 1].sum(axis=1) + futile_decisions[:, -1, 2],
    )
    np.testing.assert_allclose(
        result.candidates.false_no_go_rate,
        effective_decisions[:, :, 0].sum(axis=1) + effective_decisions[:, -1, 4],
    )
    np.testing.assert_allclose(
        result.candidates.correct_go_rate,
        effective_decisions[:, :, 1].sum(axis=1) + effective_decisions[:, -1, 2],
    )
    assert np.all(result.candidates.feasible)
    expected_selection = min(
        range(result.candidates.parameters.shape[0]),
        key=lambda index: (
            -result.candidates.correct_go_rate[index],
            result.candidates.expected_sample_size[0, index],
            index,
        ),
    )
    assert result.selected_index == expected_selection
    assert not result.candidates.parameters.flags.writeable
    assert not result.candidates.decision_probability.flags.writeable


def test_randomized_binary_calibration_rejects_invalid_truth_and_work_before_tables() -> None:
    with pytest.raises(ValueError, match="futile treatment-control difference"):
        _calibration(futile_truth=(0.1, 0.9))
    with pytest.raises(ValueError, match="max_work"):
        _calibration(max_work=1)


def test_binary_grid_ties_retain_input_order_with_roundoff_differences() -> None:
    result = _calibration(
        max_subjects=4,
        theta_lrv=0.0,
        theta_cmv=0.2,
        futile_truth=(0.2, 0.2),
        effective_truth=(0.1, 0.7),
        control_prior=(1.0, 2.0),
        treatment_prior=(2.0, 1.0),
        arm_assignments=(0, 1, 0, 1),
        looks=(2, 4),
        lambda_lrv_grid=(0.6, 0.8),
        lambda_cmv_grid=(0.2, 0.4),
        gamma_lrv_grid=(0.0, 0.5),
        gamma_cmv_grid=(0.0, 0.5),
        false_go_limit=0.87,
        false_no_go_limit=0.05,
        false_consider_limit=0.2,
        graduate_at_interim=True,
        objective="cgr",
    )
    assert result.selected_index == 0
    ess = _calibration(
        max_subjects=4,
        theta_lrv=0.0,
        theta_cmv=0.2,
        futile_truth=(0.2, 0.2),
        effective_truth=(0.1, 0.7),
        control_prior=(1.0, 2.0),
        treatment_prior=(2.0, 1.0),
        arm_assignments=(0, 1, 0, 1),
        looks=(2, 4),
        lambda_lrv_grid=(0.6, 0.8),
        lambda_cmv_grid=(0.2, 0.4),
        gamma_lrv_grid=(0.0, 0.5),
        gamma_cmv_grid=(0.0, 0.5),
        false_go_limit=0.87,
        false_no_go_limit=0.05,
        false_consider_limit=0.2,
        graduate_at_interim=True,
        objective="ess_futile",
    )
    assert ess.selected_index == 4
