import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betainc, betaincc

from mdanderson_stats.beta_binomial import BetaBinomialPosterior
from mdanderson_stats.beta_comparison import compare_beta_difference
from mdanderson_stats.bop2_dc_categorical import (
    _combine_actions,
    bop2_dc_categorical_design,
)
from mdanderson_stats.bop2_dc_categorical_calibration import calibrate_bop2_dc_categorical
from mdanderson_stats.bop2_dc_categorical_simulation import simulate_bop2_dc_categorical


def _indicators():
    return [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


def test_arbitrary_category_marginals_and_any_all_composition():
    design = bop2_dc_categorical_design(
        4,
        _indicators(),
        combination="any",
        directions=("greater", "less", "greater"),
        lrv=(0.1, 0.8, 0.1),
        cmv=(0.3, 0.6, 0.5),
        prior=[1, 1, 1, 1, 1],
        looks=[4],
    )
    state = design.monitor([1, 2, 0, 0, 1])
    expected = [
        betaincc(5, 4, 0.1),
        betaincc(5, 4, 0.3),
        betainc(4, 5, 0.8),
        betainc(4, 5, 0.6),
        betaincc(3, 6, 0.1),
        betaincc(3, 6, 0.5),
    ]
    np.testing.assert_allclose(state.posterior_probability.ravel(), expected)
    assert _combine_actions(np.array([2, 1, 0], dtype=np.int8), "any", True) == 2
    assert _combine_actions(np.array([2, 1, 0], dtype=np.int8), "all", True) == 1


def test_randomized_marginal_beta_difference_and_fixed_allocation_replay():
    design = bop2_dc_categorical_design(
        4,
        _indicators(),
        combination="all",
        directions=("greater", "less", "greater"),
        lrv=(0.0, 0.0, -0.2),
        cmv=(0.2, -0.2, 0.2),
        prior=[1, 2, 3, 4, 5],
        control_prior=[2, 1, 3, 2, 1],
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        comparison_tolerance=1e-8,
    )
    counts = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]])
    state = design.monitor(counts)
    indicator = np.asarray(_indicators(), dtype=bool)
    for j in range(3):
        ca = float(np.sum(np.asarray([2, 1, 3, 2, 1])[indicator[j]]))
        cb = float(np.sum(np.asarray([2, 1, 3, 2, 1])[~indicator[j]]))
        ta = float(np.sum(np.asarray([1, 2, 3, 4, 5])[indicator[j]]))
        tb = float(np.sum(np.asarray([1, 2, 3, 4, 5])[~indicator[j]]))
        cs = int(counts[0, indicator[j]].sum())
        ts = int(counts[1, indicator[j]].sum())
        control = BetaBinomialPosterior(ca + cs, cb + counts[0].sum() - cs)
        treatment = BetaBinomialPosterior(ta + ts, tb + counts[1].sum() - ts)
        for criterion, margin in enumerate((design.lrv[j], design.cmv[j])):
            reference = compare_beta_difference(
                control,
                treatment,
                float(margin),
                absolute_tolerance=design.comparison_tolerance,
            )
            expected = (
                reference.below_margin if design.directions[j] == "less" else reference.above_margin
            )
            np.testing.assert_allclose(
                state.posterior_probability[j, criterion], expected, atol=1e-12
            )
    replay_design = bop2_dc_categorical_design(
        4,
        _indicators(),
        combination="all",
        directions=("greater", "less", "greater"),
        lrv=(0.0, 0.0, -0.2),
        cmv=(0.2, -0.2, 0.2),
        prior=[1, 2, 3, 4, 5],
        control_prior=[2, 1, 3, 2, 1],
        arm_assignments=[0, 1, 0, 1],
        looks=[4],
        comparison_tolerance=1e-8,
    )
    replay = replay_design.replay([0, 4, 0, 4])
    assert replay.outcomes_observed.shape == (4,)
    assert replay.arm_assignments_observed.tolist() == [0, 1, 0, 1]
    assert replay.states[-1].total_n == 4


def test_absorbing_single_arm_replay_and_serial_simulation_calibration():
    design = bop2_dc_categorical_design(
        4,
        [[1, 0, 0, 1, 0]],
        combination="any",
        directions="greater",
        lrv=0.4,
        cmv=0.6,
        prior=[1, 1, 1, 1, 1],
        looks=[2, 4],
        lambda_lrv=0.9,
        lambda_cmv=0.9,
        gamma_lrv=0,
        gamma_cmv=0,
    )
    replay = design.replay([1, 1, 0, 0])
    assert replay.terminal_decision == "stop_no_go"
    assert replay.outcomes_observed.tolist() == [1, 1]
    first = simulate_bop2_dc_categorical(design, [0, 0, 0, 0.5, 0.5], n_trials=20, rng=711)
    second = simulate_bop2_dc_categorical(design, [0, 0, 0, 0.5, 0.5], n_trials=20, rng=711)
    np.testing.assert_array_equal(first.trial_seeds, second.trial_seeds)
    np.testing.assert_array_equal(first.terminal_decisions, second.terminal_decisions)
    np.testing.assert_allclose(first.terminal_probabilities.sum(), 1)
    calibrated = calibrate_bop2_dc_categorical(
        [design],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5],
        n_trials=20,
        false_go_limit=1,
        false_no_go_limit=1,
        rng=712,
    )
    assert calibrated.selected_index == 0
    assert calibrated.trial_seeds.shape == (1, 2, 20)
    assert calibrated.terminal_probabilities.shape == (1, 2, 5)


def test_oversized_nonmaterializing_matrix_rows_rejected_before_conversion():
    class OversizedRow:
        def __len__(self):
            return 257

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversized row should be rejected before conversion")

    with pytest.raises(ValueError, match="same bounded category count"):
        bop2_dc_categorical_design(
            4,
            [[0, 1], OversizedRow()],
            combination="any",
            directions="greater",
            lrv=0.1,
            cmv=0.2,
            prior=[1, 1],
        )


def test_independent_r_reference_probabilities_for_single_and_randomized_cases():
    fixture = Path(__file__).parent / "fixtures" / "bop2-dc-categorical-reference.csv"
    with fixture.open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    indicators = ((1, 1, 0, 0, 1), (0, 1, 1, 0, 0), (0, 0, 1, 1, 1))
    designs = {
        "single_any": bop2_dc_categorical_design(
            8,
            indicators,
            combination="any",
            directions=("greater", "less", "greater"),
            lrv=(0.3, 0.5, 0.4),
            cmv=(0.55, 0.35, 0.7),
            prior=(0.4, 0.8, 0.3, 0.6, 0.9),
            looks=(5, 8),
        ),
        "randomized_all": bop2_dc_categorical_design(
            8,
            indicators,
            combination="all",
            directions=("greater", "less", "greater"),
            lrv=(0.05, -0.1, 0.1),
            cmv=(0.15, -0.2, 0.2),
            prior=(0.7, 0.3, 0.5, 0.2, 0.4),
            control_prior=(0.2, 0.6, 0.4, 0.5, 0.3),
            arm_assignments=(0, 0, 1, 1, 0, 0, 1, 1),
            looks=(4, 8),
            comparison_tolerance=1e-8,
        ),
    }
    counts = {
        ("single_any", "final"): [3, 1, 1, 1, 2],
        ("single_any", "interim"): [2, 1, 0, 1, 1],
        ("randomized_all", "final"): [[0, 2, 0, 1, 1], [2, 0, 1, 0, 1]],
        ("randomized_all", "interim"): [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0]],
    }
    for (case, stage), category_counts in counts.items():
        rows = [row for row in reference if row["case"] == case and row["stage"] == stage]
        observed = designs[case].monitor(category_counts).posterior_probability.ravel()
        expected = np.array([float(row["probability"]) for row in rows])
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-8)


def test_multicandidate_calibration_metrics_replay_from_trial_seeds():
    common = dict(
        max_subjects=6,
        indicators=_indicators(),
        combination="any",
        directions=("greater", "less", "greater"),
        lrv=(0.1, 0.8, 0.1),
        cmv=(0.3, 0.6, 0.5),
        prior=(0.2, 0.2, 0.2, 0.2, 0.2),
        looks=(3, 6),
    )
    candidates = (
        bop2_dc_categorical_design(**common, lambda_lrv=0.75, lambda_cmv=0.45),
        bop2_dc_categorical_design(**common, lambda_lrv=0.9, lambda_cmv=0.6),
    )
    futile = np.array([0.1, 0.1, 0.1, 0.35, 0.35])
    effective = np.array([0.35, 0.25, 0.1, 0.15, 0.15])

    for objective in ("cgr", "futile_ess"):
        result = calibrate_bop2_dc_categorical(
            candidates,
            futile,
            effective,
            n_trials=20,
            false_go_limit=1,
            false_no_go_limit=1,
            false_consider_limit=1,
            objective=objective,
            rng=20261005,
        )
        assert result.feasible.tolist() == [True, True]
        for candidate_index, candidate in enumerate(candidates):
            for scenario_index, truth in enumerate((futile, effective)):
                terminal = []
                sizes = []
                for seed in result.trial_seeds[candidate_index, scenario_index]:
                    local = np.random.default_rng(int(seed))
                    tape = local.choice(candidate.n_categories, candidate.max_subjects, p=truth)
                    replay = candidate.replay(tape)
                    terminal.append(replay.terminal_decision)
                    sizes.append(replay.outcomes_observed.size)
                for terminal_index, label in enumerate(result.terminal_names):
                    probability = np.mean(np.asarray(terminal) == label)
                    np.testing.assert_allclose(
                        result.terminal_probabilities[
                            candidate_index, scenario_index, terminal_index
                        ],
                        probability,
                        atol=0,
                    )
                    np.testing.assert_allclose(
                        result.terminal_mcse[candidate_index, scenario_index, terminal_index],
                        np.sqrt(probability * (1 - probability) / len(terminal)),
                        atol=1e-15,
                    )
                if scenario_index == 0:
                    false_go = np.mean(np.isin(terminal, ("graduate", "final_go")))
                    np.testing.assert_allclose(
                        result.false_go_rate[candidate_index], false_go, atol=0
                    )
                    np.testing.assert_allclose(
                        result.futile_expected_sample_size[candidate_index], np.mean(sizes)
                    )
                    np.testing.assert_allclose(
                        result.futile_expected_sample_size_mcse[candidate_index],
                        np.std(sizes, ddof=1) / np.sqrt(len(sizes)),
                    )
                else:
                    false_no_go = np.mean(np.isin(terminal, ("stop_no_go", "final_no_go")))
                    correct_go = np.mean(np.isin(terminal, ("graduate", "final_go")))
                    np.testing.assert_allclose(
                        result.false_no_go_rate[candidate_index], false_no_go, atol=0
                    )
                    np.testing.assert_allclose(
                        result.correct_go_rate[candidate_index], correct_go, atol=0
                    )
        if objective == "cgr":
            expected_index = min(
                range(len(candidates)),
                key=lambda i: (
                    -result.correct_go_rate[i],
                    result.futile_expected_sample_size[i],
                    i,
                ),
            )
        else:
            expected_index = min(
                range(len(candidates)),
                key=lambda i: (
                    result.futile_expected_sample_size[i],
                    -result.correct_go_rate[i],
                    i,
                ),
            )
        assert result.selected_index == expected_index
