import numpy as np

from mdanderson_stats.bop2_dc_randomized_paired import bop2_dc_randomized_paired_design
from mdanderson_stats.bop2_dc_randomized_paired_simulation import (
    simulate_bop2_dc_randomized_paired,
)


def _design():
    return bop2_dc_randomized_paired_design(
        4,
        "multiple_efficacy",
        [0.0, 0.0],
        [0.2, 0.2],
        control_prior=[1, 1, 1, 1],
        treatment_prior=[1, 1, 1, 1],
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        lambda_lrv=[0.8, 0.8],
        lambda_cmv=[0.8, 0.8],
    )


def test_degenerate_joint_truth_has_exact_terminal_outcome():
    result = simulate_bop2_dc_randomized_paired(
        _design(), [0, 0, 0, 1], [0, 0, 0, 1], n_trials=8, rng=41
    )
    assert result.terminal_names == (
        "graduate",
        "stop_no_go",
        "final_go",
        "final_consider",
        "final_no_go",
    )
    assert result.terminal_counts.sum() == 8
    assert result.terminal_counts[1] == 8
    assert result.terminal_sample_sizes.tolist() == [2] * 8
    assert result.expected_sample_size == 2
    assert result.expected_sample_size_mcse == 0


def test_seeded_paths_replay_and_aggregate_summaries_are_consistent():
    design = _design()
    control = np.array([0.1, 0.2, 0.15, 0.55])
    treatment = np.array([0.2, 0.25, 0.1, 0.45])
    result = simulate_bop2_dc_randomized_paired(design, control, treatment, n_trials=12, rng=2026)
    terminal_counts = np.zeros(5, dtype=np.int64)
    terminal_sizes = []
    terminal_indices = {name: i for i, name in enumerate(result.terminal_names)}
    for trial_index, seed in enumerate(result.trial_seeds):
        trial_rng = np.random.default_rng(int(seed))
        outcomes = [
            int(trial_rng.choice(4, p=control if arm == 0 else treatment))
            for arm in design.arm_assignments
        ]
        replay = design.replay(outcomes)
        terminal_counts[terminal_indices[replay.terminal_decision]] += 1
        terminal_sizes.append(replay.states[-1].total_n)
        assert replay.terminal_decision in result.terminal_names
        assert replay.terminal_decision == result.terminal_decisions[trial_index]
    np.testing.assert_array_equal(result.terminal_counts, terminal_counts)
    np.testing.assert_array_equal(result.terminal_sample_sizes, terminal_sizes)
    np.testing.assert_allclose(result.terminal_probabilities, terminal_counts / 12)
    np.testing.assert_allclose(
        result.terminal_mcse,
        np.sqrt(result.terminal_probabilities * (1 - result.terminal_probabilities) / 12),
    )
    assert result.look_action_counts.sum() == result.look_reached_counts.sum()
    assert not result.trial_seeds.flags.writeable


def test_simulation_preflights_work_and_truth_before_random_draws():
    design = _design()
    rng = np.random.default_rng(9)
    before = rng.bit_generator.state
    try:
        simulate_bop2_dc_randomized_paired(design, [0.2, 0.2, 0.2, 0.2], [0.25] * 4, rng=rng)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("invalid truth vector was accepted")
    assert rng.bit_generator.state == before
