"""Nonuniform conditional timing for binary TOP analysis."""

import numpy as np

from mdanderson_stats.top_binary import TOPBinaryDesign
from mdanderson_stats.top_calendar import run_top_binary_trial
from mdanderson_stats.top_calibration import optimize_top_binary


def test_followup_mixture_matches_hand_calculation_and_uniform_default():
    followup = np.array([0.25, 0.5, 0.75])
    uniform = TOPBinaryDesign(10, 0.2, 0.8, 0, prior=[1, 1], looks=[10])
    explicit_uniform = TOPBinaryDesign(
        10, 0.2, 0.8, 0, prior=[1, 1], looks=[10], timing_probabilities=[1 / 3] * 3
    )
    early = TOPBinaryDesign(
        10, 0.2, 0.8, 0, prior=[1, 1], looks=[10], timing_probabilities=[1, 0, 0]
    )
    assert not uniform.timing_probabilities.flags.writeable
    np.testing.assert_allclose(uniform.timing_probabilities, [1 / 3] * 3)
    np.testing.assert_allclose(
        uniform.evaluate_followup(7, 2, followup, 1).effective_sample_size,
        explicit_uniform.evaluate_followup(7, 2, followup, 1).effective_sample_size,
    )
    # The first-third CDF is clip(3t, 0, 1): weights .75, 1, 1.
    decision = early.evaluate_followup(7, 2, followup, 1)
    assert decision.effective_sample_size == 9.75
    assert decision.posterior_beta == 8.75


def test_calendar_replay_uses_analysis_timing_for_pending_patients():
    early = TOPBinaryDesign(2, 0.5, 0.9, 0, looks=[2], timing_probabilities=[1, 0, 0])
    late = TOPBinaryDesign(2, 0.5, 0.9, 0, looks=[2], timing_probabilities=[0, 0, 1])
    early_trial = run_top_binary_trial(early, [0.2, 0.6], [np.inf, np.inf], 1)
    late_trial = run_top_binary_trial(late, [0.2, 0.6], [np.inf, np.inf], 1)
    assert early_trial.steps[0].effective_sample_size == 1.0
    assert late_trial.steps[0].effective_sample_size == 0.0


def test_calibration_propagates_analysis_timing_without_changing_truth_timing():
    timing = np.array([0.7, 0.2, 0.1])
    result = optimize_top_binary(
        4,
        0.2,
        0.5,
        1.0,
        2.0,
        cutoff_scales=[0.9],
        gammas=[0.0],
        type1_error=0.99,
        looks=[4],
        timing_probabilities=timing,
        trials=100,
        validation_trials=100,
        arrival="fixed",
        rng=19,
    )
    np.testing.assert_allclose(result.design.timing_probabilities, timing)
    assert not result.design.timing_probabilities.flags.writeable


def test_tiny_prior_survives_integer_failure_subtraction():
    design = TOPBinaryDesign(
        10,
        0.5,
        0.8,
        0.0,
        prior=[1e-320, 1e-320],
        looks=[10],
    )
    decision = design.evaluate(10, 10, 0, 0)
    assert decision.posterior_beta == 1e-320
