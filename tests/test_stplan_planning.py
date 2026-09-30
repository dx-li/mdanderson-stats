"""Behavioral checks for bounded inverse STPLAN planning."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.stplan_continuous import stplan_normal_one_sample_power
from mdanderson_stats.stplan_discrete import (
    stplan_exact_binomial_power,
    stplan_retention_probability,
)
from mdanderson_stats.stplan_planning import STPLAN_METHODS, stplan_solve


def test_scalar_continuous_solve_returns_forward_verified_inputs():
    solved = stplan_solve(
        "stplan_normal_one_sample_power",
        compute="sample_size",
        target_power=0.8,
        bounds=(2, 200),
        parameters={"difference": 0.5, "sd": 1.0, "sides": 2},
    )
    assert_allclose(solved.achieved_power, 0.8, atol=1e-8, rtol=0)
    assert_allclose(
        stplan_normal_one_sample_power(
            solved.inputs["difference"],
            solved.inputs["sd"],
            solved.inputs["sample_size"],
            alpha=solved.inputs["alpha"],
            sides=int(solved.inputs["sides"]),
        ),
        solved.achieved_power,
        rtol=0,
        atol=1e-14,
    )
    assert len(STPLAN_METHODS) == 26


def test_exact_binomial_sample_size_returns_first_attaining_integer():
    fixed = {"null_probability": 0.2, "alternative_probability": 0.4}
    solved = stplan_solve(
        "stplan_exact_binomial_power",
        compute="sample_size",
        target_power=0.6,
        bounds=(1, 100),
        parameters=fixed,
    )
    candidates = np.arange(1, int(solved.value) + 1, dtype=float)
    powers = stplan_exact_binomial_power(
        fixed["null_probability"], fixed["alternative_probability"], candidates
    )
    assert solved.integer_search and solved.integer_goal == "smallest"
    assert solved.value == candidates[np.flatnonzero(powers >= 0.6)[0]]
    assert solved.achieved_power >= 0.6
    assert solved.previous_power is not None and solved.previous_power < 0.6


def test_retention_threshold_uses_largest_attainable_integer_and_k_vectors_freeze():
    solved = stplan_solve(
        "stplan_retention_probability",
        compute="minimum_remaining",
        target_power=0.8,
        bounds=(0, 10),
        parameters={"initial_size": 10, "dropout_rate": 0.1, "duration": 1},
    )
    assert solved.integer_goal == "largest"
    assert solved.achieved_power >= 0.8
    assert stplan_retention_probability(10, solved.value + 1, 0.1, 1) < 0.8

    ksample = stplan_solve(
        "stplan_binomial_k_sample_power",
        compute="sample_sizes",
        target_power=0.8,
        bounds=(20, 10_000),
        parameters={"probabilities": np.array([0.1, 0.2, 0.4])},
        allocation_weights=[1, 2, 1],
    )
    sizes = ksample.inputs["sample_sizes"]
    assert_allclose(np.sum(sizes), ksample.value, rtol=1e-14)
    with pytest.raises(ValueError):
        sizes.setflags(write=True)
