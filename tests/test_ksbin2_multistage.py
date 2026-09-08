"""Exhaustive paired Bernoulli paths for multistage KSBIN2 designs."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KStageTwoSampleBinomial, ksbin2_probability_table


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize(
    "pair", [(0.2, 0.6), (0.5, 0.5), (0, 0), (1, 1), (0, 1), (1, 0), (0.9, 0.1)]
)
def test_exhaustive_paths(alternative, pair):
    d = KStageTwoSampleBinomial(
        [[2, 2], [3, 3], [4, 4]], [0, 0, 1], [2, 2], criteria=(1,), alternative=alternative
    )
    rejection = np.zeros(3)
    quitting = np.zeros(3)
    continuation = np.zeros(3)
    arrivals = [np.zeros((n + 1, n + 1)) for n in [2, 3, 4]]
    for first in itertools.product([0, 1], repeat=4):
        for second in itertools.product([0, 1], repeat=4):
            p1, p2 = pair
            weight = (
                p1 ** sum(first)
                * (1 - p1) ** (4 - sum(first))
                * p2 ** sum(second)
                * (1 - p2) ** (4 - sum(second))
            )
            for stage, n in enumerate([2, 3, 4]):
                point = (sum(first[:n]), sum(second[:n]))
                arrivals[stage][point] += weight
                ordering = d.orderings[stage]
                row = next(i for i, p in enumerate(ordering.events) if tuple(p) == point)
                group = np.searchsorted(ordering.group_end, row)
                if group <= d.reject_group[stage]:
                    rejection[stage] += weight
                    break
                if stage == 2 or group >= d.quit_group[stage]:
                    quitting[stage] += weight
                    break
                continuation[stage] += weight
    r = d.operating_characteristics(*pair)
    assert_allclose(
        [r.rejection, r.quitting, r.continuation], [rejection, quitting, continuation], atol=3e-14
    )
    assert_allclose(
        r.expected_sample_size, (rejection + quitting) @ np.array(d.cumulative_trials), atol=3e-14
    )
    for stage in range(3):
        assert_allclose(d.stage_distribution(stage + 1, *pair), arrivals[stage], atol=3e-14)
    assert_allclose(np.cumsum(r.rejection + r.quitting) + r.continuation, 1, atol=3e-14)


def test_broadcast_and_disabled_boundaries():
    d = KStageTwoSampleBinomial([[2, 3], [4, 3], [4, 5]], [-1, -1, -1], [-1, -1])
    r = d.operating_characteristics([[0], [0.4], [1]], [0, 0.6, 1])
    assert r.rejection.shape == (3, 3, 3)
    assert_allclose(r.rejection, 0)
    assert_allclose(r.expected_sample_size, np.broadcast_to([4, 5], (3, 3, 2)), atol=2e-14)
    assert_allclose(r.continuation[..., :2], 1, atol=2e-14)
    assert_allclose(r.quitting[..., -1], 1, atol=2e-14)


def test_single_stage_matches_table():
    t = ksbin2_probability_table(4, 5, 0.6, 0.2)
    for group in [-1, 0, len(t.significance) - 1]:
        d = KStageTwoSampleBinomial([[4, 5]], [group], [])
        r = d.operating_characteristics(0.6, 0.2)
        assert_allclose(r.rejection_probability, t.select_group(group).power, atol=2e-14)


@pytest.mark.parametrize(
    "sizes,reject,quit",
    [
        ([[2, 2], [3, 3]], [0], [2]),
        ([[2, 2]], [-2], []),
        ([[2, 2]], [999], []),
        ([[2, 2], [3, 3]], [0, 0], [0]),
        ([[2, 2], [3, 3]], [-1, -1], [0]),
        ([[2, 2], [2, 2]], [-1, -1], [-1]),
        ([[2, 2], [1, 3]], [-1, -1], [-1]),
        ([[0, 2]], [-1], []),
        ([[101, 2]], [-1], []),
        ([2, 2], [-1], []),
    ],
)
def test_invalid_design(sizes, reject, quit):
    with pytest.raises(ValueError):
        KStageTwoSampleBinomial(sizes, reject, quit)


@pytest.mark.parametrize("stage", [0, 2, True, 1.5])
def test_invalid_stage(stage):
    with pytest.raises(ValueError):
        KStageTwoSampleBinomial([[2, 2]], [-1], []).stage_distribution(stage, 0.2, 0.6)


def test_invalid_probability():
    d = KStageTwoSampleBinomial([[2, 2]], [-1], [])
    with pytest.raises(ValueError):
        d.operating_characteristics(-0.1, 0.6)


NATIVE = json.loads((Path(__file__).parent / "fixtures/ksbin2_transition.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", NATIVE)
def test_native_transition_coefficients(case):
    first = np.array(case["initial"])
    final = first + case["increment"]
    d = KStageTwoSampleBinomial(
        [first, final], [0, -1], [2], criteria=(1,), alternative=case["alternative"]
    )
    expected = np.zeros(tuple(final + 1))
    for i, j, probability in case["rows"]:
        expected[int(i), int(j)] = probability
    assert_allclose(d.stage_distribution(2, 0.5, 0.5), expected, rtol=2e-13, atol=1e-16)
