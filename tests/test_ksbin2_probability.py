"""Native probability tables and independent exhaustive trial outcomes."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import ksbin2_probability_table

CASES = json.loads((Path(__file__).parent / "fixtures/ksbin2_probability.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_probability_routines(case):
    r = ksbin2_probability_table(
        *case["trials"],
        *case["probabilities"],
        criteria=case["criteria"],
        alternative=case["alternative"],
    )
    assert_allclose(r.significance, case["significance"], rtol=3e-12, atol=2e-14)
    assert_allclose(r.power, case["power"], rtol=3e-12, atol=2e-14)


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("pair", [(0.2, 0.6), (0, 1), (0.49, 0.51)])
def test_exhaustive_sequences(alternative, pair):
    p1, p2 = pair[::-1] if alternative == "greater" else pair
    grid = np.array([0, 0.13, 0.5, 0.87, 1])
    r = ksbin2_probability_table(
        2, 3, p1, p2, criteria=(1,), alternative=alternative, null_grid=grid
    )
    expected = np.zeros((6, len(r.ordering.group_end)))
    thresholds = r.ordering.score[r.ordering.group_end]
    for a in itertools.product([0, 1], repeat=2):
        for b in itertools.product([0, 1], repeat=3):
            i, j = sum(a), sum(b)
            difference = i / 2 - j / 3
            score = (
                -abs(difference)
                if alternative == "two-sided"
                else (-difference if alternative == "greater" else difference)
            )
            region = score <= thresholds
            for k, (pa, pb) in enumerate([*zip(grid, grid), (p1, p2)]):
                weight = pa**i * (1 - pa) ** (2 - i) * pb**j * (1 - pb) ** (3 - j)
                expected[k, region] += weight
    assert_allclose(r.null_rejection, expected[:5], atol=2e-15)
    assert_allclose(r.significance, np.max(expected[:5], axis=0), atol=2e-15)
    assert_allclose(r.power, expected[5], atol=2e-15)
    assert_allclose(r.null_rejection.max(axis=0), r.significance)
    for i, p in enumerate(r.maximizing_null_probability):
        assert_allclose(r.null_rejection[np.flatnonzero(grid == p)[0], i], r.significance[i])


def test_broadcast_and_complete_tied_regions():
    r = ksbin2_probability_table(4, 4, [[0.4], [0.6]], [0, 0.1, 0.2], criteria=(1,))
    assert r.power.shape == (2, 3, 9)
    assert np.all(np.diff(r.power, axis=-1) >= 0)
    assert_allclose(r.power[..., -1], 1, atol=2e-15)
    assert_allclose(r.null_rejection[:, -1], 1, atol=2e-15)
    for i, p1 in enumerate([0.4, 0.6]):
        for j, p2 in enumerate([0, 0.1, 0.2]):
            scalar = ksbin2_probability_table(4, 4, p1, p2, criteria=(1,))
            assert_allclose(r.power[i, j], scalar.power)


def test_default_grids_and_two_sided_complement_symmetry():
    two = ksbin2_probability_table(3, 4, 0.2, 0.6, alternative="two-sided")
    full = ksbin2_probability_table(
        3, 4, 0.2, 0.6, alternative="two-sided", null_grid=np.arange(101) * 0.01
    )
    assert_allclose(two.null_grid, np.arange(51) * 0.01)
    assert_allclose(two.significance, full.significance, atol=2e-14)
    one = ksbin2_probability_table(3, 4, 0.6, 0.2)
    assert_allclose(one.null_grid, np.arange(51) * 0.02)


def test_small_power_and_maximum_grid():
    r = ksbin2_probability_table(100, 100, 0.001, 0, criteria=(1,), null_grid=[0.5])
    assert_allclose(r.power[0], 1e-300, rtol=2e-13, atol=0)
    assert r.ordering.events.shape == (10201, 2)
    assert_allclose(r.power[-1], 1, atol=2e-14)


@pytest.mark.parametrize(
    "changes",
    [
        {"probability1": np.nan},
        {"probability2": 1.1},
        {"probability1": -0.1},
        {"probability1": 0.1},
        {"probability2": 0.6},
        {"null_grid": []},
        {"null_grid": [0.5, 0.5]},
        {"null_grid": [0.8, 0.2]},
        {"null_grid": [[0.2, 0.4]]},
        {"null_grid": [-0.1, 0.5]},
        {"null_grid": [0, np.inf]},
        {"null_grid": 0.5},
    ],
)
def test_invalid_inputs(changes):
    args = dict(trials1=3, trials2=4, probability1=0.6, probability2=0.2)
    args.update(changes)
    with pytest.raises(ValueError):
        ksbin2_probability_table(**args)
