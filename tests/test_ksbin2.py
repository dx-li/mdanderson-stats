"""KSBIN2 evidence ordering: native grids and independent identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import ksbin2_ordering, ksbin2_statistic

CASES = json.loads((Path(__file__).parent / "fixtures/ksbin2.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_grid(case):
    n1, n2 = case["trials"]
    k1, k2 = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing="ij")
    score = ksbin2_statistic(
        k1, n1, k2, n2, criteria=case["criteria"], alternative=case["alternative"]
    )
    assert_allclose(score.ravel(), case["score"], rtol=2e-12, atol=1e-13)


def test_statistical_identities_and_endpoints():
    # 1/4 versus 3/4: difference .5, Pearson chi-square 2, unpooled Z sqrt(8/3).
    for criterion, expected in [(1, -0.5), (3, -2), (4, -np.sqrt(8 / 3))]:
        assert_allclose(
            ksbin2_statistic(1, 4, 3, 4, criteria=[criterion], alternative="less"), expected
        )
    expected = -(2 * np.log(0.25) + 6 * np.log(0.75) - 8 * np.log(0.5))
    assert_allclose(ksbin2_statistic(1, 4, 3, 4, criteria=[2], alternative="less"), expected)
    assert ksbin2_statistic(0, 4, 4, 4, criteria=[4], alternative="less") == -999
    for criterion in range(1, 5):
        assert_allclose(
            ksbin2_statistic([0, 1, 4], 4, [0, 2, 8], 8, criteria=[criterion]), 0, atol=0
        )


def test_weighted_combination_and_mirror():
    a = ksbin2_statistic(np.arange(6), 5, 2, 7, criteria=[4, 2, 1])
    b = sum(
        0.001**i * ksbin2_statistic(np.arange(6), 5, 2, 7, criteria=[c])
        for i, c in enumerate([4, 2, 1])
    )
    assert_allclose(a, b)
    assert_allclose(
        a, ksbin2_statistic(2, 7, np.arange(6), 5, criteria=[4, 2, 1], alternative="less")
    )
    assert_allclose(
        -np.abs(a),
        ksbin2_statistic(np.arange(6), 5, 2, 7, criteria=[4, 2, 1], alternative="two-sided"),
    )


def test_complete_ordered_space_and_ties():
    r = ksbin2_ordering(4, 4, criteria=[1])
    assert r.events.shape == (25, 2)
    assert len(set(map(tuple, r.events))) == 25
    assert np.all(np.diff(r.score) >= 0)
    assert r.group_end[-1] == 24
    assert len(r.group_end) == 9
    starts = np.r_[0, r.group_end[:-1] + 1]
    for start, end in zip(starts, r.group_end):
        assert_allclose(r.score[start : end + 1], r.score[start])
    assert_allclose(r.score, ksbin2_statistic(r.events[:, 0], 4, r.events[:, 1], 4, criteria=[1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trials1": 0},
        {"trials2": 101},
        {"events1": 5},
        {"events2": -1},
        {"events1": 0.5},
        {"events1": np.nan},
        {"criteria": []},
        {"criteria": [1, 1]},
        {"criteria": [5]},
        {"criteria": [[1, 2]]},
        {"criteria": [1.5]},
        {"alternative": "up"},
    ],
)
def test_invalid_counts_and_criteria(kwargs):
    args = dict(events1=1, trials1=4, events2=2, trials2=4)
    args.update(kwargs)
    with pytest.raises(ValueError):
        ksbin2_statistic(**args)


@pytest.mark.parametrize("sizes", [(0, 2), (101, 2), (2.5, 3), ([2], 3)])
def test_invalid_ordering_sizes(sizes):
    with pytest.raises(ValueError):
        ksbin2_ordering(*sizes)
