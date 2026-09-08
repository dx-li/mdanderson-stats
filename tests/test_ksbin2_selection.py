"""Native mid-p convention and inclusive rejection-region selection."""

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import ksbin2_probability_table

CASES = json.loads((Path(__file__).parent / "fixtures/ksbin2_probability.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_midp_adjustment(case):
    r = ksbin2_probability_table(
        *case["trials"],
        *case["probabilities"],
        criteria=case["criteria"],
        alternative=case["alternative"],
    )
    assert_allclose(r.midp_significance, case["midp_significance"], rtol=3e-12, atol=2e-14)
    assert np.all(r.midp_significance >= r.pointwise_midp_significance - 1e-15)


def test_independent_pointwise_half_tied_mass():
    t = ksbin2_probability_table(3, 4, 0.6, 0.2, criteria=(1,))
    previous = -1
    for group, end in enumerate(t.ordering.group_end):
        for ip, p in enumerate(t.null_grid):
            expected = 0.0
            for row, (k1, k2) in enumerate(t.ordering.events[: end + 1]):
                weight = 1 if row <= previous else 0.5
                expected += (
                    weight
                    * comb(3, int(k1))
                    * comb(4, int(k2))
                    * p ** (k1 + k2)
                    * (1 - p) ** (7 - k1 - k2)
                )
            assert_allclose(t.null_midp[ip, group], expected, atol=2e-15)
        previous = end
    # Maximization and the midpoint operation do not commute.
    assert np.max(t.midp_significance - t.pointwise_midp_significance) > 0.1


@pytest.mark.parametrize("method", ["ordinary", "midp"])
@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
def test_largest_eligible_region_and_exact_level_ties(method, alternative):
    p1, p2 = (0.6, 0.2) if alternative == "greater" else (0.2, 0.6)
    t = ksbin2_probability_table(4, 5, p1, p2, alternative=alternative)
    levels = t.significance if method == "ordinary" else t.midp_significance
    for alpha in [0, 0.01, 0.05, 0.1, 0.5, 1, *levels]:
        r = t.select(alpha, method=method)
        eligible = np.flatnonzero(levels <= alpha)
        assert r.group == (int(eligible[-1]) if len(eligible) else -1)
        assert r.reported_significance <= alpha
        if r.group == -1:
            assert r.events.shape == (0, 2)
            assert r.power == 0
            assert r.significance == 0
        else:
            assert r.last_row == t.ordering.group_end[r.group]
            assert_allclose(r.events, t.ordering.events[: r.last_row + 1])
            assert_allclose(r.power, t.power[..., r.group])
            assert r.significance == t.significance[r.group]
        independent = sum(
            comb(4, int(i))
            * comb(5, int(j))
            * p1**i
            * (1 - p1) ** (4 - i)
            * p2**j
            * (1 - p2) ** (5 - j)
            for i, j in r.events
        )
        assert_allclose(r.power, independent, atol=2e-14)


def test_midp_selection_keeps_actual_size_and_full_power():
    t = ksbin2_probability_table(1, 1, 0.7, 0.2, criteria=(1,))
    r = t.select(0.125, method="midp")
    assert r.group == 0
    assert r.reported_significance == 0.125
    assert r.significance == 0.25
    assert_allclose(r.power, 0.7 * 0.8)
    assert t.select(0.125).group == -1
    assert t.select_group(-1).last_row == -1
    assert t.select_group(len(t.significance) - 1).events.shape == (4, 2)


def test_broadcast_power_and_explicit_group():
    t = ksbin2_probability_table(4, 4, [[0.5], [0.7]], [0.1, 0.2, 0.3])
    for group in [-1, 0, len(t.significance) - 1]:
        r = t.select_group(group)
        assert r.power.shape == (2, 3)
        assert_allclose(r.power, 0 if group == -1 else t.power[..., group])


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, [0.05]])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        ksbin2_probability_table(3, 4, 0.6, 0.2).select(alpha)


@pytest.mark.parametrize("group", [-2, 999, True, 1.5, [1]])
def test_invalid_group(group):
    with pytest.raises(ValueError):
        ksbin2_probability_table(3, 4, 0.6, 0.2).select_group(group)


def test_invalid_method():
    t = ksbin2_probability_table(3, 4, 0.6, 0.2)
    with pytest.raises(ValueError):
        t.select(0.05, method="pointwise")
    with pytest.raises(ValueError):
        t.select_group(0, method="randomized")
