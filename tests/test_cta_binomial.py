"""Native BINCOMP and rational conditional-binomial tail probabilities."""

import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import binomial_comparison

CASES = json.loads((Path(__file__).parent / "fixtures/cta_binomial.json").read_text())["cases"]


def exact_tails(events, sizes):
    k, other = map(int, events)
    n = k + other
    p = Fraction(int(sizes[0]), int(sum(sizes)))
    masses = [math.comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(n + 1)]
    return float(sum(masses[: k + 1])), float(sum(masses[k:]))


@pytest.mark.parametrize("case", CASES)
def test_native_event_selection_and_legacy_reports(case):
    fit = binomial_comparison(case["observed"], groups=case["groups"], legacy=True)
    assert_array_equal(fit.events, case["events"])
    assert_allclose(fit.reported_tails, case["reported_tails"], rtol=1e-5)
    assert_allclose(fit.pvalue, case["pvalue"], rtol=1e-5)
    less, greater = exact_tails(fit.events, fit.group_sizes)
    assert_allclose([fit.p_less, fit.p_greater], [less, greater], rtol=1e-13)
    chosen = less if fit.events[0] < fit.events[1] else greater
    assert_allclose(fit.pvalue, 2 * chosen, rtol=1e-13)


@pytest.mark.parametrize("event_index", [0, 1])
@pytest.mark.parametrize("groups", ["rows", "columns"])
def test_small_tables_against_exact_rational_binomial(groups, event_index):
    for cells in itertools.product(range(4), repeat=4):
        table = np.array(cells).reshape(2, 2)
        oriented = table if groups == "rows" else table.T
        sizes = oriented.sum(axis=1)
        if np.any(sizes == 0):
            continue
        fit = binomial_comparison(table, groups=groups, event_index=event_index)
        tails = exact_tails(oriented[:, event_index], sizes)
        assert_allclose([fit.p_less, fit.p_greater], tails, rtol=2e-14)
        assert_allclose(fit.pvalue, min(1, 2 * min(tails)), rtol=2e-14)
        assert_allclose(fit.reported_tails, tails, rtol=2e-14)


def test_source_selects_raw_count_direction_and_can_report_more_than_one():
    table = [[4, 6], [5, 95]]
    source = binomial_comparison(table, legacy=True)
    corrected = binomial_comparison(table)
    assert source.pvalue > 1.9 and corrected.pvalue < 0.02
    assert_allclose(source.reported_tails, [source.p_less] * 2)
    assert_allclose(corrected.pvalue, 2 * corrected.p_greater)
    no_events = binomial_comparison([[0, 10], [0, 20]])
    assert no_events.pvalue == no_events.p_less == no_events.p_greater == 1
    assert binomial_comparison([[0, 10], [0, 20]], legacy=True).pvalue == 2


def test_batch_orientation_swapping_and_readonly_results():
    a = np.array([[1, 9], [5, 15]])
    fit = binomial_comparison(np.stack([a, a[::-1]]).reshape(1, 2, 2, 2))
    transposed = binomial_comparison(a.T, groups="columns")
    assert_allclose(fit.pvalue[0], np.full(2, transposed.pvalue))
    assert_allclose(fit.p_less[0, 0], fit.p_greater[0, 1])
    assert_array_equal(fit.events[0], [[1, 5], [5, 1]])
    assert_array_equal(fit.event_index, [[0, 0]])
    for value in (
        fit.group_sizes,
        fit.events,
        fit.event_index,
        fit.null_probability,
        fit.p_less,
        fit.p_greater,
        fit.pvalue,
        fit.reported_tails,
    ):
        assert not value.flags.writeable
    assert_array_equal(a, [[1, 9], [5, 15]])


def test_automatic_minority_rule_and_explicit_override():
    assert binomial_comparison([[9, 1], [15, 5]]).event_index == 1
    assert binomial_comparison([[10, 10], [10, 10]]).event_index == 1
    for table in ([[1, 9], [9, 1]], [[1, 9], [5, 5]]):
        with pytest.raises(ValueError, match="common minority"):
            binomial_comparison(table)
        fit = binomial_comparison(table, event_index=0)
        assert_array_equal(fit.events, np.asarray(table)[:, 0])


@pytest.mark.parametrize(
    "table",
    [
        [[-1, 1], [2, 3]],
        [[0.5, 1], [2, 3]],
        [[1, np.nan], [2, 3]],
        [[1, np.inf], [2, 3]],
        [[1]],
        [[1, 2, 3], [4, 5, 6]],
        [[0, 0], [1, 2]],
        [[2**52, 0], [0, 2**52]],
    ],
)
def test_invalid_tables(table):
    with pytest.raises(ValueError):
        binomial_comparison(table, event_index=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": "bad"},
        {"event_index": 2},
        {"event_index": True},
        {"event_index": 0.5},
        {"legacy": 1},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        binomial_comparison([[1, 9], [5, 15]], **kwargs)
