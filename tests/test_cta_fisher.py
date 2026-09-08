"""Native FISHXT and independent rational fixed-margin enumeration."""

import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import fisher_exact

CASES = json.loads((Path(__file__).parent / "fixtures/cta_fisher.json").read_text())["cases"]


def exact(table, alternative, legacy=False):
    a, b, c, d = map(int, np.asarray(table).ravel())
    total, row, col = a + b + c + d, a + b, a + c
    masses = {
        x: Fraction(math.comb(row, x) * math.comb(total - row, col - x), math.comb(total, col))
        for x in range(max(0, row + col - total), min(row, col) + 1)
    }
    if alternative == "two-sided":
        selected = [x for x in masses if masses[x] <= masses[a]]
    else:
        lower = alternative == "less" or (alternative == "source" and a * d < b * c)
        selected = (
            sorted((x for x in masses if x <= a), reverse=True)
            if lower
            else [x for x in masses if x >= a]
        )
    if legacy:
        for i, x in enumerate(selected):
            if masses[x] * 100_000 < masses[a]:
                selected = selected[: i + 1]
                break
    return float(sum((masses[x] for x in selected), Fraction(0))), float(masses[a]), len(selected)


@pytest.mark.parametrize("case", CASES)
def test_native_tail_direction_and_cutoff(case):
    result = fisher_exact(case["observed"], legacy=True)
    p, observed, terms = exact(case["observed"], "source", legacy=True)
    assert_allclose(result.pvalue, p, rtol=2e-13)
    assert_allclose(result.observed_probability, observed, rtol=2e-13)
    assert result.terms == terms == case["terms"]
    # FISHXT sums log factorials in single precision. At N=400 the native
    # probability differs from the exact rational sum by 6.28e-4 relative.
    # The tight rational comparison above verifies the double-precision value.
    assert_allclose(result.pvalue, case["pvalue"], rtol=7e-4)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater", "source"])
def test_exhaustive_small_tables_against_integer_combinations(alternative):
    for cells in itertools.product(range(5), repeat=4):
        table = np.array(cells).reshape(2, 2)
        fit = fisher_exact(table, alternative=alternative)
        p, observed, terms = exact(table, alternative)
        assert_allclose(fit.pvalue, p, rtol=2e-14, err_msg=str(table))
        assert_allclose(fit.observed_probability, observed, rtol=2e-14)
        assert fit.terms == terms
        assert not fit.truncated


def test_legacy_cutoff_includes_triggering_term_and_default_does_not_truncate():
    table = [[100, 100], [100, 100]]
    source = fisher_exact(table, legacy=True)
    full = fisher_exact(table, alternative="source")
    assert source.alternative == "source" and source.truncated
    assert source.terms == 25 and full.terms == 101
    assert source.pvalue < full.pvalue
    assert not source.source_lower_tail  # Equal cross-products choose upper.
    assert_allclose(fisher_exact(table).pvalue, 1, rtol=1e-14)
    assert_allclose(full.pvalue, (1 + full.observed_probability) / 2, rtol=1e-14)


def test_batch_transpose_and_exchange_symmetries():
    a = np.array([[1, 9], [11, 3]])
    batch = np.stack([a, a.T, a[::-1]]).reshape(1, 3, 2, 2)
    fit = fisher_exact(batch)
    assert fit.pvalue.shape == (1, 3)
    assert_allclose(fit.pvalue[0], np.full(3, 41 / 14858), rtol=1e-14)
    lower = fisher_exact(a, alternative="less")
    upper = fisher_exact(a[::-1], alternative="greater")
    assert_allclose(lower.pvalue, upper.pvalue, rtol=1e-14)
    assert_array_equal(fit.source_lower_tail, [[True, True, False]])
    for value in (
        fit.pvalue,
        fit.observed_probability,
        fit.terms,
        fit.support_size,
        fit.source_lower_tail,
        fit.truncated,
    ):
        assert not value.flags.writeable
    assert_array_equal(a, [[1, 9], [11, 3]])


def test_original_total_limit_and_underflow_are_handled():
    center = fisher_exact(np.full((2, 2), 12500))
    assert_allclose(center.pvalue, 1, rtol=1e-12)
    assert center.support_size == 25001
    extreme = fisher_exact([[0, 25000], [25000, 0]])
    assert extreme.pvalue == 0 and extreme.observed_probability == 0
    assert extreme.terms == 2
    # A relative cutoff is evaluated in logs even when all tail masses underflow.
    rare = fisher_exact([[100, 24900], [24900, 100]], legacy=True)
    assert rare.pvalue == 0 and rare.truncated and rare.terms < 101


@pytest.mark.parametrize(
    "table",
    [
        [[-1, 1], [2, 3]],
        [[0.5, 1], [2, 3]],
        [[1, np.nan], [2, 3]],
        [[1, np.inf], [2, 3]],
        [[1]],
        [[1, 2, 3], [4, 5, 6]],
        [[25000, 1], [0, 25000]],
        [[2**53, 0], [0, 0]],
    ],
)
def test_invalid_tables(table):
    with pytest.raises(ValueError):
        fisher_exact(table)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alternative": "bad"},
        {"legacy": 1},
        {"legacy": True, "alternative": "two-sided"},
        {"legacy": True, "alternative": "less"},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        fisher_exact([[1, 2], [3, 4]], **kwargs)
