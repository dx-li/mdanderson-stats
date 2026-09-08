"""Native brackets and independently enumerated discrete rejection regions."""

import json
from math import comb, fsum
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_power

CASES = json.loads((Path(__file__).parent / "fixtures/binomial_power.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_bracket(case):
    c = case
    r = binomial_power(c["trials"], c["null"], c["alternative"], c["alpha"])
    b = c["bracket"]
    if b is None or b[0] < 0:
        # XBIN1 uses invalid sentinels, and errors for the empty upper region.
        assert r.significance == 0 and r.power == 0
        if b is not None:
            assert_allclose(
                [r.next_critical, r.next_significance, r.next_power], b[3:], rtol=2e-9, atol=1e-14
            )
    else:
        assert_allclose(
            [
                r.critical,
                r.significance,
                r.power,
                r.next_critical,
                r.next_significance,
                r.next_power,
            ],
            b,
            rtol=2e-9,
            atol=1e-14,
        )


@pytest.mark.parametrize("n", [1, 2, 10, 30])
@pytest.mark.parametrize("p0,pa", [(0.2, 0.06), (0.3, 0.8), (0.8, 0.3)])
def test_all_regions_exhaustively(n, p0, pa):
    pmf0 = [comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(n + 1)]
    pmfa = [comb(n, k) * pa**k * (1 - pa) ** (n - k) for k in range(n + 1)]
    if pa > p0:
        pmf0, pmfa = pmf0[::-1], pmfa[::-1]
    sizes = np.array([0] + [fsum(pmf0[:j]) for j in range(1, n + 2)])
    powers = np.array([0] + [fsum(pmfa[:j]) for j in range(1, n + 2)])
    # Midpoints avoid treating independent summation's rounding as a tie rule.
    levels = (sizes[:-1] + sizes[1:]) / 2
    resolved = (levels > 0) & (levels < 1) & (np.diff(sizes) > 1e-13)
    levels = levels[resolved]
    r = binomial_power(n, p0, pa, levels)
    selected = np.arange(n + 1)[resolved]
    expected = selected - 1 if pa < p0 else n - selected + 1
    assert_allclose(r.critical, expected, atol=0)
    assert_allclose(r.significance, sizes[:-1][resolved], atol=1e-14)
    assert_allclose(r.power, powers[:-1][resolved], atol=1e-14)
    assert_allclose(r.next_significance, sizes[1:][resolved], atol=1e-14)
    assert_allclose(r.next_power, powers[1:][resolved], atol=1e-14)
    assert np.all(r.significance <= levels) and np.all(r.next_significance > levels)


def test_exact_size_tie_and_mixed_direction_broadcast():
    r = binomial_power([[1], [2]], 0.5, [0.25, 0.75], 0.5)
    assert_allclose(r.critical, [[0, 1], [0, 2]])
    assert_allclose(r.significance, [[0.5, 0.5], [0.25, 0.25]])
    assert r.power.shape == (2, 2)


@pytest.mark.parametrize(
    "n,p0,pa,alpha",
    [
        (0, 0.2, 0.4, 0.05),
        (2.5, 0.2, 0.4, 0.05),
        (10, 0, 0.4, 0.05),
        (10, 0.2, 1, 0.05),
        (10, 0.2, 0.2, 0.05),
        (10, 0.2, 0.4, 0),
        (10, 0.2, 0.4, 1),
        (10, np.nan, 0.4, 0.05),
    ],
)
def test_invalid(n, p0, pa, alpha):
    with pytest.raises(ValueError):
        binomial_power(n, p0, pa, alpha)


def test_extreme_alpha_uses_representable_tail_probabilities():
    result = binomial_power(200, [0.02, 0.98], [0.5, 0.5], 3e-307)
    assert_allclose(result.critical, [190, 10])
    assert np.all(result.significance > 0)
    assert np.all(result.significance <= result.alpha)
    assert np.all(result.next_significance > result.alpha)


def test_large_sample_sizes_use_discrete_search_without_count_enumeration():
    result = binomial_power([1e6, 1e10], 1e-5, 2e-5, 0.05)
    assert np.all(result.significance <= 0.05)
    assert np.all(result.next_significance > 0.05)
    assert_allclose(result.next_critical, result.critical - 1)
