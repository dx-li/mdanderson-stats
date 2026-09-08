"""Independent high-precision checks for tiny inclusive binomial tails."""

from decimal import Decimal, localcontext
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_test


def decimal_tail(k, n, probability, greater):
    with localcontext() as context:
        context.prec = 100
        p = Decimal.from_float(probability)
        indices = range(k, n + 1) if greater else range(k + 1)
        return float(sum(Decimal(comb(n, j)) * p**j * (1 - p) ** (n - j) for j in indices))


@pytest.mark.parametrize("k", [170, 180, 185, 188, 189, 190, 191, 192, 193, 194, 195, 196, 200])
@pytest.mark.parametrize("greater", [False, True])
def test_tiny_tail_against_decimal_mass_sum(k, greater):
    n = 200
    events, p = (k, 0.02) if greater else (n - k, 0.98)
    expected = decimal_tail(events, n, p, greater)
    result = binomial_test(events, n, p)
    actual = result.p_greater if greater else result.p_less
    assert_allclose(actual, expected, rtol=2e-11, atol=2 * np.nextafter(0.0, 1.0))


def test_extreme_tail_broadcast_and_compatibility():
    k = np.array([185, 190, 191, 195])[:, None]
    p = np.array([0.02, 0.03])
    result = binomial_test(k, 200, p)
    expected = [
        [decimal_tail(int(events), 200, float(prob), True) for prob in p] for events in k[:, 0]
    ]
    assert_allclose(result.p_greater, expected, rtol=2e-11, atol=2 * np.nextafter(0.0, 1.0))
    assert binomial_test(30, 30, 1e-10).p_greater > 0
    assert binomial_test(30, 30, 1e-10, legacy_cutoffs=True).p_greater == 0


def test_large_trial_counts_keep_the_existing_domain():
    n = np.array([2**31 - 1, 2**31, 2**40, 2**52], dtype=float)
    # A Cephes integer overflow here would give NaNs or silently wrong tails.
    p = 700 / n
    r = binomial_test(0, n, p)
    assert_allclose(r.p_less, np.exp(n * np.log1p(-p)), rtol=2e-12, atol=0)
    assert np.all(r.p_greater == 1)


@pytest.mark.parametrize("n,k,p", [(1000, 800, 0.2), (2000, 1700, 0.4), (500, 450, 0.08)])
def test_larger_extreme_tail_against_decimal(n, k, p):
    expected = decimal_tail(k, n, p, True)
    assert_allclose(
        binomial_test(k, n, p).p_greater, expected, rtol=2e-10, atol=2 * np.nextafter(0.0, 1.0)
    )


def test_opposite_tail_orientation_avoids_cephes_cancellation():
    n = float(2**31 - 1)
    p = 700 / n
    base = np.exp(n * np.log1p(-p))
    expected = base * (1 + n * p / (1 - p))
    assert_allclose(binomial_test(1, n, p).p_less, expected, rtol=2e-12, atol=0)
    p = 1 - p
    expected = np.exp(n * np.log(p)) * (1 + n * (1 - p) / p)
    assert_allclose(binomial_test(n - 1, n, p).p_greater, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("n", [2**31 - 1, 2**31, 2**40, 2**52])
def test_cephes_integer_guard_on_extreme_upper_tail(n):
    probability = 1 / n
    with localcontext() as context:
        context.prec = 100
        p = Decimal.from_float(probability)
        # Mean is approximately one. The Chernoff bound beyond count 300
        # is below binary64's range, so omitted terms cannot affect rounding.
        expected = float(
            sum(Decimal(comb(n, j)) * p**j * (1 - p) ** (n - j) for j in range(160, 301))
        )
    assert_allclose(binomial_test(160, n, probability).p_greater, expected, rtol=2e-10, atol=0)
