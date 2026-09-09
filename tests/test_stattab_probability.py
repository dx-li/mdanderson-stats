"""Discrete terms use exact sums and independent high-precision log probabilities."""

import math
from decimal import Decimal, localcontext
from functools import cache

import numpy as np
import pytest
from test_cdflib_gamma_support_reference import positive_log_gamma
from test_stattab_reference import rows

from mdanderson_stats import (
    stattab_binomial_term as binomial,
)
from mdanderson_stats import (
    stattab_negative_binomial_term as negative_binomial,
)
from mdanderson_stats import (
    stattab_poisson_term as poisson,
)


def power(base, exponent):
    return Decimal(1) if exponent == 0 else Decimal(base) ** exponent


@cache
def independent(family, count, parameter, chance=None, complement=None):
    with localcontext() as ctx:
        ctx.prec = 800
        k = Decimal(math.floor(count))
        if family == "poisson":
            mean = Decimal.from_float(parameter)
            if mean == 0:
                return float(k == 0)
            logp = -mean - positive_log_gamma(k + 1)
            if k:
                logp += k * mean.ln()
        else:
            n = Decimal(math.floor(parameter))
            p = (
                Decimal.from_float(chance)
                if chance is not None
                else 1 - Decimal.from_float(complement)
            )
            q = Decimal.from_float(complement) if complement is not None else 1 - p
            if p <= q:
                q = 1 - p
            else:
                p = 1 - q
            if family == "binomial":
                if k == 0 and n == 0:
                    return 1.0
                if p == 0:
                    return float(k == 0)
                if q == 0:
                    return float(k == n)
                logp = (
                    positive_log_gamma(n + 1)
                    - positive_log_gamma(k + 1)
                    - positive_log_gamma(n - k + 1)
                )
                logp += k * p.ln() + (n - k) * q.ln()
            else:
                if n == 0:
                    return float(k == 0)
                if p == 0:
                    return 0.0
                if q == 0:
                    return float(k == 0)
                logp = positive_log_gamma(k + n) - positive_log_gamma(n) - positive_log_gamma(k + 1)
                logp += n * p.ln() + k * q.ln()
        return float(logp.exp()) if logp > -10000 else 0.0


BINOMIAL = [
    (s, n, p, None)
    for n in (0, 1, 2, 5, 20, 100)
    for s in range(n + 1)
    for p in (0, 0.125, 0.5, 0.875, 1)
]
NB = [
    (f, s, p, None)
    for f in (0, 1, 2, 10, 100)
    for s in (0, 1, 3, 20)
    for p in (0, 0.125, 0.5, 0.875, 1)
]
POISSON = [(n, m) for n in (0, 0.5, 1, 2.5, 7, 8, 20, 100) for m in (0, 0.125, 1, 3, 20, 100)]


@pytest.mark.parametrize("s,n,p,q", BINOMIAL)
def test_binomial_matches_exact_integer_probabilities(s, n, p, q):
    actual = float(binomial(s, n, p, q))
    # Small integer counts: no special-function oracle is needed.
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float(Decimal(math.comb(n, s)) * power(p, s) * power(1 - p, n - s))
    assert actual == pytest.approx(expected, rel=3e-13, abs=0)


@pytest.mark.parametrize("f,s,p,q", NB)
def test_negative_binomial_matches_exact_integer_probabilities(f, s, p, q):
    actual = float(negative_binomial(f, s, p, q))
    if s == 0:
        expected = float(f == 0)
    else:
        with localcontext() as ctx:
            ctx.prec = 100
            expected = float(Decimal(math.comb(f + s - 1, f)) * power(p, s) * power(1 - p, f))
    assert actual == pytest.approx(expected, rel=3e-13, abs=0)


@pytest.mark.parametrize("n,mean", POISSON)
def test_poisson_matches_independent_factorial_probability(n, mean):
    with localcontext() as ctx:
        ctx.prec = 100
        k = math.floor(n)
        expected = float((-Decimal(mean)).exp() * power(mean, k) / Decimal(math.factorial(k)))
    assert float(poisson(n, mean)) == pytest.approx(expected, rel=3e-13, abs=0)


EXTREME = [
    ("binomial", 1, 2, 5e-324, None),
    ("binomial", 1, 3, None, 1e-162),
    ("binomial", 0, 1e308, 1e-308, None),
    ("binomial", 1e308, 1e308, None, 1e-308),
    ("binomial", 5e99, 1e100, 0.5, None),
    ("binomial", 5e307, 1e308, 0.5, None),
    ("binomial", 1, 1e100, 1e-100, None),
    ("binomial", 100, 1e10, 1e-8, None),
    ("neg_binomial", 0, 1e308, None, 1e-308),
    ("neg_binomial", 1, 1, 5e-324, None),
    ("neg_binomial", 2, 1, None, 1e-162),
    ("neg_binomial", 1e100, 1e100, 0.5, None),
    ("neg_binomial", 1e308, 1e308, 0.5, None),
    ("neg_binomial", 1e308, 1, 1e-308, None),
    ("poisson", 0, 744, None, None),
    ("poisson", 1, 5e-324, None, None),
    ("poisson", 2, 1e-160, None, None),
    ("poisson", 100, 1, None, None),
    ("poisson", 1e100, 1e100, None, None),
    ("poisson", 1e308, 1e308, None, None),
    ("poisson", 1e308, 5e-324, None, None),
    ("poisson", 1, 1e308, None, None),
]


@pytest.mark.parametrize("family,k,n,p,q", EXTREME)
def test_subnormal_and_huge_count_probabilities(family, k, n, p, q):
    fn = {"binomial": binomial, "neg_binomial": negative_binomial, "poisson": poisson}[family]
    with np.errstate(all="raise"):
        value = float(fn(k, n) if family == "poisson" else fn(k, n, p, q))
    expected = independent(family, k, n, p, q)
    assert abs(value - expected) <= max(abs(expected) * 3e-13, math.ulp(expected))
    assert 0 <= value <= 1


def test_consistent_truncation_repairs_native_fractional_zero_count_terms():
    assert float(binomial(0.5, 4.75, 0.5)) == 0.0625
    assert float(negative_binomial(0.5, 3.75, 0.5)) == pytest.approx(0.125, rel=3e-13)
    assert float(poisson(0.5, 3)) == pytest.approx(math.exp(-3))
    assert float(binomial(0.5, 4.75, 0.5)) != rows("binomial_boundary_0")[0][-1]
    assert float(poisson(0.5, 3)) != rows("poisson_boundary_0")[0][-1]
    assert float(binomial(2.5, 4.75, 0.5)) == pytest.approx(
        rows("binomial_boundary_1")[0][-1], abs=5.1e-7
    )
    assert float(poisson(2.5, 3)) == pytest.approx(rows("poisson_boundary_1")[0][-1], abs=5.1e-7)
    assert float(negative_binomial(2, 3, 0.5)) == pytest.approx(0.1875, rel=3e-13)
    assert rows("neg_binomial_which_1")[0][-1] == 0


@pytest.mark.parametrize("fn,args", [(binomial, (1, 3)), (negative_binomial, (1, 3))])
def test_complements_preserve_tiny_chance_and_reject_inconsistent_pairs(fn, args):
    np.testing.assert_array_equal(fn(*args, p=0.25), fn(*args, q=0.75))
    with pytest.raises(ValueError):
        fn(*args)
    with pytest.raises(ValueError):
        fn(*args, p=0.25, q=0.8)
    with pytest.raises(ValueError):
        fn(*args, p=-0.1)
    with pytest.raises(ValueError):
        fn(*args, q=math.nan)


@pytest.mark.parametrize(
    "fn,args", [(binomial, (2, 3, 0.5)), (negative_binomial, (2, 3, 0.5)), (poisson, (2, 3))]
)
def test_broadcast_ownership_empty_inputs_and_validation(fn, args):
    input_values = np.array([[0], [1], [2]])
    actual = fn(input_values, np.array([3, 4]), *args[2:])
    assert actual.shape == (3, 2)
    for i in range(3):
        for j, n in enumerate([3, 4]):
            assert actual[i, j] == fn(i, n, *args[2:])
    saved = actual.copy()
    input_values[:] = 1
    np.testing.assert_array_equal(actual, saved)
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    assert fn([], 3, *args[2:]).shape == (0,)
    for invalid in [-1, math.nan, math.inf]:
        with pytest.raises(ValueError):
            fn(invalid, *args[1:])
        with pytest.raises(ValueError):
            fn(args[0], invalid, *args[2:])
    with pytest.raises(ValueError):
        fn([1, 2], [3, 4, 5], *args[2:])


def test_binomial_validates_original_counts_before_truncation():
    with pytest.raises(ValueError):
        binomial(0.9, 0.5, 0.5)


@pytest.mark.parametrize("n,p", [(1, 0.01), (10, 0.25), (100, 0.5), (1000, 0.999)])
def test_binomial_probabilities_sum_to_one(n, p):
    values = binomial(np.arange(n + 1), n, p)
    assert math.fsum(values) == pytest.approx(1, rel=3e-13)


def test_negative_binomial_and_poisson_recurrence_and_normalization():
    k = np.arange(100)
    nb = negative_binomial(k, 3, 0.5)
    pois = poisson(k, 3)
    np.testing.assert_allclose(nb[1:] / nb[:-1], 0.5 * (k[:-1] + 3) / (k[:-1] + 1), rtol=3e-13)
    np.testing.assert_allclose(pois[1:] / pois[:-1], 3 / (k[:-1] + 1), rtol=3e-13)
    assert math.fsum(nb) == pytest.approx(1, rel=3e-13)
    assert math.fsum(pois) == pytest.approx(1, rel=3e-13)


@pytest.mark.parametrize("q", [8e-163, 1e-162, 2e-162, 1e-160, 5e-160])
def test_smallest_probability_bins_normalize_before_rounding(q):
    for fn, family, k, n in [
        (binomial, "binomial", 1, 3),
        (negative_binomial, "neg_binomial", 2, 1),
    ]:
        actual = float(fn(k, n, q=q))
        expected = independent(family, k, n, None, q)
        assert abs(actual - expected) <= math.ulp(expected)
    actual = float(poisson(2, q))
    expected = independent("poisson", 2, q)
    assert abs(actual - expected) <= math.ulp(expected)
