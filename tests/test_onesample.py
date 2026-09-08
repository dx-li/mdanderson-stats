import json
from math import comb, exp, fsum
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_interval, binomial_test, poisson_interval, poisson_test

CASES = json.loads((Path(__file__).parent / "fixtures/onesample.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_calculation_module(case):
    c = case
    if c["distribution"] == "binomial":
        n = c["events"] + c["failures"]
        result = binomial_test(c["events"], n, c["null"], legacy_cutoffs=True)
        bounds = binomial_interval(c["events"], n, c["confidence"])
    else:
        result = poisson_test(c["events"], c["null"], exposure=c["exposure"], legacy_cutoffs=True)
        bounds = poisson_interval(c["events"], c["confidence"], c["exposure"])
    assert_allclose(
        [result.p_less, result.p_greater], [c["p_less"], c["p_greater"]], rtol=2e-9, atol=1e-15
    )
    assert_allclose(result.estimate, c["estimate"])
    assert_allclose(bounds, c["bounds"], rtol=2e-8, atol=1e-12)


@pytest.mark.parametrize("p", [0, 0.05, 0.3, 0.8, 1])
def test_binomial_explicit_probability_sums(p):
    n = 20
    mass = [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    r = binomial_test(np.arange(n + 1), n, p)
    assert_allclose(r.p_less, [fsum(mass[: k + 1]) for k in range(n + 1)], rtol=2e-13, atol=1e-15)
    assert_allclose(r.p_greater, [fsum(mass[k:]) for k in range(n + 1)], rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("mean", [0, 0.1, 2, 15])
def test_poisson_explicit_probability_sums(mean):
    mass = [exp(-mean)]
    for j in range(1, 250):
        mass.append(mass[-1] * mean / j)
    r = poisson_test(np.arange(21), mean / 2, exposure=2)
    assert_allclose(r.p_less, [fsum(mass[: k + 1]) for k in range(21)], rtol=2e-13, atol=1e-15)
    assert_allclose(r.p_greater, [fsum(mass[k:]) for k in range(21)], rtol=2e-13, atol=1e-15)


def test_cutoffs_are_explicit_and_small_tails_not_cancelled():
    p = 1e-12
    exact = binomial_test(1, 30, p)
    assert_allclose(exact.p_greater, -np.expm1(30 * np.log1p(-p)), rtol=1e-14)
    assert binomial_test(1, 30, p, legacy_cutoffs=True).p_greater == 0
    assert binomial_test(29, 30, 1 - p).p_less > 0
    assert binomial_test(29, 30, 1 - p, legacy_cutoffs=True).p_less == 0
    assert_allclose(poisson_test(1, p).p_greater, -np.expm1(-p), rtol=1e-14)
    assert poisson_test(1, p, legacy_cutoffs=True).p_greater == 0


def test_broadcast_shapes():
    assert binomial_test([[1], [2]], 10, [0.1, 0.5, 0.9]).p_less.shape == (2, 3)
    assert poisson_test([[1], [2]], [1, 2, 3]).p_greater.shape == (2, 3)


@pytest.mark.parametrize(
    "args",
    [
        (1, 0, 0.5),
        (3, 2, 0.5),
        (-1, 2, 0.5),
        (0.5, 2, 0.5),
        (1, 2, -0.1),
        (1, 2, 1.1),
        (1, 2, np.nan),
    ],
)
def test_invalid_binomial(args):
    with pytest.raises(ValueError):
        binomial_test(*args)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((1, -1), {}),
        ((-0.1, 1), {}),
        ((1, 1), {"exposure": 0}),
        ((1, 1e308), {"exposure": 1e308}),
        ((1, 1), {"legacy_cutoffs": 1}),
    ],
)
def test_invalid_poisson(args, kwargs):
    with pytest.raises(ValueError):
        poisson_test(*args, **kwargs)
