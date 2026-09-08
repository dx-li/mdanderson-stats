import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import binom

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_binomial.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_counts_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    kwargs = dict(
        n=case["n"], p=case["p"], legacy=True, source="c" if case["language"] == "c" else "fortran"
    )
    values, states = [], []
    for n, p in case["parameters"]:
        kwargs.update(n=n, p=p)
        values.append(bank.binomial(**kwargs)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, case["values"])
    assert_array_equal(states, case["states"])
    if case["n"] >= 0:
        bank.reinitialize()
        assert_array_equal(bank.binomial(len(values), **kwargs), values)
        assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize("n,p", [(10, 0.3), (100, 0.3), (1000, 0.8), (10000, 0.5)])
def test_distribution(n, p, legacy, source):
    values = RandlibGenerator().binomial(20000, n=n, p=p, legacy=legacy, source=source)
    variance = n * p * (1 - p)
    assert abs(values.mean() - n * p) < 6 * np.sqrt(variance / len(values))
    assert abs(values.var() - variance) < 0.06 * variance
    assert abs(np.mean(values <= int(n * p)) - binom.cdf(int(n * p), n, p)) < 0.015
    assert values.dtype == np.int64
    assert not values.flags.writeable


@pytest.mark.parametrize("n,p", [(10, 0.3), (100, 0.3), (10000, 0.5), (1000000000, 0.01)])
def test_modern_quantile_bracket_and_scalar_batch(n, p):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.binomial(100, n=n, p=p)
    u = reference.uniform(100)
    assert np.all(binom.cdf(values, n, p) >= u - 1e-12)
    assert np.all(binom.cdf(values - 1, n, p) <= u + 1e-12)
    assert bank.get_seeds() == reference.get_seeds()
    bank.reinitialize()
    assert_array_equal(values, [bank.binomial(n=n, p=p)[0] for _ in values])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("n,p,expected", [(0, 0.3, 0), (10, 0, 0), (10, 1, 10)])
def test_degenerate_draw_consumption(n, p, expected, legacy):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.binomial(100, n=n, p=p, legacy=legacy), expected)
    reference.integers(100)
    assert bank.get_seeds() == reference.get_seeds()


def test_c_unit_uniform_restart():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    assert bank.binomial(n=10, p=0, legacy=True, source="c")[0] == 0
    reference = RandlibGenerator(seed)
    reference.integers(2)
    assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_budget_and_trial_limits(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    assert bank.binomial(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.binomial(20, n=100, p=0.3, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": True},
        {"n": 1.2},
        {"n": 2**53},
        {"p": -1},
        {"p": 1.1},
        {"p": float("nan")},
        {"p": float("inf")},
        {"n": 2147483647, "legacy": True},
        {"p": 1e-300, "legacy": True},
        {"p": 1 - 1e-12, "legacy": True},
        {"legacy": 1},
        {"source": "bad"},
        {"source": "c"},
        {"size": True},
        {"max_attempts": 0},
    ],
)
def test_invalid_requests(kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.binomial(**kwargs)
    assert bank.get_seeds() == before


def test_legacy_integer_overflow_rolls_back_and_modern_large_endpoints():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    with pytest.raises(ArithmeticError, match="integer-square overflow"):
        bank.binomial(n=2000000000, p=0.5, legacy=True)
    assert bank.get_seeds() == seed
    assert bank.binomial(n=2**53 - 1, p=1)[0] == 2**53 - 1
    assert bank.binomial(n=2**53 - 1, p=0)[0] == 0


def test_legacy_rounded_complement_and_modern_small_probability():
    # In the original float32 recurrence, 1-p rounds to one here.
    assert_array_equal(RandlibGenerator().binomial(100, n=1000000000, p=1e-8, legacy=True), 0)
    values = RandlibGenerator().binomial(100, n=1000000000, p=1e-8)
    assert abs(values.mean() - 10) < 2
