import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import nbinom

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/randlib_negative_binomial.json").read_text()
)


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
        values.append(bank.negative_binomial(**kwargs)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, case["values"])
    assert_array_equal(states, case["states"])
    if case["n"] >= 0:
        bank.reinitialize()
        assert_array_equal(bank.negative_binomial(len(values), **kwargs), values)
        assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize("n,p", [(1, 0.1), (1, 0.9), (4, 0.5), (100, 0.3), (10000, 0.9)])
def test_distribution(n, p, legacy, source):
    values = RandlibGenerator().negative_binomial(20000, n=n, p=p, legacy=legacy, source=source)
    mean = n * (1 - p) / p
    variance = mean / p
    assert abs(values.mean() - mean) < 6 * np.sqrt(variance / len(values))
    assert abs(values.var() - variance) < 0.07 * variance
    assert abs(np.mean(values <= int(mean)) - nbinom.cdf(int(mean), n, p)) < 0.015
    assert values.dtype == np.int64
    assert not values.flags.writeable


@pytest.mark.parametrize("n,p", [(1, 0.1), (1, 0.9999999), (100, 0.3), (1000000, 0.5)])
def test_default_bracket_and_scalar_batch(n, p):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.negative_binomial(100, n=n, p=p)
    u = reference.uniform(100)
    assert np.all(nbinom.cdf(values - 1, n, p) <= u + 1e-12)
    assert np.all(nbinom.cdf(values, n, p) >= u - 1e-12)
    assert bank.get_seeds() == reference.get_seeds()
    bank.reinitialize()
    assert_array_equal(values, [bank.negative_binomial(n=n, p=p)[0] for _ in values])


def test_default_certain_success_still_consumes_uniforms():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.negative_binomial(100, n=10, p=1), 0)
    reference.integers(100)
    assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_and_budget_roll_back(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    assert bank.negative_binomial(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.negative_binomial(20, n=10, p=0.3, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": -1},
        {"n": True},
        {"n": 1.5},
        {"n": 2**53},
        {"n": 2147483648, "legacy": True},
        {"p": 0},
        {"p": -1},
        {"p": 1.1},
        {"p": np.nan},
        {"p": np.inf},
        {"p": [0.5]},
        {"p": 1, "legacy": True},
        {"p": 1e-50, "legacy": True},
        {"p": 0.999999999, "legacy": True},
        {"legacy": 1},
        {"source": "c"},
        {"source": "bad", "legacy": True},
        {"size": -1},
        {"size": True},
        {"max_attempts": 0},
    ],
)
def test_invalid_parameters_do_not_advance(kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.negative_binomial(**kwargs)
    assert bank.get_seeds() == before


@pytest.mark.parametrize("legacy,p", [(True, 1e-30), (False, 1e-30)])
def test_overflow_rolls_back(legacy, p):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError):
        bank.negative_binomial(10, n=100, p=p, legacy=legacy)
    assert bank.get_seeds() == before


@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
def test_one_success_matches_geometric_inverse(p):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.negative_binomial(1000, n=1, p=p)
    expected = np.floor(np.log1p(-reference.uniform(1000)) / np.log1p(-p)).astype(np.int64)
    assert_array_equal(values, expected)
    assert bank.get_seeds() == reference.get_seeds()
