import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import binom

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_multinomial.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_counts_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    kwargs = dict(
        n=case["n"], p=case["p"], legacy=True, source="c" if case["language"] == "c" else "fortran"
    )
    values, states = [], []
    for _ in case["values"]:
        values.append(bank.multinomial(**kwargs)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, case["values"])
    assert_array_equal(states, case["states"])
    bank.reinitialize()
    assert_array_equal(bank.multinomial(len(values), **kwargs), values)
    assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize(
    "n,p", [(1, [0.2, 0.3, 0.5]), (10, [0.7, 0.2, 0.1]), (1000, [0.1, 0.2, 0.3, 0.4])]
)
def test_joint_distribution(n, p, legacy, source):
    p = np.array(p)
    values = RandlibGenerator().multinomial(20000, n=n, p=p, legacy=legacy, source=source)
    assert_array_equal(values.sum(axis=1), n)
    covariance = n * (np.diag(p) - np.outer(p, p))
    assert np.all(abs(values.mean(axis=0) - n * p) < 6 * np.sqrt(np.diag(covariance) / len(values)))
    np.testing.assert_allclose(
        np.cov(values, rowvar=False, bias=True), covariance, rtol=0.06, atol=0.003 * n
    )
    for j in range(len(p)):
        assert (
            abs(np.mean(values[:, j] <= int(n * p[j])) - binom.cdf(int(n * p[j]), n, p[j])) < 0.015
        )
    assert values.dtype == np.int64
    assert not values.flags.writeable


@pytest.mark.parametrize(
    "n,p",
    [
        (0, [0.2, 0.3, 0.5]),
        (100, [0.2, 0.3, 0.5]),
        (100, [0, 1, 0]),
        (10, [1]),
        (1000000, [0.3, 0.7]),
    ],
)
def test_default_scalar_batch_and_consumption(n, p):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.multinomial(100, n=n, p=p)
    reference.integers(100 * (len(p) - 1))
    assert bank.get_seeds() == reference.get_seeds()
    assert_array_equal(values.sum(axis=1), n)
    bank.reinitialize()
    assert_array_equal(values, [bank.multinomial(n=n, p=p)[0] for _ in values])


def test_two_categories_match_binomial():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.multinomial(100, n=100, p=[0.3, 0.7])
    assert_array_equal(values[:, 0], reference.binomial(100, n=100, p=0.3))
    assert_array_equal(values[:, 1], 100 - values[:, 0])
    assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_and_budget_rollback(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    assert bank.multinomial(0, p=[0.2, 0.3, 0.5], legacy=legacy).shape == (0, 3)
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.multinomial(20, n=100, p=[0.2, 0.3, 0.5], legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before


def test_legacy_zero_trials_consumes_first_binomial_only():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.multinomial(20, n=0, p=[0.2, 0.3, 0.5], legacy=True), 0)
    reference.integers(20)
    assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": True},
        {"n": 1.5},
        {"n": 2**53},
        {"n": 2147483647, "legacy": True},
        {"p": []},
        {"p": 0.5},
        {"p": [[0.5, 0.5]]},
        {"p": [0.2, 0.3]},
        {"p": [-0.1, 1.1]},
        {"p": [np.nan, 1]},
        {"p": [np.inf, 0]},
        {"p": [1], "legacy": True},
        {"p": [1, 0], "legacy": True},
        {"p": [0.999999, 0.000001], "legacy": True},
        {"p": [1e-50, 1], "legacy": True},
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
        bank.multinomial(**kwargs)
    assert bank.get_seeds() == before


def test_resource_limit_covers_output_elements():
    bank = RandlibGenerator(max_draws=5)
    before = bank.get_seeds()
    with pytest.raises(ValueError, match="categories"):
        bank.multinomial(3, p=[0.5, 0.5])
    with pytest.raises(ValueError, match="categories"):
        bank.multinomial(0, p=[1 / 6] * 6)
    assert bank.get_seeds() == before
