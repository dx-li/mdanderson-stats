import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import poisson

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_poisson.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_counts_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    kwargs = dict(legacy=True, source="c" if case["language"] == "c" else "fortran")
    values, states = [], []
    for mu in case["parameters"]:
        values.append(bank.poisson(mu=mu, **kwargs)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, case["values"])
    assert_array_equal(states, case["states"])
    if case["mu"] >= 0:
        bank.reinitialize()
        assert_array_equal(bank.poisson(len(values), mu=case["mu"], **kwargs), values)
        assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize("mu", [0.1, 2, 9.9, 10, 100, 10000])
def test_distribution(mu, legacy, source):
    values = RandlibGenerator().poisson(20000, mu=mu, legacy=legacy, source=source)
    assert abs(values.mean() - mu) < 6 * np.sqrt(mu / len(values))
    assert abs(values.var() - mu) < 0.07 * mu
    assert abs(np.mean(values <= int(mu)) - poisson.cdf(int(mu), mu)) < 0.015
    assert values.dtype == np.int64
    assert not values.flags.writeable


@pytest.mark.parametrize("mu", [0, 1e-8, 0.1, 9.9, 10, 10000, 1e9])
def test_default_probability_bracket_and_scalar_batch(mu):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.poisson(100, mu=mu)
    u = reference.uniform(100)
    assert np.all(poisson.cdf(values - 1, mu) <= u + 1e-12)
    assert np.all(poisson.cdf(values, mu) >= u - 1e-12)
    assert bank.get_seeds() == reference.get_seeds()
    bank.reinitialize()
    assert_array_equal(values, [bank.poisson(mu=mu)[0] for _ in values])


@pytest.mark.parametrize("legacy", [False, True])
def test_zero_empty_and_budget_rollback(legacy):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    before = bank.get_seeds()
    assert bank.poisson(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    assert_array_equal(bank.poisson(100, mu=0, legacy=legacy), 0)
    reference.integers(100)
    assert bank.get_seeds() == reference.get_seeds()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.poisson(20, mu=10, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": -1},
        {"mu": np.nan},
        {"mu": np.inf},
        {"mu": [1]},
        {"mu": 2**53},
        {"mu": 1e-50, "legacy": True},
        {"mu": 2147483647, "legacy": True},
        {"source": "c"},
        {"source": "bad", "legacy": True},
        {"legacy": 1},
        {"size": True},
        {"size": -1},
        {"max_attempts": 0},
    ],
)
def test_invalid_parameters_do_not_advance(kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.poisson(**kwargs)
    assert bank.get_seeds() == before


def test_tiny_mean_legacy_rounding_is_explicit():
    seed = (2082061899, 1481316021)
    modern, legacy = RandlibGenerator(seed), RandlibGenerator(seed)
    assert modern.poisson(mu=1e-8)[0] == 1
    assert legacy.poisson(mu=1e-8, legacy=True)[0] == 0
    assert modern.get_seeds() == legacy.get_seeds()


def test_legacy_integer_overflow_rolls_back_entire_batch():
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="integer overflow"):
        bank.poisson(20, mu=2147483520, legacy=True)
    assert bank.get_seeds() == before


def test_default_quantile_outside_exact_integer_range_rolls_back():
    bank = RandlibGenerator((2082061899, 1481316021))
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="invalid Poisson quantiles"):
        bank.poisson(mu=2**53 - 1)
    assert bank.get_seeds() == before
