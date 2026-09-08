import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betainc

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_beta.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_beta_values_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    source = "c" if case["language"] == "c" else "fortran"
    values, states = [], []
    for a, b in case["shapes"]:
        values.append(bank.beta(a=a, b=b, legacy=True, source=source)[0])
        states.append(bank.get_seeds())
    assert_allclose(
        values,
        np.array(case["values"], dtype=np.float32).astype(float),
        rtol=6e-6,
        atol=2 * float(np.finfo(np.float32).smallest_subnormal),
    )
    assert_array_equal(states, case["states"])
    if not case["mixed"]:
        bank.reinitialize()
        a, b = case["shapes"][0]
        assert_array_equal(bank.beta(len(values), a=a, b=b, legacy=True, source=source), values)
        assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize("a,b", [(0.1, 0.2), (0.5, 5), (1, 1), (2, 3), (100, 200)])
def test_distribution(a, b, legacy, source):
    n = 20000
    values = RandlibGenerator().beta(n, a=a, b=b, legacy=legacy, source=source)
    mean = a / (a + b)
    variance = a * b / ((a + b) ** 2 * (a + b + 1))
    assert abs(values.mean() - mean) < 6 * np.sqrt(variance / n)
    assert abs(values.var() - variance) < 0.08 * variance
    assert abs(np.mean(values <= mean) - betainc(a, b, mean)) < 0.015
    assert np.all((values >= 0) & (values <= 1))
    assert not values.flags.writeable


@pytest.mark.parametrize("a,b", [(0.1, 0.2), (1, 1), (2, 3), (100, 200)])
def test_inverse_cdf_and_scalar_batch(a, b):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.beta(100, a=a, b=b)
    assert_allclose(betainc(a, b, values), reference.uniform(100), atol=5e-13, rtol=0)
    assert bank.get_seeds() == reference.get_seeds()
    bank.reinitialize()
    assert_array_equal(values, [bank.beta(a=a, b=b)[0] for _ in values])


def test_uniform_special_case_and_swapped_shape_antithetic_identity():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.beta(100), reference.uniform(100))
    bank = RandlibGenerator()
    paired = RandlibGenerator()
    paired.set_antithetic(True)
    assert_allclose(bank.beta(100, a=2, b=3), 1 - paired.beta(100, a=3, b=2), atol=3e-15, rtol=0)
    assert bank.get_seeds() == paired.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_budget_empty_and_endpoints(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    assert bank.beta(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.beta(20, a=2, b=3, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before
    values = bank.beta(20, a=0.0001, b=0.0002, legacy=legacy)
    assert np.any((values == 0) | (values == 1))
    assert np.all(np.isfinite(values))
    assert bank.get_seeds() != before


def test_source_minimum_and_invalid_intermediate_coefficients():
    bank = RandlibGenerator()
    before = bank.get_seeds()
    minimum = float(np.float32(1e-37))
    with pytest.raises(ValueError, match="source minimum"):
        bank.beta(a=minimum, b=2, legacy=True, source="c")
    assert bank.get_seeds() == before
    assert np.isfinite(bank.beta(a=minimum, b=2, legacy=True)[0])
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="coefficients"):
        bank.beta(20, a=1e30, b=1e30, legacy=True)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0},
        {"b": 0},
        {"a": -1},
        {"a": float("nan")},
        {"b": float("inf")},
        {"a": 1e-38, "legacy": True},
        {"b": 1e300, "legacy": True},
        {"legacy": 1},
        {"source": "bad"},
        {"source": "c"},
        {"max_attempts": 0},
        {"size": True},
        {"size": -1},
        {"a": None},
    ],
)
def test_invalid_requests(kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.beta(**kwargs)
    assert bank.get_seeds() == before
