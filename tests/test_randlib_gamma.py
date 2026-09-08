import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import gammainc

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_gamma.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_gamma_values_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    options = dict(
        shape=case["shape"],
        rate=case["rate"],
        legacy=True,
        source="c" if case["language"] == "c" else "fortran",
    )
    values, states = [], []
    for shape in case["shapes"]:
        options["shape"] = shape
        values.append(bank.gamma(**options)[0])
        states.append(bank.get_seeds())
    # Allow float32 transcendental-library rounding across platforms.
    expected = np.array(case["values"], dtype=np.float32).astype(float)
    assert_allclose(
        values, expected, rtol=4e-6, atol=2 * float(np.finfo(np.float32).smallest_subnormal)
    )
    assert_array_equal(states, case["states"])
    bank.reinitialize()
    if case["shape"] > 0:
        assert_array_equal(bank.gamma(len(values), **options), values)
        assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
@pytest.mark.parametrize("shape", [0.1, 1, 5, 20])
def test_distribution(legacy, source, shape):
    n, rate = 20000, 2.5
    values = RandlibGenerator().gamma(n, shape=shape, rate=rate, legacy=legacy, source=source)
    assert np.all(values >= 0)
    mean, variance = shape / rate, shape / rate**2
    assert abs(values.mean() - mean) < 6 * np.sqrt(variance / n)
    assert abs(values.var() - variance) < 6 * variance * np.sqrt((2 + 6 / shape) / n)
    assert abs(np.mean(values <= mean) - gammainc(shape, shape)) < 0.015
    assert not values.flags.writeable


@pytest.mark.parametrize("shape", [0.1, 0.5, 1, 3.686, 13.022, 100000])
def test_inverse_cdf_and_fixed_consumption(shape):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.gamma(1000, shape=shape, rate=2.3)
    assert_allclose(gammainc(shape, values * 2.3), reference.uniform(1000), atol=3e-13, rtol=0)
    assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_failed_budget_overflow_empty_and_underflow(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.gamma(20, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before
    rate = float(np.finfo(np.float32 if legacy else np.float64).smallest_subnormal)
    with pytest.raises(ArithmeticError, match="overflow"):
        bank.gamma(20, rate=rate, legacy=legacy)
    assert bank.get_seeds() == before
    assert bank.gamma(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    values = bank.gamma(20, shape=0.0001, legacy=legacy)
    assert np.all(np.isfinite(values))
    assert np.any(values == 0)
    assert bank.get_seeds() != before


@pytest.mark.parametrize(
    "options",
    [
        {"shape": 0},
        {"shape": -1},
        {"shape": float("nan")},
        {"shape": float("inf")},
        {"rate": 0},
        {"rate": -1},
        {"rate": float("nan")},
        {"rate": float("inf")},
        {"shape": 1e-300, "legacy": True},
        {"rate": 1e-300, "legacy": True},
        {"shape": 1e300, "legacy": True},
        {"rate": 1e300, "legacy": True},
        {"legacy": 1},
        {"source": "bad"},
        {"source": "c"},
        {"max_attempts": 0},
        {"size": -1},
        {"size": True},
    ],
)
def test_invalid_options_leave_state(options):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.gamma(**options)
    assert bank.get_seeds() == before
