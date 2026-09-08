import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_normal.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_normal_values_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    options = dict(
        mean=case["mean"],
        sd=case["sd"],
        legacy=True,
        source="c" if case["language"] == "c" else "fortran",
    )
    values, states = [], []
    for _ in case["values"]:
        values.append(bank.normal(**options)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, np.array(case["values"], dtype=np.float32).astype(float))
    assert_array_equal(states, case["states"])
    bank.reinitialize()
    assert_array_equal(bank.normal(len(values), **options), values)
    assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
def test_distribution_and_degenerate_consumption(legacy, source):
    options = dict(legacy=legacy, source=source)
    values = RandlibGenerator().normal(50000, mean=-2, sd=3, **options)
    assert abs(values.mean() + 2) < 0.06
    assert abs(values.var() - 9) < 0.2
    for z in [-2, -1, 0, 1, 2]:
        assert abs(np.mean(values <= -2 + 3 * z) - ndtr(z)) < 0.01
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.normal(100, mean=2, sd=0, **options), 2)
    reference.normal(100, **options)
    assert bank.get_seeds() == reference.get_seeds()
    before = bank.get_seeds()
    assert bank.normal(0, **options).size == 0
    assert bank.get_seeds() == before
    assert not values.flags.writeable


def test_inverse_cdf_and_antithetic_symmetry():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.normal(1000)
    assert_allclose(ndtr(values), reference.uniform(1000), atol=3e-16, rtol=0)
    assert bank.get_seeds() == reference.get_seeds()
    other = RandlibGenerator()
    other.set_antithetic(True)
    assert_allclose(values, -other.normal(1000), atol=2e-13, rtol=0)
    assert bank.get_seeds() == other.get_seeds()


@pytest.mark.parametrize("legacy", [False, True])
def test_failed_budget_and_overflow_preserve_state(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.normal(20, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before
    sd = float(np.finfo(np.float32 if legacy else np.float64).max)
    with pytest.raises(ArithmeticError, match="overflow"):
        bank.normal(20, sd=sd, legacy=legacy)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "options",
    [
        {"mean": float("nan")},
        {"mean": float("inf")},
        {"sd": -1},
        {"sd": float("nan")},
        {"sd": float("inf")},
        {"sd": 1e-300, "legacy": True},
        {"mean": -1e-300, "legacy": True},
        {"mean": 1e300, "legacy": True},
        {"sd": 1e300, "legacy": True},
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
        bank.normal(**options)
    assert bank.get_seeds() == before
