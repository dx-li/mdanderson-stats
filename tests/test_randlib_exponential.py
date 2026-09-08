import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_exponential.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_exponential_values_and_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    source = "c" if case["language"] == "c" else "fortran"
    values, states = [], []
    for _ in case["values"]:
        values.append(bank.exponential(mean=case["mean"], legacy=True, source=source)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, np.array(case["values"], dtype=np.float32).astype(float))
    assert_array_equal(states, case["states"])
    batch = RandlibGenerator(case["seed"], stream=case["stream"])
    batch.set_antithetic(case["antithetic"])
    assert_array_equal(
        batch.exponential(len(values), mean=case["mean"], legacy=True, source=source), values
    )
    assert batch.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize("legacy", [False, True])
def test_empirical_distribution_and_zero_mean(legacy):
    values = RandlibGenerator().exponential(50000, mean=2.5, legacy=legacy)
    assert np.all(values >= 0)
    assert abs(values.mean() - 2.5) < 0.06
    assert abs(values.var() - 6.25) < 0.3
    assert abs(np.mean(values <= 2.5) - (1 - np.exp(-1))) < 0.01
    bank, reference = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(bank.exponential(100, mean=0, legacy=legacy), 0)
    reference.exponential(100, mean=1, legacy=legacy)
    assert bank.get_seeds() == reference.get_seeds()
    before = bank.get_seeds()
    assert bank.exponential(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    assert not values.flags.writeable


def test_modern_inverse_transform_and_fixed_consumption():
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.exponential(100, mean=3)
    uniforms = reference.uniform(100)
    assert_array_equal(values, -np.log(uniforms) * 3)
    assert bank.get_seeds() == reference.get_seeds()


def test_exact_half_source_boundary():
    seed = ((1073741825 * pow(40014, -1, 2147483563)) % 2147483563, pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    assert bank.exponential(legacy=True)[0] == 0
    raw = RandlibGenerator(seed)
    raw.integers(1)
    assert bank.get_seeds() == raw.get_seeds()
    assert RandlibGenerator(seed).exponential()[0] > 0


def test_c_upper_endpoint_is_reported_without_mutating_bank():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="source table"):
        bank.exponential(legacy=True, source="c")
    assert bank.get_seeds() == before
    assert np.isfinite(bank.exponential(legacy=True)[0])


@pytest.mark.parametrize("legacy", [False, True])
def test_failed_budget_and_overflow_preserve_state(legacy):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.exponential(20, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before
    mean = float(np.finfo(np.float32 if legacy else np.float64).max)
    with pytest.raises(ArithmeticError, match="overflow"):
        bank.exponential(20, mean=mean, legacy=legacy)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "options",
    [
        {"mean": -1},
        {"mean": float("nan")},
        {"mean": float("inf")},
        {"mean": 1e-300, "legacy": True},
        {"mean": 1e300, "legacy": True},
        {"legacy": 1},
        {"source": "bad"},
        {"source": "c"},
        {"max_attempts": 0},
    ],
)
def test_invalid_options(options):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.exponential(**options)
    assert bank.get_seeds() == before
