import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_sampling.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_sampling_and_consumed_states(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    rows = []
    for _ in range(5):
        z = int(bank.integer_uniform(case["low"], case["high"], legacy=True)[0])
        rows.append([z, *bank.get_seeds()])
    assert_array_equal(rows, case["integers"])
    u = bank.uniform(
        low=case["lower"],
        high=case["upper"],
        legacy=True,
        source="c" if case["language"] == "c" else "fortran",
    )[0]
    assert u == float(np.float32(case["uniform"][0]))
    assert bank.get_seeds() == tuple(case["uniform"][1:])
    permutation = bank.permutation([-3, -2, -1, 0, 1, 2, 3], legacy=True)
    assert_array_equal(permutation, case["permutation"][:7])
    assert bank.get_seeds() == tuple(case["permutation"][7:])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("width", [2, 17, 1073741783])
def test_batches_consume_exactly_the_same_draws_as_scalar_calls(legacy, width):
    batched, scalar = RandlibGenerator(), RandlibGenerator()
    values = batched.integer_uniform(0, width - 1, 200, legacy=legacy)
    expected = [scalar.integer_uniform(0, width - 1, legacy=legacy)[0] for _ in range(200)]
    assert_array_equal(values, expected)
    assert batched.get_seeds() == scalar.get_seeds()
    assert not values.flags.writeable


def test_default_unbiased_rejection_against_raw_oracle():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank, oracle = RandlibGenerator(seed), RandlibGenerator(seed)
    expected = []
    rejected = 0
    for raw in oracle.integers(100):
        value = int(raw) - 1
        if value >= (2147483562 // 4) * 4:
            rejected += 1
            continue
        expected.append(-2 + value % 4)
        if len(expected) == 20:
            break
    assert rejected == 1
    assert_array_equal(bank.integer_uniform(-2, 1, 20), expected)
    raw_state = RandlibGenerator(seed)
    raw_state.integers(21)
    assert bank.get_seeds() == raw_state.get_seeds()


def test_full_range_and_legacy_rejection_budget():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    assert bank.integer_uniform(0, 2147483561)[0] == 2147483561
    bank.reinitialize()
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.integer_uniform(0, 2147483561, legacy=True, max_attempts=10)
    assert bank.get_seeds() == before
    bank.set_antithetic(True)
    assert bank.integer_uniform(0, 2147483561, legacy=True, max_attempts=1)[0] == 0


def test_constant_bounds_have_different_draw_consumption():
    bank = RandlibGenerator()
    initial = bank.get_seeds()
    assert_array_equal(bank.integer_uniform(-4, -4, 10), -4)
    assert bank.get_seeds() == initial
    assert_array_equal(bank.uniform(10, low=2, high=2), 2)
    oracle = RandlibGenerator()
    oracle.integers(10)
    assert bank.get_seeds() == oracle.get_seeds()
    assert_array_equal(bank.permutation([]), [])
    assert_array_equal(bank.permutation([9]), [9])
    assert bank.get_seeds() == oracle.get_seeds()


def test_permutation_preserves_multiset_input_and_source_swap_order():
    original = np.array([3, -2, 3, 0, 8, 9])
    bank, oracle = RandlibGenerator(), RandlibGenerator()
    expected = original.copy()
    for i in range(len(expected) - 1):
        j = int(oracle.integer_uniform(i, len(expected) - 1)[0])
        expected[i], expected[j] = expected[j], expected[i]
    result = bank.permutation(original)
    assert_array_equal(result, expected)
    assert_array_equal(np.sort(result), np.sort(original))
    assert_array_equal(original, [3, -2, 3, 0, 8, 9])
    assert bank.get_seeds() == oracle.get_seeds()
    assert not result.flags.writeable
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError):
        bank.permutation(list(range(10)), max_attempts=2)
    assert bank.get_seeds() == before


def test_large_real_bounds_do_not_overflow():
    limit = np.finfo(float).max
    bank = RandlibGenerator()
    values = bank.uniform(1000, low=-limit, high=limit)
    assert np.all(np.isfinite(values))
    assert np.all((values >= -limit) & (values <= limit))
    assert_array_equal(bank.uniform(1000, low=limit, high=limit), limit)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low": 2, "high": 1},
        {"low": np.nan},
        {"high": np.inf},
        {"low": -1e38, "high": 3e38, "legacy": True},
    ],
)
def test_invalid_real_bounds_preserve_state(kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.uniform(**kwargs)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "low,high,options",
    [
        (0, 2**31 - 1, {}),
        (2, 1, {}),
        (False, 1, {}),
        (0, 1.0, {}),
        (0, 1, {"max_attempts": 0}),
        (0, 1, {"legacy": 1}),
    ],
)
def test_invalid_integer_options_preserve_state(low, high, options):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.integer_uniform(low, high, **options)
    assert bank.get_seeds() == before


@pytest.mark.parametrize("values", [[1.0, 2], [[1, 2]], [True, 1], [2**31]])
def test_invalid_permutations(values):
    with pytest.raises(ValueError):
        RandlibGenerator().permutation(values)


def test_c_scaling_can_round_to_upper_endpoint():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    bank = RandlibGenerator(seed)
    assert bank.uniform(legacy=True, source="c")[0] == 1.0
    bank.reinitialize()
    assert bank.uniform(legacy=True)[0] == 1 - 2**-24
    bank.reinitialize()
    assert bank.uniform()[0] < 1.0
