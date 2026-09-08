import json
from math import isqrt
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import RandlibGenerator, ranlist_integers

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_streams.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_stream_controls(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    rows = []

    def capture():
        rows.append([int(bank.integers()[0]), *bank.get_seeds()])

    for _ in range(4):
        capture()
    bank.set_antithetic(True)
    capture()
    capture()
    bank.reinitialize(0)
    capture()
    bank.reinitialize(1)
    capture()
    bank.advance_state(case["exponent"])
    capture()
    bank.reinitialize(-1)
    capture()
    bank.set_seeds((123, 456))
    capture()
    bank.set_antithetic(False)
    capture()
    bank.select(1)
    capture()
    bank.set_all_seeds((9876, 54321))
    capture()
    assert_array_equal(rows, case["rows"])


def test_batches_and_interleaved_streams_preserve_state():
    bank = RandlibGenerator()
    first = bank.integers(100)
    bank.select(2)
    second = bank.integers(77)
    bank.select(1)
    continued = bank.integers(30)
    assert_array_equal(first, ranlist_integers(np.arange(1, 101)))
    assert_array_equal(second, ranlist_integers(np.arange(1, 78), stream=2))
    assert_array_equal(continued, ranlist_integers(np.arange(101, 131)))
    state = bank.get_seeds()
    assert bank.integers(0).size == 0
    assert bank.get_seeds() == state
    assert not first.flags.writeable
    assert_array_equal(RandlibGenerator().integers(100), first)


def test_antithetic_complements_raw_draws_and_has_per_stream_state():
    bank, ordinary = RandlibGenerator(), RandlibGenerator()
    bank.set_antithetic(True)
    assert_array_equal(bank.integers(1000) + ordinary.integers(1000), 2147483563)
    assert bank.get_seeds() == ordinary.get_seeds()
    bank.select(2)
    assert_array_equal(bank.integers(10), ranlist_integers(np.arange(1, 11), stream=2))
    bank.select(1)
    bank.reinitialize()
    assert_array_equal(
        bank.uniform(100, legacy=True),
        (
            (2147483563 - ranlist_integers(np.arange(1, 101))).astype(np.float32)
            * np.float32(4.656613057e-10)
        ).astype(float),
    )


def test_next_block_uses_block_start_not_current_position():
    bank = RandlibGenerator()
    bank.integers(123)
    bank.reinitialize(1)
    assert_array_equal(bank.integers(8), ranlist_integers(2**30 + np.arange(1, 9)))
    bank.reinitialize(0)
    assert_array_equal(bank.integers(8), ranlist_integers(2**30 + np.arange(1, 9)))
    bank.reinitialize()
    assert_array_equal(bank.integers(8), ranlist_integers(np.arange(1, 9)))


def test_advance_adopts_new_initial_and_block_states():
    bank = RandlibGenerator((123, 456))
    bank.integers(5)
    before = bank.get_seeds()
    bank.advance_state(1000)
    expected = (
        before[0] * pow(40014, 2**1000, 2147483563) % 2147483563,
        before[1] * pow(40692, 2**1000, 2147483399) % 2147483399,
    )
    assert bank.get_seeds() == expected
    bank.integers(10)
    bank.reinitialize()
    assert bank.get_seeds() == expected
    # Establish the prime-modulus premise used by the exponent reduction.
    assert all(all(m % d for d in range(2, isqrt(m) + 1)) for m in (2147483563, 2147483399))


def test_set_all_seeds_resets_stream_one_when_another_stream_is_selected():
    bank = RandlibGenerator()
    bank.integers(5)
    bank.select(2)
    bank.set_antithetic(True)
    bank.set_all_seeds((12, 34))
    assert bank.stream == 2
    assert_array_equal(
        bank.integers(5), 2147483563 - ranlist_integers(np.arange(1, 6), seed=(12, 34), stream=2)
    )
    bank.select(1)
    assert_array_equal(bank.integers(5), ranlist_integers(np.arange(1, 6), seed=(12, 34)))


@pytest.mark.parametrize(
    "method,value",
    [
        ("select", 0),
        ("select", 33),
        ("select", True),
        ("integers", -1),
        ("integers", 1.0),
        ("integers", True),
        ("integers", 1000001),
        ("reinitialize", 2),
        ("reinitialize", False),
        ("advance_state", -1),
        ("advance_state", 2**53),
        ("advance_state", 1.0),
        ("set_seeds", (0, 1)),
        ("set_all_seeds", (1, 0)),
        ("set_antithetic", 1),
    ],
)
def test_invalid_operations_do_not_change_selected_state(method, value):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        getattr(bank, method)(value)
    assert bank.stream == 1
    assert bank.get_seeds() == before


def test_uniform_bounds_and_draw_limit_override():
    bank = RandlibGenerator(max_draws=3)
    result = bank.uniform(3)
    assert np.all((result > 0) & (result < 1))
    assert not result.flags.writeable
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.uniform(1, legacy=1)
    assert bank.get_seeds() == before
    with pytest.raises(ValueError):
        bank.integers(4)
