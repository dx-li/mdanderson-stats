"""Original stream/seed comparisons and independent integer recurrences."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import ranlist_integers, ranlist_seeds, ranlist_uniform

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranlist_random.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_streams_and_block_jumps(case):
    kwargs = {"seed": tuple(case["seed"]), "stream": case["stream"]}
    actual = ranlist_integers(case["positions"], **kwargs)
    assert_array_equal(actual, case["integers"])
    source = ranlist_uniform(case["positions"], legacy=True, **kwargs)
    assert_array_equal(source.astype(np.float32), np.array(case["uniform"], dtype=np.float32))
    assert_allclose(
        ranlist_uniform(case["positions"], **kwargs), actual / 2147483563, rtol=0, atol=0
    )


@pytest.mark.parametrize("case", FIXTURE["phrases"])
def test_native_phrase_mapping(case):
    assert ranlist_seeds(case["phrase"]) == tuple(case["seed"])


def test_independent_scalar_recurrence_and_arbitrary_position_order():
    first, second = 1234567890, 123456789
    expected = []
    for _ in range(1000):
        first = (40014 * first) % 2147483563
        second = (40692 * second) % 2147483399
        difference = first - second
        expected.append(difference if difference > 0 else difference + 2147483562)
    positions = np.array([[1000, 1, 10], [10, 25, 2]])
    actual = ranlist_integers(positions)
    assert_array_equal(actual, np.array(expected)[positions - 1])
    assert not actual.flags.writeable
    assert_array_equal(ranlist_integers(positions), actual)
    assert ranlist_integers(np.empty((0, 2))).shape == (0, 2)


def test_stream_spacing_and_large_positions():
    at = np.array([1, 100, 10000])
    assert_array_equal(ranlist_integers(at, stream=2), ranlist_integers(at + 2**50, stream=1))
    position = 2**53 - 1
    first = 1234567890 * pow(40014, position, 2147483563) % 2147483563
    second = 123456789 * pow(40692, position, 2147483399) % 2147483399
    expected = first - second
    if expected < 1:
        expected += 2147483562
    assert ranlist_integers(position) == expected


def test_uniform_bounds_and_original_float32_scaling():
    # Choose component seeds so the next states nearly maximize the difference.
    seed = (65421664, 1481316021)
    raw = ranlist_integers(1, seed=seed)
    assert raw == 2147483561
    assert 0 < ranlist_uniform(1, seed=seed) < 1
    assert ranlist_uniform(1, seed=seed, legacy=True) == 1 - 2**-24
    u = ranlist_uniform(np.arange(1, 10001))
    assert np.all((u > 0) & (u < 1)) and not u.flags.writeable
    assert abs(u.mean() - 0.5) < 0.01


def test_phrase_trailing_spaces_case_and_unknown_character_rules():
    assert ranlist_seeds("") == (1234567890, 123456789)
    assert ranlist_seeds("   ") == ranlist_seeds("")
    assert ranlist_seeds("abc   ") == ranlist_seeds("abc")
    assert ranlist_seeds("abc\t") != ranlist_seeds("abc")
    assert ranlist_seeds("abc") != ranlist_seeds("ABC")
    assert ranlist_seeds("-") == ranlist_seeds("=")  # Both absent from the source table.
    seed = ranlist_seeds("trial 123")
    assert_array_equal(ranlist_integers([1, 2], seed=seed), ranlist_integers([1, 2], seed=seed))


@pytest.mark.parametrize("positions", [0, -1, 1.5, np.nan, np.inf, 2**53])
def test_invalid_positions(positions):
    with pytest.raises(ValueError):
        ranlist_integers(positions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream": 0},
        {"stream": 33},
        {"stream": True},
        {"stream": 1.5},
        {"seed": (0, 1)},
        {"seed": (1, 2147483399)},
        {"seed": (2147483563, 1)},
        {"seed": (True, 1)},
        {"seed": (1.0, 2)},
        {"seed": (1,)},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ranlist_integers(1, **kwargs)


@pytest.mark.parametrize("phrase", [None, 123, "café"])
def test_invalid_phrase(phrase):
    with pytest.raises(ValueError):
        ranlist_seeds(phrase)


def test_invalid_legacy_flag():
    with pytest.raises(ValueError):
        ranlist_uniform(1, legacy=1)
