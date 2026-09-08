import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import RandlibGenerator, ranlist_seeds

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_seeding.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_phrase_hash_and_stream(case):
    bank = RandlibGenerator(stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    bank.integers(7)
    assert bank.set_phrase(
        case["phrase"], stream=case["stream"], source="c" if case["language"] == "c" else "fortran"
    ) == tuple(case["seed"])
    values, states = [], []
    for _ in case["values"]:
        values.append(bank.integers()[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, case["values"])
    assert_array_equal(states, case["states"])


def test_phrase_reset_covers_all_streams_and_block_starts():
    bank = RandlibGenerator(stream=32)
    for stream in range(1, 33):
        bank.select(stream)
        bank.integers(7)
        bank.reinitialize(1)
    seed = bank.set_phrase("A sample phrase!")
    assert bank.stream == 1
    reference = RandlibGenerator(seed)
    for stream in range(1, 33):
        bank.select(stream)
        reference.select(stream)
        assert_array_equal(bank.integers(10), reference.integers(10))
        bank.reinitialize(0)
        reference.reinitialize(0)
        assert bank.get_seeds() == reference.get_seeds()
        bank.reinitialize(1)
        reference.reinitialize(1)
        assert bank.get_seeds() == reference.get_seeds()


@pytest.mark.parametrize(
    "moment,phrase",
    [
        (datetime(2026, 1, 1), "000000.000"),
        (datetime(2026, 1, 1, 23, 59, 59, 999999), "235959.999"),
        (datetime(2026, 1, 1, 12, 34, 56, 789999, tzinfo=UTC), "123456.789"),
    ],
)
def test_time_uses_wall_time_to_milliseconds(moment, phrase):
    bank = RandlibGenerator(stream=32)
    bank.set_antithetic(True)
    bank.integers(4)
    seed = bank.set_time(moment)
    assert seed == ranlist_seeds(phrase)
    assert bank.stream == 32
    reference = RandlibGenerator(seed, stream=32)
    reference.set_antithetic(True)
    assert_array_equal(bank.integers(10), reference.integers(10))


def test_clock_seed_can_be_recorded_and_replayed():
    bank = RandlibGenerator(stream=2)
    seed = bank.set_time()
    reference = RandlibGenerator(seed, stream=2)
    assert_array_equal(bank.integers(100), reference.integers(100))


@pytest.mark.parametrize(
    "phrase,stream", [(None, 1), ("é", 1), ("abc", 0), ("abc", 33), ("abc", True)]
)
def test_invalid_phrase_or_stream_preserves_bank(phrase, stream):
    bank = RandlibGenerator(stream=32)
    bank.integers(7)
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.set_phrase(phrase, stream=stream)
    assert bank.stream == 32
    assert bank.get_seeds() == before


@pytest.mark.parametrize("moment", ["123456.789", 1, np.nan])
def test_invalid_time_preserves_bank(moment):
    bank = RandlibGenerator(stream=32)
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.set_time(moment)
    assert bank.stream == 32
    assert bank.get_seeds() == before


@pytest.mark.parametrize("phrase", ["a b", "\tabc", "\x00", "~"])
def test_undefined_c_character_reads_are_rejected(phrase):
    bank = RandlibGenerator(stream=32)
    before = bank.get_seeds()
    with pytest.raises(ValueError, match="beyond its table"):
        bank.set_phrase(phrase, source="c")
    assert bank.stream == 32
    assert bank.get_seeds() == before
    assert bank.set_phrase(phrase) == ranlist_seeds(phrase)


def test_invalid_source_preserves_state():
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        bank.set_phrase("abc", source="bad")
    assert bank.get_seeds() == before
