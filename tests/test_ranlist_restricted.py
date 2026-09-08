import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import ranlist_integers, ranlist_restricted

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranlist_restricted.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_restricted(case):
    result = ranlist_restricted(
        case["patients"],
        case["counts"],
        balance=case["balance"],
        seed=case["seed"],
        stream=case["stream"],
        legacy=True,
    )
    assert_array_equal(result.treatments, case["treatments"])
    assert_array_equal(result.multiplier, case["multiplier"])
    length = sum(case["counts"]) * case["multiplier"]
    assert_array_equal(result.block_start, (result.patients - 1) // length * length + 1)
    assert_array_equal(result.block_end, result.block_start + length - 1)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("balance", [(1, 1), (3, 3), (1, 5)])
def test_every_complete_block_has_exact_treatment_counts(legacy, balance):
    result = ranlist_restricted(np.arange(1, 1001), [2, 3, 2], balance=balance, legacy=legacy)
    for start in np.unique(result.block_start):
        mask = result.block_start == start
        if result.block_end[mask][0] > 1000:
            continue
        assert_array_equal(
            np.bincount(result.treatments[mask], minlength=4)[1:],
            np.array([2, 3, 2]) * result.multiplier[mask][0],
        )
    if balance[0] != balance[1]:
        assert (np.unique(result.multiplier).size == 1) == legacy


@pytest.mark.parametrize("legacy", [False, True])
def test_queries_are_repeatable_shaped_and_independent(legacy):
    full = ranlist_restricted(np.arange(1, 201), [1, 2], balance=(1, 4), legacy=legacy)
    patients = np.array([[100, 1, 100], [200, 15, 7]])
    result = ranlist_restricted(patients, [1, 2], balance=(1, 4), legacy=legacy)
    for field in ["treatments", "block_start", "block_end", "multiplier"]:
        assert_array_equal(getattr(result, field), getattr(full, field)[patients - 1])
        assert not getattr(result, field).flags.writeable
    assert not result.patients.flags.writeable
    assert not result.counts.flags.writeable
    assert ranlist_restricted(100, [1, 2], balance=(1, 4), legacy=legacy).treatments.shape == ()
    assert ranlist_restricted(np.empty((0, 2)), [1, 2], legacy=legacy).treatments.shape == (0, 2)


def test_modern_matches_independent_forward_shuffle():
    # Independent stream from the already verified indexed generator. This
    # fixed case has no rejected draws, so each swap consumes one integer.
    raw = iter(ranlist_integers(np.arange(1, 1000)).tolist())
    expected = []
    for _ in range(20):
        multiplier = 1 + (next(raw) - 1) % 3
        block = [1, 2, 2] * multiplier
        for i in range(len(block) - 1):
            draw = next(raw) - 1
            width = len(block) - i
            assert draw < (2147483562 // width) * width
            j = i + draw % width
            block[i], block[j] = block[j], block[i]
        expected.extend(block)
    result = ranlist_restricted(np.arange(1, len(expected) + 1), [1, 2], balance=(1, 3))
    assert_array_equal(result.treatments, expected)


def test_integer_endpoint_rejection_and_source_skipping_are_preserved():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    assert ranlist_integers(1, seed=seed) == 2147483562
    modern = ranlist_restricted([1, 2, 3, 4], [1, 1], seed=seed)
    legacy = ranlist_restricted([1, 2, 3, 4], [1, 1], seed=seed, legacy=True)
    # Source rejects draw 1, uses draw 2 for the first block, then skips only
    # one draw for the second block, so it reuses draw 2 there.
    second_choice = 1 + (int(ranlist_integers(2, seed=seed)) - 1) % 2
    assert_array_equal(legacy.treatments, [second_choice, 3 - second_choice] * 2)
    assert_array_equal(modern.treatments[:2], [2, 1])


def test_huge_legacy_position_jumps_without_replaying_blocks():
    position = 2**53 - 1
    result = ranlist_restricted(position, [1, 1], legacy=True, max_blocks=1)
    start = (position - 1) // 2 * 2 + 1
    draw = int(ranlist_integers((start - 1) // 2 + 1)) - 1
    assert draw <= (2147483561 // 2) * 2
    assert result.treatments == 1 + draw % 2
    assert result.block_start == start
    assert result.block_end == start + 1


def test_work_limit_is_explicit_and_counts_preceding_blocks():
    with pytest.raises(ValueError, match="max_blocks"):
        ranlist_restricted(101, [1, 1], max_blocks=50)
    assert ranlist_restricted(101, [1, 1], max_blocks=51).block_start == 101
    with pytest.raises(ValueError, match="max_blocks"):
        ranlist_restricted([1, 101], [1, 1], legacy=True, max_blocks=1)
    assert ranlist_restricted(101, [1, 1], legacy=True, max_blocks=1).block_start == 101
    # Fails during generation when the upper-bound precheck cannot decide.
    with pytest.raises(ValueError, match="max_blocks"):
        ranlist_restricted(4, [1], balance=(1, 4), max_blocks=1, seed=(2, 2))


@pytest.mark.parametrize("counts", [[], [0], [-1], [1.5], [[1, 1]], [1] * 21, [501], [np.inf]])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):
        ranlist_restricted(1, counts)


@pytest.mark.parametrize(
    "options",
    [
        {"balance": (0, 1)},
        {"balance": (2, 1)},
        {"balance": (1, 251)},
        {"balance": (1.5, 2)},
        {"balance": (1,)},
        {"balance": [[1, 2]]},
        {"max_blocks": 0},
        {"max_blocks": True},
        {"max_blocks": 1.5},
        {"legacy": 1},
        {"stream": 0},
        {"stream": True},
        {"seed": (0, 1)},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        ranlist_restricted(1, [1, 1], **options)


@pytest.mark.parametrize("patients", [0, -1, 1.5, np.nan, 2**53])
def test_invalid_patients(patients):
    with pytest.raises(ValueError):
        ranlist_restricted(patients, [1, 1])


def test_largest_block_and_single_treatment():
    result = ranlist_restricted(np.arange(1, 501), [125, 125], balance=(2, 2))
    assert_array_equal(np.bincount(result.treatments)[1:], [250, 250])
    assert_array_equal(ranlist_restricted([1, 10000], [1]).treatments, [1, 1])


def test_default_rejection_consumes_the_next_draw():
    seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    draws = iter(ranlist_integers(np.arange(1, 30), seed=seed).tolist())
    expected = []
    rejected = 0
    for _ in range(2):
        block = [1, 2, 3, 4]
        for i in range(3):
            width = 4 - i
            while True:
                draw = next(draws) - 1
                if draw < (2147483562 // width) * width:
                    break
                rejected += 1
            j = i + draw % width
            block[i], block[j] = block[j], block[i]
        expected.extend(block)
    assert rejected == 1
    assert_array_equal(ranlist_restricted(np.arange(1, 9), [1] * 4, seed=seed).treatments, expected)
