from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    RanlistSession,
    RanlistSpecification,
    ranlist_restricted,
    ranlist_unrestricted,
)


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_interleaved_enrollment_matches_indexed_kernels(restricted, legacy):
    spec = RanlistSpecification(
        (1, 2),
        restricted=restricted,
        legacy=legacy,
        strata=("North", "South"),
        treatments=("A", "B"),
        balance=(1, 3) if restricted else (1, 1),
    )
    original = RanlistSession(spec)
    state, first = original.enroll([[2, 1, 2], [1, 1, 2]])
    assert original.current_patients == (0, 0)
    assert state.current_patients == (3, 3)
    assert_array_equal(first.patients, [[1, 1, 2], [2, 3, 3]])
    assert_array_equal(first.strata, [[2, 1, 2], [1, 1, 2]])
    later, second = state.enroll([1, 2, 2])
    assert later.current_patients == (4, 5)
    assert_array_equal(second.patients, [4, 4, 5])
    for stream in [1, 2]:
        mask = first.strata == stream
        options = dict(seed=spec.seed, stream=stream, legacy=legacy)
        if restricted:
            expected = ranlist_restricted(
                first.patients[mask], spec.weights, balance=spec.balance, **options
            ).treatments
        else:
            expected = ranlist_unrestricted(
                first.patients[mask], spec.weights, **options
            ).treatments
        assert_array_equal(first.treatments[mask], expected)
    inquiry = later.inquire(first.patients, strata=first.strata)
    assert_array_equal(inquiry.treatments, first.treatments)
    assert later.current_patients == (4, 5)
    with pytest.raises(ValueError, match="not been randomized"):
        later.inquire(6, strata=2)
    # List generation can inspect future assignments without enrollment.
    assert spec.allocate(6, strata=2).treatments in (1, 2)
    for field in [first.patients, first.strata, first.treatments]:
        assert not field.flags.writeable


def test_batch_enrollment_equals_sequential_enrollment():
    spec = RanlistSpecification((2, 3, 2), restricted=True, balance=(1, 5), strata=("", "", ""))
    arrivals = [3, 1, 3, 2, 1, 1, 2, 3, 3, 1] * 10
    batch_state, batch = RanlistSession(spec).enroll(arrivals)
    state = RanlistSession(spec)
    expected = []
    for stream in arrivals:
        state, result = state.enroll(stream)
        assert result.treatments.shape == ()
        expected.append(int(result.treatments))
    assert_array_equal(batch.treatments, expected)
    assert batch_state.current_patients == state.current_patients


def test_failed_allocation_does_not_advance_state():
    spec = RanlistSpecification((1, 1), restricted=True, max_blocks=1)
    state = RanlistSession(spec, (2,))
    with pytest.raises(ValueError, match="max_blocks"):
        state.enroll()
    assert state.current_patients == (2,)
    assert state.inquire(2).patients == 2


def test_snapshots_and_empty_batches():
    weights, names, counters = [1, 2], ["North", "South"], [3, 4]
    spec = RanlistSpecification(weights, strata=names)
    state = RanlistSession(spec, counters)
    weights[0], names[0], counters[0] = 9, "Changed", 999
    assert spec.weights == (1, 2)
    assert spec.strata == ("North", "South")
    assert state.current_patients == (3, 4)
    later, empty = state.enroll(np.empty((0, 2)))
    assert later == state
    assert empty.treatments.shape == (0, 2)
    with pytest.raises(FrozenInstanceError):
        state.current_patients = (0, 0)


def test_broadcast_queries_and_source_label_widths():
    spec = RanlistSpecification(
        (1, 1), strata=("A" * 30, "B" * 30), title=("T" * 80,) * 9, phrase="P" * 31
    )
    result = spec.allocate([[1], [2]], strata=[1, 2])
    assert result.treatments.shape == (2, 2)
    assert_array_equal(result.patients, [[1, 1], [2, 2]])
    assert_array_equal(result.strata, [[1, 2], [1, 2]])
    assert RanlistSpecification((1,), strata=(" ",), phrase="abc ").phrase == "abc"


@pytest.mark.parametrize(
    "options",
    [
        {"strata": ()},
        {"strata": ("",) * 21},
        {"strata": ("A", "")},
        {"strata": "A"},
        {"strata": ("A" * 31,)},
        {"strata": ("é",)},
        {"title": ()},
        {"title": ("",)},
        {"title": ("a",) * 10},
        {"title": ("\na",)},
        {"treatments": ("A",)},
        {"treatments": ("A", "")},
        {"phrase": "p" * 32},
        {"phrase": "\t"},
        {"restricted": 1},
        {"legacy": 1},
        {"balance": (1, 2)},
        {"max_blocks": 0},
        {"max_blocks": True},
        {"seed": (0, 1)},
    ],
)
def test_invalid_specification(options):
    with pytest.raises(ValueError):
        RanlistSpecification((1, 2), **options)


@pytest.mark.parametrize("counters", [(1, 2), (-1,), (1.5,), (2**53,), [[1]]])
def test_invalid_counters(counters):
    with pytest.raises(ValueError):
        RanlistSession(RanlistSpecification((1, 2)), counters)


@pytest.mark.parametrize("strata", [0, 2, -1, 1.5, np.inf])
def test_invalid_strata(strata):
    state = RanlistSession(RanlistSpecification((1, 2)))
    with pytest.raises(ValueError):
        state.enroll(strata)
    with pytest.raises(ValueError):
        state.specification.allocate(1, strata=strata)


def test_counter_limit_does_not_round_or_overflow():
    state = RanlistSession(RanlistSpecification((1,)), (2**53 - 2,))
    later, result = state.enroll()
    assert result.patients == 2**53 - 1
    assert later.current_patients == (2**53 - 1,)
    with pytest.raises(ValueError, match="patient number limit"):
        later.enroll()


def test_unenrolled_inquiry_is_rejected_before_generating_future_blocks():
    state = RanlistSession(RanlistSpecification((1, 1), restricted=True, max_blocks=1))
    with pytest.raises(ValueError, match="not been randomized"):
        state.inquire(100000)
