import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    RanlistSession,
    RanlistSpecification,
    load_ranlist_session,
    ranlist_parameter_text,
    read_ranlist_parameters,
    save_ranlist_session,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranlist_parameters.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_parameter_records(case):
    state = read_ranlist_parameters(case["text"])
    spec = state.specification
    assert state.current_patients == (12, 345)
    assert spec.strata == ("North", "South")
    assert spec.treatments == ("Control", "Low", "High")
    assert spec.seed == (8241812, 190815113)
    assert spec.phrase == "abc"
    assert spec.legacy
    assert spec.restricted == case["restricted"]
    if spec.restricted:
        assert spec.weights == (1, 2, 3)
        assert spec.balance == (1, 4)
        assert ranlist_parameter_text(state) == case["text"]
    else:
        assert_array_equal(
            spec.weights, np.array([0.062, 0.1, 2.346], dtype=np.float32).astype(float)
        )
        assert spec.balance == (1, 1)
        original = replace(spec, weights=(0.0625, 0.1, 2.3456))
        rendered = ranlist_parameter_text(
            RanlistSession(original, state.current_patients), allow_rounding=True
        )
        native = case["text"].replace("  0  0\n", "  1  1\n")
        assert rendered == native
        with pytest.raises(ValueError, match="changes a weight"):
            ranlist_parameter_text(RanlistSession(original))
    restored = read_ranlist_parameters(ranlist_parameter_text(state))
    assert restored == state
    assert_array_equal(
        restored.inquire([1, 12], strata=1).treatments, state.inquire([1, 12], strata=1).treatments
    )


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_json_roundtrip_and_resume(tmp_path, restricted, legacy):
    spec = RanlistSpecification(
        (1, 2) if restricted else (0.123456789, 0.987654321),
        restricted=restricted,
        legacy=legacy,
        strata=("", ""),
        max_blocks=123,
        balance=(1, 3) if restricted else (1, 1),
    )
    state, _ = RanlistSession(spec).enroll([2, 1, 2, 1, 1])
    path = tmp_path / "session.json"
    save_ranlist_session(state, path)
    restored = load_ranlist_session(path)
    assert restored == state
    expected_state, expected = state.enroll([1, 2, 2])
    resumed, actual = restored.enroll([1, 2, 2])
    assert resumed == expected_state
    assert_array_equal(actual.treatments, expected.treatments)
    save_ranlist_session(resumed, path)
    assert load_ranlist_session(path) == resumed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session.json"]


def test_unnamed_records_and_implicit_decimal():
    state = RanlistSession(RanlistSpecification((0.1, 0.2), strata=("", ""), legacy=True))
    text = ranlist_parameter_text(state)
    assert text.splitlines().count(" " * 30) == 2
    parsed = read_ranlist_parameters(text.replace(" 0.100", "   100").replace("\n", "\r\n"))
    assert parsed.specification.strata == ("", "")
    assert parsed.specification.treatments == ("", "")
    assert_array_equal(
        parsed.specification.allocate([1, 2]).treatments,
        state.specification.allocate([1, 2]).treatments,
    )


@pytest.mark.parametrize(
    "spec,counters",
    [
        (RanlistSpecification((1, 2)), (0,)),
        (RanlistSpecification((10, 1), restricted=True, legacy=True), (0,)),
        (RanlistSpecification((0.0001, 1), legacy=True), (0,)),
        (RanlistSpecification((100, 1), legacy=True), (0,)),
        (RanlistSpecification((1, 2), legacy=True), (1000000,)),
    ],
)
def test_unrepresentable_source_export(spec, counters):
    with pytest.raises(ValueError):
        ranlist_parameter_text(RanlistSession(spec, counters), allow_rounding=True)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("Parameter file for RANLST", "invalid header"),
        lambda s: "\n".join(s.splitlines()[:-1]),
        lambda s: s + "unexpected\n",
        lambda s: s.replace("\n1\n", "\n*\n", 1),
        lambda s: s.replace("North", "Nørth"),
        lambda s: s.replace(" 2 3T", " 0 3T"),
    ],
)
def test_invalid_source_records(change):
    text = FIXTURE["cases"][0]["text"]
    changed = change(text)
    assert changed != text
    with pytest.raises(ValueError):
        read_ranlist_parameters(changed)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(version=2),
        lambda p: p.update(version=True),
        lambda p: p.update(extra=1),
        lambda p: p.pop("current_patients"),
        lambda p: p.update(current_patients=[]),
        lambda p: p.update(current_patients=[True]),
        lambda p: p["specification"].pop("seed"),
        lambda p: p["specification"].update(strata=None),
        lambda p: p["specification"].update(weights=["1"]),
        lambda p: p["specification"].update(treatments=[]),
        lambda p: p["specification"].update(weights=[float("nan")]),
    ],
)
def test_corrupt_json_is_rejected(tmp_path, change):
    path = tmp_path / "state.json"
    save_ranlist_session(RanlistSession(RanlistSpecification((1,))), path)
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_ranlist_session(path)


def test_duplicate_json_fields_are_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"version": 1, "version": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        load_ranlist_session(path)


def test_failed_replacement_cleans_temporary_file(tmp_path):
    # Real filesystem failure: a directory cannot be replaced by a regular file.
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("unchanged")
    with pytest.raises(OSError):
        save_ranlist_session(RanlistSession(RanlistSpecification((1,))), destination)
    assert (destination / "keep").read_text() == "unchanged"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("case", FIXTURE["workflows"])
def test_complete_native_program_enrolls_and_rewrites_python_parameters(case):
    initial = read_ranlist_parameters(case["initial"])
    updated, allocations = initial.enroll(case["arrivals"])
    assert_array_equal(allocations.treatments, case["treatments"])
    assert updated.current_patients == (4, 6)
    native_updated = read_ranlist_parameters(case["updated"])
    assert native_updated == updated
    assert native_updated.inquire(4, strata=2).treatments == case["inquiry"]
    assert ranlist_parameter_text(updated) == case["updated"]
