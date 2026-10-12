"""Portable native references and explicit safety differences in current rules."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.uaroet_desktop_rules import (
    UAROETCohortState,
    uaroet_desktop_assignment,
    uaroet_desktop_screen,
    uaroet_replay_cohorts,
    uaroet_utility_cutoff,
)

ROOT = Path(__file__).parent / "fixtures"


def cases(name):
    return json.loads((ROOT / f"uaroet-desktop-{name}.json").read_text())["cases"]


@pytest.mark.parametrize("case", cases("acceptability"))
def test_native_screen(case):
    x = case["inputs"]
    D = len(x["tails"])
    actual = uaroet_desktop_screen(
        x["tails"],
        x.get("eff_tails", [0] * D),
        x.get("utility", [20, 40, 60]),
        x.get("best", [0.1, 0.2, 0.7]),
        bad_toxicity=x["bad_tox"],
        bad_efficacy=x["bad_eff"],
        complete=x.get("observed", [3] * D),
        highest_allowed=x.get("highest", 2),
        toxicity_cutoff=x.get("threshold", 0.8),
        efficacy_cutoff=x.get("threshold", 0.8),
        utility_delta=x.get("delta", 100),
        minimum_best=x.get("min_best", 0),
        minimum_bad_efficacy=x.get("min_bad_eff", 1),
        minimum_complete=x.get("min_observed", 0),
        minimum_total_toxicity=x.get("min_total_tox", 0),
        filter_utility=x.get("filter_utility", True),
        profile="legacy_1_8",
    )
    assert list(actual) == case["status"]


@pytest.mark.parametrize("case", cases("randomization"))
def test_native_assignment(case):
    x = case["inputs"]
    a = uaroet_desktop_assignment(
        tuple(x["status"]),
        x["utility"],
        [0.2, 0.4, 0.6],
        x["treated"],
        hybrid=x["hybrid"],
        mean_weight=x["mean_weight"],
        surplus=x["surplus"],
        uniform=x["value"],
        profile="legacy_1_8",
    )
    assert a.selected == case["result"]["selected"]
    np.testing.assert_allclose(a.weights, case["result"]["weights"], atol=2e-15)


@pytest.mark.parametrize("case", cases("shrink"))
def test_native_cutoff(case):
    x = case["inputs"]
    actual = uaroet_utility_cutoff(
        planned=x["planned"],
        steps=x["steps"],
        observed=x["observed"],
        initial=x["initial_cutoff"],
        final=x["final_cutoff"],
        profile="legacy_1_8",
    )
    assert actual == pytest.approx(case["result"], abs=2e-14)


@pytest.mark.parametrize("case", cases("controller"))
def test_native_live_cohorts(case):
    x = case["inputs"]
    state = UAROETCohortState((0, 0, 0))
    trace = []
    for utility, status in zip(x["utility_sequence"], x["status_sequence"]):
        selected = state.choose(tuple(status), utility, first_size=x["first"], profile="legacy_1_8")
        if selected is None:
            trace.append(
                dict(selected=-1, stop=True, new_open=state.new_open, old_open=state.old_open)
            )
            break
        state = state.advance(selected, new_size=x["new"], old_size=x["old"], profile="legacy_1_8")
        trace.append(
            dict(selected=selected, stop=False, new_open=state.new_open, old_open=state.old_open)
        )
    assert trace == case["trace"]


@pytest.mark.parametrize("case", cases("replay"))
def test_native_saved_history(case):
    a = uaroet_replay_cohorts(
        tuple(case["doses"]),
        dose_count=3,
        new_size=case["new"],
        old_size=case["old"],
        profile="legacy_1_8",
    )
    expected = case["result"]
    for key in ("first_seen", "total", "last", "new_open", "old_open"):
        assert getattr(a, key) == (tuple(expected[key]) if key == "first_seen" else expected[key])
    assert (a.new_start, a.old_start) == (expected["new_start"], expected["old_start"])


def test_safe_screen_never_restores_toxic_lowest_dose():
    kwargs = dict(
        bad_toxicity=[1, 1],
        bad_efficacy=[0, 0],
        complete=[1, 1],
        highest_allowed=1,
        minimum_total_toxicity=3,
    )
    safe = uaroet_desktop_screen([0.9, 0.9], [0, 0], [20, 40], [0.5, 0.5], **kwargs)
    legacy = uaroet_desktop_screen(
        [0.9, 0.9], [0, 0], [20, 40], [0.5, 0.5], profile="legacy_1_8", **kwargs
    )
    assert safe == (-2, -2) and legacy == (1, -2)


def test_safe_first_cohort_does_not_bypass_toxicity():
    state = UAROETCohortState((1, 0), 1)
    assert state.choose((-2, 1), [20, 40], first_size=3) == 1
    assert state.choose((-2, 1), [20, 40], first_size=3, profile="legacy_1_8") == 0


def test_old_cohort_length_is_corrected_only_in_safe_profile():
    safe = legacy = UAROETCohortState((0, 0))
    for _ in range(4):
        safe = safe.advance(0, new_size=2, old_size=3)
        legacy = legacy.advance(0, new_size=2, old_size=3, profile="legacy_1_8")
    assert safe.old_open and not legacy.old_open
    assert not safe.advance(0, new_size=2, old_size=3).old_open


def test_ties_and_undefined_weights_are_explicit():
    args = ((1, 1), [40, 40], [0.5, 0.5], [0, 0])
    assert uaroet_desktop_assignment(*args).selected == 0
    assert uaroet_desktop_assignment(*args, profile="legacy_1_8").selected == 1
    assert uaroet_desktop_assignment(*args, profile="legacy_1_8", final=True).selected == 0
    result = uaroet_desktop_assignment((1, 1), [0, 0], [0, 0], [4, 4], hybrid=False)
    assert result.selected is None and result.reason == "zero_eligible_weight"
    assert np.isfinite(result.weights).all()


def test_safe_zero_width_schedule_keeps_initial_cutoff():
    args = dict(planned=10, steps=10, observed=0, initial=100, final=0)
    assert uaroet_utility_cutoff(**args) == 100
    assert uaroet_utility_cutoff(**args, profile="legacy_1_8") == 0


@pytest.mark.parametrize(
    "state",
    [
        dict(first_seen=()),
        dict(first_seen=(0, 0), total=1),
        dict(first_seen=(1, 1), total=2),
        dict(first_seen=(0, 0), new_open=True),
        dict(first_seen=(1, 0), total=1, last=1, new_open=True),
    ],
)
def test_invalid_cohort_states_rejected(state):
    with pytest.raises(ValueError):
        UAROETCohortState(**state)
