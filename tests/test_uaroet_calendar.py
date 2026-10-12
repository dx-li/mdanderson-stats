"""Native delay arithmetic and exact observation-time boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import UAROETDesktopStudy, read_uaroet_desktop_study
from mdanderson_stats.uaroet_calendar import (
    UAROETTimedPatient,
    uaroet_outcome_delays,
    uaroet_study_as_of,
)

ROOT = Path(__file__).parent / "fixtures"


def design():
    return replace(
        read_uaroet_desktop_study((ROOT / "uaroet-python-study.json").read_text()), patients=()
    )


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "uaroet-desktop-calendar.json").read_text())["cases"]
)
def test_native_rounded_uniform_delay(case):
    result = uaroet_outcome_delays(
        case["efficacy_window"], case["toxicity_window"], case["uniforms"], profile="legacy_1_8"
    )
    assert list(result) == case["result"]


def test_safe_uniform_law_does_not_round_positive_delays_to_zero():
    assert uaroet_outcome_delays([0.1, 0.2], [0.1, 0.2], [0.5, 0.5]) == pytest.approx((0.15, 0.15))
    assert uaroet_outcome_delays([0.1, 0.2], [0.1, 0.2], [0.5, 0.5], profile="legacy_1_8") == (0, 0)


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "uaroet-desktop-likelihood.json").read_text())["cases"]
)
def test_native_as_of_likelihood_inclusion(case):
    j = np.asarray(case["joint"])
    D = max(2, len(j))
    E, T = j.shape[1:]
    base = design()
    s = UAROETDesktopStudy(
        D,
        E,
        T,
        np.zeros(D * (E + T - 2)),
        np.ones(D * (E + T - 2)),
        50 * (np.arange(E)[:, None] / (E - 1) + 1 - np.arange(T)[None, :] / (T - 1)),
        base.trial_parameters,
        base.simulation_parameters,
    )
    rows = sorted(case["patients"], key=lambda r: r[3])
    timed = tuple(UAROETTimedPatient(i + 1, *row) for i, row in enumerate(rows))
    view = uaroet_study_as_of(s, timed, case["clock"], profile="legacy_1_8")
    counts = view.counts()
    used = counts.likelihood.sum() if case["partial"] else counts.complete.sum()
    assert used == case["result"]["used"]


def test_as_of_keeps_equality_and_pending_assignment_bookkeeping():
    patients = (UAROETTimedPatient(1, 0, 1, 0, 0, 2, 1), UAROETTimedPatient(2, 1, 0, 1, 3, 2, 1))
    s = design()
    assert uaroet_study_as_of(s, patients, 0).patients == ((1, 0, -1, -1),)
    assert uaroet_study_as_of(s, patients, 1).patients == ((1, 0, -1, 0),)
    assert uaroet_study_as_of(s, patients, 2).patients == ((1, 0, 1, 0),)
    assert uaroet_study_as_of(s, patients, 3).patients == ((1, 0, 1, 0), (2, 1, -1, -1))
    assert uaroet_study_as_of(s, patients, 5).patients == ((1, 0, 1, 0), (2, 1, 0, 1))


def test_as_of_never_silently_discards_existing_conduct_data():
    s = replace(design(), patients=((1, 0, 1, 0),))
    with pytest.raises(ValueError, match="without patient rows"):
        uaroet_study_as_of(s, (), 0)


@pytest.mark.parametrize(
    "rows",
    [
        (UAROETTimedPatient(2, 0, 1, 0, 0, 1, 1),),
        (UAROETTimedPatient(1, 0, 1, 0, 2, 1, 1), UAROETTimedPatient(2, 0, 1, 0, 1, 1, 1)),
        (UAROETTimedPatient(1, 2, 1, 0, 0, 1, 1),),
        (UAROETTimedPatient(1, 0, 2, 0, 0, 1, 1),),
    ],
)
def test_invalid_timed_history_rejected(rows):
    with pytest.raises(ValueError):
        uaroet_study_as_of(design(), rows, 10)


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "uaroet-desktop-calendar.json").read_text())["arrival_cases"]
)
def test_native_suspension_discards_arrivals_and_includes_deadline_equality(case):
    from mdanderson_stats.uaroet_calendar import _advance_arrival

    values = iter(case["intervals"])
    used = []

    def draw():
        value = next(values)
        used.append(value)
        return value

    clock, discarded = _advance_arrival(
        case["previous"], case["wait_until"] if case["enabled"] else 0, draw, 1000
    )
    assert clock == case["result"]["clock"]
    assert discarded == case["result"]["discarded"]
    assert len(used) == case["result"]["variates_used"]


def test_arrival_budget_cannot_hang_on_zero_intervals():
    from mdanderson_stats.uaroet_calendar import _advance_arrival

    with pytest.raises(RuntimeError, match="budget"):
        _advance_arrival(0, 1, lambda: 0.0, 3)
