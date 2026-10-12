"""Complete original desktop calendar traces and explicit legacy time defects."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.boin_desktop_tite_calendar import replay_boin_desktop_tite

ROOT = Path(__file__).parents[1] / "research/fixtures/boin-desktop"
ROWS = [
    r
    for file in ("tite-calendar.csv", "tite-calendar-extended.csv")
    for r in csv.DictReader((ROOT / file).read_text().splitlines())
]


def inputs(row):
    target = float(row["target"])
    probability = {
        "gradient": [0.1, target, 0.6],
        "all": [0.95] * 3,
        "none": [0.01] * 3,
        "latent": [0.55] * 3,
    }[row["truth"]]
    size = int(row.get("cohort_size", 3))
    quantiles = np.tile(np.resize([0.1, 0.5, 0.9], size), 6)
    potential = np.where(quantiles[:, None] < probability, float(row["delay"]), np.inf)
    design = BOINDesign(
        target=target,
        extra_safe=row["extra_safe"] == "TRUE",
        early_stop_patients=int(row["early"]),
        stay_at_one_of_three=row["runin"] == "TRUE" and target == 0.25,
    )
    kwargs = dict(
        cohort_size=size,
        max_pending_fraction=float(row.get("max_pending", 0.5)),
        trimester_probabilities=None
        if row.get("prior", "uniform") == "uniform"
        else [0.1, 0.2, 0.7],
    )
    return design, np.full(10000, float(row["gap"])), potential, kwargs


@pytest.mark.parametrize("row", ROWS)
def test_original_calendar_trace(row):
    design, gaps, potential, kwargs = inputs(row)
    result = replay_boin_desktop_tite(
        design, gaps, potential, 1, profile="desktop_legacy", **kwargs
    )
    for field, actual in (
        ("patients", result.patients),
        ("toxicities", result.toxicities),
        ("doses", result.doses),
        ("entry", result.enrollment_times),
    ):
        np.testing.assert_allclose(actual, np.fromstring(row[field], sep=";"), atol=1e-12)
    assert abs(result.final_time - float(row["duration"])) < 1e-12
    assert result.extra_checks == int(row["waits"])
    selected = np.fromstring(row["selection"], sep=";")
    expected = int(np.argmax(selected)) + 1 if np.any(selected) else None
    assert result.selected_dose == expected
    assert not result.patients.flags.writeable


def test_safe_calendar_removes_future_outcome_stop():
    row = next(
        r
        for r in ROWS
        if r["truth"] == "latent"
        and r["early"] == "6"
        and r["gap"] == "0.1"
        and r["delay"] == "0.9"
        and r["extra_safe"] == "FALSE"
    )
    design, gaps, potential, kwargs = inputs(row)
    legacy = replay_boin_desktop_tite(
        design, gaps, potential, 1, profile="desktop_legacy", **kwargs
    )
    safe = replay_boin_desktop_tite(design, gaps, potential, 1, **kwargs)
    assert legacy.used_future_stop and legacy.stop_reason == "legacy_future_outcome_stop"
    assert not safe.used_future_stop and safe.stop_reason == "stop_safety"
    assert safe.final_time >= safe.decision_times.max()


def test_safe_time_cannot_precede_actual_decision():
    design = BOINDesign(target=0.25, early_stop_patients=12)
    potential = np.full((6, 3), 0.2)
    legacy = replay_boin_desktop_tite(
        design, np.full(20, 2.0), potential, 1, profile="desktop_legacy"
    )
    safe = replay_boin_desktop_tite(design, np.full(20, 2.0), potential, 1)
    assert legacy.final_time == 5 and legacy.decision_times.max() == 6
    assert safe.final_time == 6


def test_tape_and_input_budgets_are_explicit():
    design = BOINDesign(target=0.25, early_stop_patients=12)
    with pytest.raises(ValueError, match="exhausted"):
        replay_boin_desktop_tite(design, [0.1], np.full((6, 3), np.inf), 1)
    with pytest.raises(ValueError, match="booleans"):
        replay_boin_desktop_tite(design, [0.1], np.full((6, 3), np.inf), 1, cohort_size=True)
    with pytest.raises(ValueError, match="invalid bounded"):
        replay_boin_desktop_tite(design, [-0.1], np.full((6, 3), np.inf), 1)
