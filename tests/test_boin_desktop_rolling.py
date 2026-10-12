"""Original desktop Rolling Six traces and observed-safety compatibility defects."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop_rolling import replay_boin_desktop_rolling_six

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/rolling-calendar.csv")
        .read_text()
        .splitlines()
    )
)


def inputs(row):
    p = {
        "gradient": [0.05, 0.25, 0.6],
        "all": [0.95] * 3,
        "none": [0.01] * 3,
        "latent": [0.55] * 3,
        "adverse": [0.01, 0.55, 0.8],
    }[row["truth"]]
    potential = np.where(np.resize([0.1, 0.5, 0.9], 18)[:, None] < p, float(row["delay"]), np.inf)
    return np.full(10000, float(row["gap"])), potential


@pytest.mark.parametrize("row", ROWS)
def test_original_calendar(row):
    gaps, potential = inputs(row)
    if row["error"]:
        assert row["error"] == "reference patient budget exceeded"
        with pytest.raises(ValueError, match="exceed six"):
            replay_boin_desktop_rolling_six(
                gaps,
                potential,
                1,
                cohort_size=int(row["size"]),
                start_dose=int(row["start"]),
                profile="desktop_legacy",
            )
        return
    result = replay_boin_desktop_rolling_six(
        gaps,
        potential,
        1,
        cohort_size=int(row["size"]),
        start_dose=int(row["start"]),
        profile="desktop_legacy",
    )
    for field, actual in (
        ("patients", result.patients),
        ("toxicities", result.toxicities),
        ("doses", result.doses),
        ("entry", result.enrollment_times),
    ):
        np.testing.assert_allclose(actual, np.fromstring(row[field], sep=";"), atol=1e-12)
    assert abs(result.reported_final_time - float(row["duration"])) < 1e-12
    assert abs(result.decision_times[-1] - float(row["decision"])) < 1e-12
    selected = np.fromstring(row["selection"], sep=";")
    assert result.selected_dose == (int(np.argmax(selected)) + 1 if selected.any() else None)


def test_safe_profile_blocks_reescalation_into_known_toxic_dose():
    observed = 0
    for row in ROWS:
        if row["truth"] != "adverse":
            continue
        gaps, potential = inputs(row)
        legacy = None
        if not row["error"]:
            legacy = replay_boin_desktop_rolling_six(
                gaps,
                potential,
                1,
                cohort_size=int(row["size"]),
                start_dose=int(row["start"]),
                profile="desktop_legacy",
            )
        safe = replay_boin_desktop_rolling_six(
            gaps, potential, 1, cohort_size=int(row["size"]), start_dose=int(row["start"])
        )
        assert not safe.unsafe_assignments.any() and not safe.selected_eliminated
        if legacy is not None:
            observed += int(legacy.unsafe_assignments.any())
        assert safe.reported_final_time == safe.actual_final_time
    assert observed > 0


def test_reported_legacy_time_is_separate_from_chronological_time():
    result = replay_boin_desktop_rolling_six(
        np.full(100, 2.0), np.full((18, 3), np.inf), 1, cohort_size=3, profile="desktop_legacy"
    )
    assert result.reported_final_time < result.actual_final_time
    assert result.actual_final_time >= result.decision_times[-1]


def test_invalid_capacity_and_arrival_tape_are_rejected():
    with pytest.raises(ValueError, match="bounded"):
        replay_boin_desktop_rolling_six([0.1], np.full((18, 3), np.inf), 1, cohort_size=4)
    with pytest.raises(ValueError, match="exhausted"):
        replay_boin_desktop_rolling_six([0.1], np.full((18, 3), np.inf), 1)
