"""Independent arithmetic checks of native calendar defects, not safe defaults."""

import json
from pathlib import Path

import numpy as np
import pytest

DATA = json.loads((Path(__file__).parent / "fixtures/prt-desktop-calendar.json").read_text())


@pytest.mark.parametrize("case", DATA["cases"])
def test_native_asof_boundary_and_daily_index(case):
    window = case["endpoints"][-1]
    result = case["result"]
    for i, (arrival, delay) in enumerate(case["rows"]):
        event = case["clock"] >= arrival + delay and delay <= window
        observed = delay if event else case["clock"] - arrival
        daily = (
            int(observed) if 1 <= observed < window else int(window + 1) if observed > window else 0
        )
        interval = max(0, int(np.searchsorted(case["endpoints"], daily, side="right")) + int(event))
        assert result["observed_time"][i] == observed
        assert result["censored"][i] == int(not event)
        assert result["daily_index"][i] == daily
        assert result["interval"][i] == interval


@pytest.mark.parametrize("case", DATA["final_cases"])
def test_native_final_clock_uses_last_three_patients_only(case):
    window = case["endpoints"][-1]
    expected = max(arrival + min(delay, window) for arrival, delay in case["rows"][-3:])
    assert case["result"]["clock"] == expected
    assert case["after_accrual_clock"] == expected
    assert expected <= max(arrival + min(delay, window) for arrival, delay in case["rows"])


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/prt-desktop-calendar-control.json").read_text())[
        "cases"
    ],
)
def test_native_suspension_skips_one_arrival_and_only_full_cohort_shrinks(case):
    remaining = case["maximum"] - case["enrolled"]
    index = case["index"] + int(remaining > 0)
    assert case["result"]["index"] == index
    assert case["result"]["clock"] == (case["arrivals"][index] if remaining else 0.125)
    assert case["result"]["enrolled"] == case["enrolled"]
    assert case["result"]["cohort"] == (
        remaining if case["cohort"] == 3 and remaining < 3 else case["cohort"]
    )
