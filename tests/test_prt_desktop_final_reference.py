"""Native final choice includes eligibility, initialization and last-tie quirks."""

import json
from pathlib import Path

import pytest

CASES = json.loads((Path(__file__).parent / "fixtures/prt-desktop-final.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_final_selection_eligibility_and_fallback(case):
    eligible = [
        i
        for i, (n, xi) in enumerate(zip(case["assessed"], case["exceedance"], strict=True))
        if n >= 3 and xi < 0.9
    ]
    # Native initialization includes dose1 even when it is ineligible.
    minimum = min([case["distance"][0]] + [case["distance"][i] for i in eligible if i > 0])
    selected = case["current"]
    stopped = case["exceedance"][0] > 0.9
    if not stopped:
        for i in eligible:
            if case["distance"][i] == minimum:
                selected = i + 1
    assert case["selected"] == selected
    assert case["stopped"] == int(stopped)


def test_ineligible_initial_minimum_can_leave_current_dose_unchanged():
    c = next(
        c
        for c in CASES
        if c["assessed"] == [2, 3, 4]
        and c["exceedance"] == [0.1, 0.5, 0.9]
        and c["distance"] == [0.1, 0.2, 0.3]
        and c["current"] == 3
    )
    assert c["selected"] == 3  # Dose3 is itself ineligible at the exact cutoff.
