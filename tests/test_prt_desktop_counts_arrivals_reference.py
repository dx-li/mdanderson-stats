"""Independent arithmetic checks for recovered input counts and arrival process."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "prt-desktop-counts.json").read_text())["cases"]
)
def test_native_file_censor_indicator_and_assessment_counts(case):
    D, C = case["doses"], case["intervals"]
    assigned, assessed, events = (np.zeros(D) for _ in range(3))
    for dose, interval, censored in case["rows"]:
        assigned[dose - 1] += 1
        if (not censored and 1 <= interval <= C) or (censored and interval == C):
            assessed[dose - 1] += 1
        if not censored and 1 <= interval <= C:
            events[dose - 1] += 1
    for name, value in zip(
        ("assigned", "assessed", "events"), (assigned, assessed, events), strict=True
    ):
        np.testing.assert_array_equal(value, case["result"][name])


@pytest.mark.parametrize(
    "case", json.loads((ROOT / "prt-desktop-arrivals.json").read_text())["cases"]
)
def test_native_arrivals_are_sorted_finite_uniform_calendar_points(case):
    expected = np.sort(np.array(case["uniforms"]) * 800 * case["mean"])
    np.testing.assert_allclose(expected, case["arrivals"], rtol=2e-16, atol=1e-11)
    assert np.all(np.diff(case["arrivals"]) >= 0)
    assert len(case["arrivals"]) == 800
