"""Late native simulation rules; inference and earlier screening are excluded."""

import json
from pathlib import Path

import pytest

CASES = json.loads(
    (Path(__file__).parent / "fixtures/prt-desktop-simulation-rules.json").read_text()
)["cases"]


@pytest.mark.parametrize("c", CASES)
def test_native_late_allocation(c):
    k = c["current"]
    xi = c["exceedance"]
    selected, suspend = k, 0
    if xi[k - 1] >= 0.9:
        if k > 1:
            minimum = min(
                [c["distance"][0]] + [c["distance"][i] for i in range(k - 1) if xi[i] < 0.9]
            )
            selected = max(i + 1 for i in range(k - 1) if c["distance"][i] == minimum)
    elif xi[k - 1] >= 0.3 or k == len(xi):
        suspend = int(c["pending"] > 0 and c["pe"] > 0.95)
    elif not c["pending"]:
        if c["next_pe"] <= 0.95:
            selected = k + 1
        else:
            suspend = 1
    elif c["pn"] >= 0.05 and c["next_pe"] <= 0.95:
        selected = k + 1
    else:
        suspend = 1
    if xi[k - 1] < 0.3 and k < len(xi) and xi[k] >= 0.9:
        selected, suspend = k, 0
    assert (c["selected"], c["suspend"], c["stopped"]) == (selected, suspend, int(xi[0] > 0.9))


@pytest.mark.parametrize(
    "c",
    json.loads((Path(__file__).parent / "fixtures/prt-desktop-screening.json").read_text())[
        "cases"
    ],
)
def test_native_screening_precedes_posterior_fit(c):
    a = c["assigned"][c["current"] - 1]
    pending = a - c["assessed"]
    if (
        pending >= 10
        or (a == 3 and c["observed"] == 0)
        or (c["assigned"][0] <= 3 and c["first_interval"] == 0)
    ):
        expected = (False, 1)
    elif a >= 3 and c["observed"] == 0:
        expected = (False, int(c["final"]))
    else:
        expected = (True, 0)
    assert (c["infer"], c["suspend"]) == expected
