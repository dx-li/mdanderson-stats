"""Independent native conduct threshold checks; no posterior estimator parity."""

import json
from pathlib import Path

import pytest

CASES = json.loads((Path(__file__).parent / "fixtures/prt-desktop-conduct-rules.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_acceptable_branch_uses_complemented_epsilon(case):
    k = case["current"] - 1
    if case["exceedance"][0] > 0.9:
        expected = 3
    elif case["exceedance"][k] >= 0.9:
        expected = 4
    elif case["assigned"][k] == case["assessed"][k]:
        expected = 5.11
    else:
        expected = 5.1 if case["pe"] <= 1 - 0.05 else 5.2
    assert case["code"] == pytest.approx(expected, abs=5e-7)
