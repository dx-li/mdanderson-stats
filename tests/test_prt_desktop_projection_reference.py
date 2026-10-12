"""Recovered native projection safeguards and ordering differ from the paper API."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import prt_isotonic_projection

CASES = json.loads((Path(__file__).parent / "fixtures/prt-desktop-projection.json").read_text())[
    "cases"
]


def transforms(x):
    D = x.shape[1]
    cov = np.cov(x, rowvar=False)
    means = {}
    for lo in range(D):
        for hi in range(lo, D):
            weights = np.linalg.solve(cov[lo : hi + 1, lo : hi + 1], np.ones(hi - lo + 1))
            means[lo, hi] = x[:, lo : hi + 1] @ (weights / weights.sum())
    native = np.column_stack(
        [
            np.maximum.reduce(
                [np.minimum.reduce([means[lo, hi] for hi in range(d, D)]) for lo in range(d + 1)]
            )
            for d in range(D)
        ]
    )
    paper = np.column_stack(
        [
            np.minimum.reduce(
                [np.maximum.reduce([means[lo, hi] for lo in range(d + 1)]) for hi in range(d, D)]
            )
            for d in range(D)
        ]
    )
    return native, paper


@pytest.mark.parametrize("case", CASES)
def test_native_executes_max_min_with_lower_floor_only(case):
    raw, _ = transforms(np.asarray(case["input"]))
    np.testing.assert_allclose(np.maximum(0, raw), case["result"], rtol=0, atol=1e-11)


def test_native_lower_safeguard_does_not_establish_valid_probabilities():
    lower = [np.asarray(c["result"]) for c in CASES if c["kind"] == "negative"]
    upper = [np.asarray(c["result"]) for c in CASES if c["kind"] == "above_one"]
    assert all(np.min(x) == 0 for x in lower)
    assert all(np.max(x) > 1.005 for x in upper)
    for c in CASES:
        if c["kind"] in ("negative", "above_one"):
            with pytest.raises(ArithmeticError, match=r"leaves \[0,1\]"):
                prt_isotonic_projection(np.asarray(c["input"])[:, None, :])


def test_published_min_max_is_not_silently_replaced_by_native_order():
    c = next(c for c in CASES if c["kind"] == "independent" and len(c["input"][0]) == 10)
    x = np.asarray(c["input"])
    native, paper = transforms(x)
    assert np.max(np.abs(native - paper)) > 0.049
    np.testing.assert_allclose(
        prt_isotonic_projection(x[:, None, :]).probability[:, 0], paper, rtol=0, atol=1e-12
    )
