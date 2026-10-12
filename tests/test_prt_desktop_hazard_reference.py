"""Independent arithmetic checks for native hazard calibration and event defects."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betaln

CASES = json.loads((Path(__file__).parent / "fixtures/prt-desktop-hazard.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_beta_shaped_hazard_scale_search_and_grid(case):
    x = np.arange(1, 101) / 101
    density = np.exp(
        (case["alpha"] - 1) * np.log(x)
        + (case["beta"] - 1) * np.log1p(-x)
        - betaln(case["alpha"], case["beta"])
    )
    grid = np.array(case["grid"])
    np.testing.assert_allclose(grid[:99], np.arange(99) * case["window"] / 98, rtol=0, atol=1e-12)
    assert grid[-1] == case["window"]
    np.testing.assert_allclose(density / case["scale"], case["rates"], rtol=1e-10, atol=1e-13)
    mass = float(np.dot(np.diff(grid)[:98], density[:98]))
    scale, step = 1.0, 10.0
    tolerance = float(np.float32(1e-5))
    for i in range(100000):
        probability = -np.expm1(-mass / scale)
        if probability > case["probability"] + tolerance:
            scale += step
        if probability < case["probability"] - tolerance:
            scale -= step
            step /= 10
        if abs(probability - case["probability"]) < tolerance:
            break
    else:
        raise AssertionError("independent calibration exhausted")
    assert scale == pytest.approx(case["scale"], abs=1e-8)
    assert i + 1 == case["iterations"]
    assert abs(-np.expm1(-case["integrated_hazard"]) - case["probability"]) < tolerance


@pytest.mark.parametrize("case", CASES)
def test_native_generation_reuses_first_uniform_and_uses_shifted_rate(case):
    for sample in case["samples"]:
        j = int(np.searchsorted(case["cdf"], sample["first"], side="right")) + 1
        assert j == sample["category"]
        rate = case["rates"][j - 1]
        conditional = (
            1.0 if j == 100 else -np.expm1(-(case["grid"][j - 1] - case["grid"][j - 2]) * rate)
        )
        delay = case["grid"][j - 2] - np.log1p(-sample["first"] * conditional) / rate
        np.testing.assert_allclose(delay, sample["delay"], rtol=1e-11, atol=1e-8)
    a, b = case["samples"][::2], case["samples"][1::2]
    assert all(x["delay"] == y["delay"] for x, y in zip(a, b, strict=True))
