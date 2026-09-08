"""Calibration brackets attainable levels rather than interpolating a jump."""

import json
from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import SeqBinDesign, seqbin_calibrate, seqbin_prior


@pytest.mark.parametrize("mean,size", [(0.2, 10), (0.5, 1), (0.9, 100)])
def test_beta_mean_input_and_original_conversion(mean, size):
    a, b = seqbin_prior(mean, size)
    assert_allclose([a / (a + b), a + b], [mean, size])
    legacy = seqbin_prior(mean, size, conversion="legacy")
    assert_allclose(legacy, [mean * (size + 1) - 0.5, size - mean * (size + 1) + 0.5])
    assert_allclose(sum(legacy), size)
    if mean != 0.5:
        assert not np.isclose(legacy[0] / sum(legacy), mean)
    SeqBinDesign(10, prior=legacy)


@pytest.mark.parametrize(
    "mean,size,conversion",
    [
        (0, 10, "exact"),
        (0.5, 0, "exact"),
        (0.1, 4, "legacy"),
        (0.1, 3, "legacy"),
        (0.5, 10, "unknown"),
    ],
)
def test_invalid_prior_conversion(mean, size, conversion):
    with pytest.raises(ValueError):
        seqbin_prior(mean, size, conversion=conversion)


FIXTURE = json.loads((Path(__file__).parent / "fixtures/seqbin_calibration.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_calibration(case):
    result = seqbin_calibrate(
        50,
        0.05,
        prior=[0.5, 0.5],
        null_probability=0.2,
        looks=case["looks"],
        alternative=case["alternative"],
        selection=case["selection"],
        legacy_bounds=True,
    )
    assert_allclose(result.chosen.significance, case["significance"], atol=2e-14, rtol=0)
    assert_allclose(result.lower.significance, case["lower_level"], atol=2e-14, rtol=0)
    assert_allclose(result.upper.significance, case["upper_level"], atol=2e-14, rtol=0)
    # Native root-finder cutoffs have finite tolerances. Compare their resulting
    # discrete design, rather than requiring bitwise equal cutoffs.
    assert_allclose(
        result.chosen.design.tail_probability, case["tail_probability"], atol=2e-8, rtol=0
    )
    assert result.lower.significance <= 0.05 < result.upper.significance


def exact_attainable_levels(alternative):
    # Beta(1,1), p0=1/2, six trials: posterior tails are exact rational
    # binomial sums. Enumerate every cutoff interval and every full outcome path.
    tails = {}
    for n in range(1, 7):
        for k in range(n + 1):
            tails[n, k] = Fraction(sum(comb(n + 1, j) for j in range(k + 1, n + 2)), 2 ** (n + 1))
    breaks = sorted({Fraction(1, 1000), Fraction(999, 1000), *tails.values()})
    cuts = [(a + b) / 2 for a, b in zip(breaks[:-1], breaks[1:], strict=True)]
    levels = set()
    for cutoff in cuts:
        rejects = 0
        for path in product([0, 1], repeat=6):
            for n in range(1, 7):
                cdf = tails[n, sum(path[:n])]
                if (alternative != "less" and cdf < cutoff) or (
                    alternative != "greater" and 1 - cdf < cutoff
                ):
                    rejects += 1
                    break
        levels.add(Fraction(rejects, 64))
    return sorted(levels)


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("selection", ["conservative", "nearest"])
def test_independent_exhaustive_attainable_levels(alternative, selection):
    levels = exact_attainable_levels(alternative)
    for target in [0.05, 0.1, 0.3, 0.6]:
        expected = (
            max(v for v in levels if v <= target)
            if selection == "conservative"
            else min(levels, key=lambda v: (abs(float(v) - target), v))
        )
        result = seqbin_calibrate(
            6,
            target,
            null_probability=0.5,
            alternative=alternative,
            selection=selection,
            tail_bounds=[0.001, 0.999],
        )
        assert_allclose(result.chosen.significance, float(expected), atol=1e-15, rtol=0)
        assert result.lower is not None
        if result.upper is not None:
            lo, hi = result.lower.design.tail_probability, result.upper.design.tail_probability
            assert np.nextafter(lo, 1) == hi
            assert result.lower.significance <= target < result.upper.significance
        assert result.evaluations <= 64


def test_outside_attainable_range_and_nearest_tie():
    with pytest.raises(ValueError, match="No design"):
        seqbin_calibrate(6, 0, tail_bounds=[0.5, 0.9])
    nearest = seqbin_calibrate(6, 0, tail_bounds=[0.5, 0.9], selection="nearest")
    assert nearest.lower is None
    assert nearest.chosen is nearest.upper
    above = seqbin_calibrate(6, 1, tail_bounds=[0.001, 0.002])
    assert above.upper is None and above.chosen is above.lower
    r = seqbin_calibrate(2, 0.2, null_probability=0.5)
    midpoint = (r.lower.significance + r.upper.significance) / 2
    tied = seqbin_calibrate(2, midpoint, null_probability=0.5, selection="nearest")
    assert tied.chosen is tied.lower


@pytest.mark.parametrize(
    "kwargs",
    [
        {"significance": -1},
        {"significance": np.nan},
        {"tail_bounds": [0, 1]},
        {"tail_bounds": [0.2, 0.1]},
        {"selection": "unknown"},
    ],
)
def test_invalid_calibration(kwargs):
    with pytest.raises(ValueError):
        seqbin_calibrate(10, **{"significance": 0.05, **kwargs})


@pytest.mark.parametrize("selection", ["conservative", "nearest"])
def test_separate_calibrations_report_actual_combined_errors(selection):
    from mdanderson_stats import seqbin_calibrate_tails

    result = seqbin_calibrate_tails(30, [0.03, 0.06], selection=selection)
    low = seqbin_calibrate(30, 0.03, alternative="less", selection=selection)
    high = seqbin_calibrate(30, 0.06, alternative="greater", selection=selection)
    assert result.low.chosen.significance == low.chosen.significance
    assert result.high.chosen.significance == high.chosen.significance
    assert result.null_properties.quit_low.sum() <= result.low.chosen.significance + 1e-15
    assert result.null_properties.quit_high.sum() <= result.high.chosen.significance + 1e-15
    assert_allclose(
        result.null_properties.rejection_probability + result.null_properties.complete, 1
    )
    assert_allclose(result.design.continue_low, low.chosen.design.continue_low)
    assert_allclose(result.design.continue_high, high.chosen.design.continue_high)
    if selection == "conservative":
        assert result.null_properties.rejection_probability <= 0.09


def test_invalid_separate_calibration():
    from mdanderson_stats import seqbin_calibrate_tails

    with pytest.raises(ValueError, match="significance"):
        seqbin_calibrate_tails(10, [0.05])


def test_exact_attainable_target_and_subnormal_search_bound():
    result = seqbin_calibrate(
        6,
        1 / 64,
        null_probability=0.5,
        tail_bounds=[np.nextafter(0.0, 1.0), 0.999],
    )
    assert result.chosen.significance == 1 / 64
    assert result.lower is result.chosen
    assert result.upper.significance > 1 / 64
    assert result.evaluations <= 64
