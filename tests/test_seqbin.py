"""Native comparisons, posterior identities and exhaustive stopped sample paths."""

import json
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import SeqBinDesign

CASES = json.loads((Path(__file__).parent / "fixtures/seqbin.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_seqbin_design_and_probabilities(case):
    design = SeqBinDesign(
        20,
        prior=case["prior"],
        alternative=case["alternative"],
        looks=case["looks"],
        legacy_bounds=True,
    )
    assert_array_equal(design.continue_low, case["continue_low"])
    assert_array_equal(design.continue_high, case["continue_high"])
    result = design.operating_characteristics(0.2)
    assert_allclose(result.quit_low, case["quit_low"], atol=2e-15, rtol=2e-13)
    assert_allclose(result.quit_high, case["quit_high"], atol=2e-15, rtol=2e-13)
    assert_allclose(result.rejection_probability, case["rejection"], atol=2e-15)
    assert_allclose(result.expected_subjects, case["expected_subjects"], rtol=2e-14)
    for direction in ["low", "high"]:
        if getattr(result, f"quit_{direction}").sum() > 1e-8:
            assert_allclose(
                getattr(result, f"expected_subjects_quit_{direction}"),
                case[f"expected_{direction}"],
                rtol=2e-13,
            )


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
def test_boundaries_against_integer_beta_binomial_identity(alternative):
    design = SeqBinDesign(
        30, prior=[2, 3], null_probability=0.5, tail_probability=0.125, alternative=alternative
    )
    for n, low, high in zip(design.looks, design.continue_low, design.continue_high, strict=True):
        for k in range(n + 1):
            a, b = 2 + k, 3 + n - k
            # At p0=1/2, every binomial term has a common exact binary denominator.
            cdf = sum(comb(a + b - 1, j) for j in range(a, a + b)) / 2 ** int(a + b - 1)
            assert (k < low) == (alternative != "greater" and 1 - cdf < 0.125)
            assert (k > high) == (alternative != "less" and cdf < 0.125)


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("looks", [[1, 2, 3, 4, 5, 6], [2, 4, 6]])
def test_exact_stopping_paths(alternative, looks):
    design = SeqBinDesign(
        6, null_probability=0.5, tail_probability=0.2, alternative=alternative, looks=looks
    )
    probability = np.array([[0.0, 0.2], [0.8, 1.0]])
    result = design.operating_characteristics(probability)
    low, high = np.zeros_like(result.quit_low), np.zeros_like(result.quit_high)
    complete, expected = np.zeros_like(probability), np.zeros_like(probability)
    for path in product([0, 1], repeat=6):
        k = sum(path)
        mass = probability**k * (1 - probability) ** (6 - k)
        for i, (n, lo, hi) in enumerate(
            zip(design.looks, design.continue_low, design.continue_high, strict=True)
        ):
            count = sum(path[:n])
            if count < lo:
                low[..., i] += mass
                expected += mass * n
                break
            if count > hi:
                high[..., i] += mass
                expected += mass * n
                break
        else:
            complete += mass
            expected += mass * 6
    assert_allclose(result.quit_low, low, atol=1e-15)
    assert_allclose(result.quit_high, high, atol=1e-15)
    assert_allclose(result.complete, complete, atol=1e-15)
    assert_allclose(result.expected_subjects, expected, atol=1e-14)
    assert_allclose(result.rejection_probability + result.complete, 1, atol=1e-14)


def test_strict_threshold_equality_continues():
    high = SeqBinDesign(2, null_probability=0.5, tail_probability=0.25)
    assert_array_equal(high.continue_high, [1, 1])
    low = SeqBinDesign(2, null_probability=0.5, tail_probability=0.25, alternative="less")
    assert_array_equal(low.continue_low, [0, 1])


def test_strong_prior_all_outcomes_stop_and_legacy_clamp():
    design = SeqBinDesign(10, prior=[30, 1])
    assert design.continue_high[0] == -1
    result = design.operating_characteristics([0, 0.2, 1])
    assert_array_equal(result.quit_high[:, 0], [1, 1, 1])
    assert_array_equal(result.expected_subjects, [1, 1, 1])
    assert_array_equal(result.complete, [0, 0, 0])
    assert np.all(np.isnan(result.expected_subjects_quit_low))
    original = SeqBinDesign(10, prior=[30, 1], legacy_bounds=True)
    assert original.continue_high[0] == 0
    assert original.operating_characteristics(0).rejection_probability == 0


def test_look_snapshot_and_maximum_size():
    looks = np.array([5, 10])
    design = SeqBinDesign(10, looks=looks)
    looks[0] = 1
    assert_array_equal(design.looks, [5, 10])
    with pytest.raises(ValueError):
        design.continue_high[0] = 0
    large = SeqBinDesign(10000)
    assert large.looks.size == 10000
    assert np.all(np.diff(large.continue_high) >= 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_subjects": 1},
        {"max_subjects": True},
        {"prior": [0, 1]},
        {"prior": [1]},
        {"null_probability": 0},
        {"tail_probability": 1},
        {"tail_probability": np.nan},
        {"looks": [5, 4, 10]},
        {"looks": [2, 5]},
        {"looks": []},
        {"looks": [1.5, 10]},
        {"alternative": "invalid"},
        {"legacy_bounds": 1},
    ],
)
def test_invalid_design(kwargs):
    arguments = {"max_subjects": 10, **kwargs}
    with pytest.raises(ValueError):
        SeqBinDesign(**arguments)


def test_invalid_probability():
    with pytest.raises(ValueError, match="probability"):
        SeqBinDesign(10).operating_characteristics([0.2, np.nan])


def test_rare_rejection_keeps_its_actual_conditional_sample_size():
    result = SeqBinDesign(20).operating_characteristics(1e-12)
    assert 0 < result.rejection_probability < 1e-8
    assert 1 <= result.expected_subjects_quit_high < 2


def test_low_strong_prior_all_outcomes_stop():
    design = SeqBinDesign(10, prior=[1, 100], alternative="less")
    assert design.continue_low[0] == 2
    result = design.operating_characteristics([0, 0.2, 1])
    assert_array_equal(result.quit_low[:, 0], [1, 1, 1])
    assert_array_equal(result.expected_subjects, [1, 1, 1])


def test_separate_tail_cutoffs_and_overlap_precedence():
    design = SeqBinDesign(
        6, null_probability=0.5, alternative="two-sided", tail_probability=[0.8, 0.7]
    )
    result = design.operating_characteristics(0.5)
    # At n=1, Beta(1,2) has upper tail .25, Beta(2,1) has upper
    # tail .75. Both are below .8 and therefore stop on the low side,
    # including the k=1 outcome that also satisfies the high-side rule.
    assert result.quit_low[0] == 1
    assert result.quit_high.sum() == 0
    assert result.expected_subjects == 1
    assert result.complete == 0
    asymmetric = SeqBinDesign(20, alternative="two-sided", tail_probability=[0.02, 0.1])
    low = SeqBinDesign(20, alternative="less", tail_probability=0.02)
    high = SeqBinDesign(20, alternative="greater", tail_probability=0.1)
    assert_array_equal(asymmetric.continue_low, low.continue_low)
    assert_array_equal(asymmetric.continue_high, high.continue_high)
