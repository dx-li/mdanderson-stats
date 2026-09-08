"""Design assistance checked against complete Bernoulli trial paths."""

import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KStageBinomial, ksbin1_boundary_table


@pytest.mark.parametrize("alternative", ["less", "greater"])
@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("cutoff", [0, 2, 6])
@pytest.mark.parametrize("probabilities", [(0.7, 0.2), (1, 0), (0.6, 0.5)])
def test_all_trial_paths(alternative, stage, cutoff, probabilities):
    p0, pa = probabilities if alternative == "less" else probabilities[::-1]
    design = KStageBinomial([2, 4, 6], [0, 1], [2, 3])
    result = ksbin1_boundary_table(design, stage, p0, pa, cutoff, alternative=alternative)
    n = design.cumulative_trials[stage - 1]
    sig, power, loss, contribution = np.zeros((4, n + 1))
    reference = np.zeros(2)
    for sequence in itertools.product([0, 1], repeat=6):
        final = sum(sequence)
        weights = np.array(
            [p0**final * (1 - p0) ** (6 - final), pa**final * (1 - pa) ** (6 - final)]
        )
        rejects = final <= cutoff if alternative == "less" else final >= cutoff
        if rejects:
            reference += weights
        if any(
            sum(sequence[:total]) <= low or sum(sequence[:total]) >= high
            for total, low, high in zip(
                design.cumulative_trials[: stage - 1], design.low, design.high, strict=False
            )
        ):
            continue
        k = sum(sequence[:n])
        for boundary in range(n + 1):
            reject = k <= boundary if alternative == "less" else k >= boundary
            quit = k >= boundary if alternative == "less" else k <= boundary
            if reject:
                sig[boundary] += weights[0]
                power[boundary] += weights[1]
            if quit and rejects:
                loss[boundary] += weights[1]
        if rejects:
            contribution[k] += weights[1]
    assert_allclose(result.significance, sig, atol=2e-14)
    assert_allclose(result.power, power, atol=2e-14)
    assert_allclose(result.power_loss, loss, atol=2e-14)
    assert_allclose(result.power_contribution, contribution, atol=2e-14)
    assert_allclose(
        [result.single_stage_significance, result.single_stage_power], reference, atol=2e-14
    )


def test_inclusive_lower_cutoff_regression():
    r = ksbin1_boundary_table(KStageBinomial([2, 4], [-1], [-1]), 1, 0.5, 0.2, 1)
    # The source incorrectly assigns zero to paths with current count == cutoff.
    assert_allclose(r.conditional_reference_power[1], 0.8**2)
    assert_allclose(r.power_loss[1], 2 * 0.2 * 0.8**3)


def test_broadcast_and_disabled_boundaries():
    design = KStageBinomial([2, 4, 6], [-1, -1], [-1, -1])
    r = ksbin1_boundary_table(design, 2, [[0.5], [0.7]], [0.1, 0.2, 0.3], 2)
    assert r.power.shape == (2, 3, 5)
    assert_allclose(r.power_loss[..., 0], r.single_stage_power, atol=1e-14)
    assert_allclose(r.significance[..., -1], 1, atol=1e-14)
    for i, p0 in enumerate([0.5, 0.7]):
        for j, pa in enumerate([0.1, 0.2, 0.3]):
            scalar = ksbin1_boundary_table(design, 2, p0, pa, 2)
            assert_allclose(r.power_loss[i, j], scalar.power_loss)


def test_remaining_trials_above_legacy_allocation_limit():
    r = ksbin1_boundary_table(KStageBinomial([1, 200], [-1], [-1]), 1, 0.2, 0.01, 0)
    assert_allclose(r.power_loss[0], 0.99**200, rtol=1e-13)
    assert r.power_loss[1] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"design": None},
        {"stage": 0},
        {"stage": 4},
        {"stage": 1.5},
        {"stage": True},
        {"null_probability": np.nan},
        {"alternative_probability": -0.1},
        {"null_probability": 0.1},
        {"alternative_probability": 0.5},
        {"single_stage_critical": -1},
        {"single_stage_critical": 7},
        {"single_stage_critical": 1.5},
        {"single_stage_critical": [1]},
        {"alternative": "two-sided"},
        {"alternative": "greater"},
    ],
)
def test_invalid_inputs(changes):
    kwargs = dict(
        design=KStageBinomial([2, 4, 6], [-1, -1], [-1, -1]),
        stage=1,
        null_probability=0.5,
        alternative_probability=0.2,
        single_stage_critical=2,
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        ksbin1_boundary_table(**kwargs)


NATIVE = json.loads((Path(__file__).parent / "fixtures/ksbin1_table.json").read_text())["cases"]


@pytest.mark.parametrize("case", NATIVE)
def test_native_table_with_documented_inclusive_correction(case):
    c = case
    p = c["probability"]
    p0 = (1 + p) / 2 if c["alternative"] == "less" else p / 2
    r = ksbin1_boundary_table(
        KStageBinomial([c["trials"], c["total"]], [-1], [-1]),
        1,
        p0,
        p,
        c["cutoff"],
        alternative=c["alternative"],
    )
    original = np.array(c["power_loss"])
    if c["alternative"] == "less":
        # Source excludes exactly cutoff events followed by zero further events.
        k = c["cutoff"]
        correction = comb(c["trials"], k) * p**k * (1 - p) ** (c["total"] - k)
        original[: k + 1] += correction
    assert_allclose(r.power_loss, original, rtol=2e-12, atol=3e-15)
    assert_allclose(r.design.stage_distribution(1, p), c["arrival"], rtol=1e-13)
