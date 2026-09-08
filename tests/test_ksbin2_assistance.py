"""Exhaustive path validation of stage assistance and reference power loss."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    KStageTwoSampleBinomial,
    ksbin2_boundary_table,
    ksbin2_probability_table,
)


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("pair", [(0.6, 0.2), (0, 1), (0.5, 0.5)])
def test_exhaustive_paths(alternative, stage, pair):
    d = KStageTwoSampleBinomial(
        [[2, 2], [3, 3], [4, 4]], [0, 0, 1], [2, 2], criteria=(1,), alternative=alternative
    )
    ref = ksbin2_probability_table(4, 4, 0.6, 0.2, criteria=(1,)).select_group(2)
    grid = np.array([0, 0.3, 0.5, 1])
    t = ksbin2_boundary_table(d, stage, *pair, null_grid=grid, reference=ref)
    groups = len(t.ordering.group_end)
    expected = np.zeros((5, groups))
    previous = np.zeros(5)
    lost = np.zeros(groups)
    reference_points = set(map(tuple, ref.events))
    for first in itertools.product([0, 1], repeat=4):
        for second in itertools.product([0, 1], repeat=4):
            k1, k2 = sum(first), sum(second)
            weights = np.array(
                [
                    a**k1 * (1 - a) ** (4 - k1) * b**k2 * (1 - b) ** (4 - k2)
                    for a, b in [*zip(grid, grid), pair]
                ]
            )
            stopped = False
            for i in range(stage - 1):
                n = i + 2
                point = (sum(first[:n]), sum(second[:n]))
                ordering = d.orderings[i]
                row = next(j for j, p in enumerate(ordering.events) if tuple(p) == point)
                group = np.searchsorted(ordering.group_end, row)
                if group <= d.reject_group[i]:
                    previous += weights
                    stopped = True
                    break
                if group >= d.quit_group[i]:
                    stopped = True
                    break
            if stopped:
                continue
            n = stage + 1
            point = (sum(first[:n]), sum(second[:n]))
            row = next(j for j, p in enumerate(t.ordering.events) if tuple(p) == point)
            group = np.searchsorted(t.ordering.group_end, row)
            expected[:, group:] += weights[:, None]
            if (k1, k2) in reference_points:
                lost[: group + 1] += weights[-1]
    assert_allclose(t.stage_null_rejection, expected[:4], atol=4e-14)
    assert_allclose(t.stage_power, expected[-1], atol=4e-14)
    assert_allclose(t.previous_null_rejection, previous[:4], atol=4e-14)
    assert_allclose(t.previous_power, previous[-1], atol=4e-14)
    assert_allclose(t.cumulative_power, expected[-1] + previous[-1], atol=4e-14)
    assert_allclose(t.cumulative_null_rejection, expected[:4] + previous[:4, None], atol=4e-14)
    assert_allclose(t.significance, (expected[:4] + previous[:4, None]).max(axis=0), atol=4e-14)
    assert_allclose(t.power_loss, lost, atol=4e-14)
    strict = np.concatenate([np.zeros((4, 1)), expected[:4, :-1]], axis=1)
    pointwise = previous[:4, None] + (strict + expected[:4]) / 2
    assert_allclose(t.null_midp, pointwise, atol=4e-14)
    assert_allclose(t.pointwise_midp_significance, pointwise.max(axis=0), atol=4e-14)


def test_first_stage_matches_native_validated_probability_table():
    d = KStageTwoSampleBinomial([[4, 5], [7, 8]], [-1, -1], [-1])
    t = ksbin2_boundary_table(d, 1, 0.6, 0.2)
    single = ksbin2_probability_table(4, 5, 0.6, 0.2)
    assert_allclose(t.cumulative_power, single.power)
    assert_allclose(t.significance, single.significance)
    assert t.power_loss is None
    assert t.conditional_reference_power is None


def test_empty_full_reference_and_broadcast_completion():
    d = KStageTwoSampleBinomial([[2, 2], [3, 2], [3, 4]], [-1, -1, -1], [-1, -1])
    ref = ksbin2_probability_table(3, 4, 0.6, 0.2)
    p1 = np.array([[0], [0.6], [1]])
    p2 = np.array([0, 0.2, 1])
    for group in [-1, len(ref.significance) - 1]:
        t = ksbin2_boundary_table(d, 2, p1, p2, reference=ref.select_group(group))
        assert t.cumulative_power.shape[:2] == (3, 3)
        assert t.conditional_reference_power.shape[:2] == (3, 3)
        assert_allclose(t.conditional_reference_power, 0 if group == -1 else 1, atol=3e-14)
        assert_allclose(t.power_loss[..., 0], 0 if group == -1 else 1, atol=3e-14)


def test_known_conditional_completion_at_cutoff():
    d = KStageTwoSampleBinomial([[1, 1], [2, 2]], [-1, -1], [-1], criteria=(1,))
    reference = ksbin2_probability_table(2, 2, 0.6, 0.2, criteria=(1,)).select_group(0)
    t = ksbin2_boundary_table(d, 1, 0.6, 0.2, reference=reference)
    row = np.flatnonzero(np.all(t.ordering.events == [1, 0], axis=1))[0]
    assert_allclose(t.conditional_reference_power[row], 0.6 * 0.8)
    assert_allclose(t.power_loss[0], (0.6 * 0.8) ** 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": 0},
        {"stage": True},
        {"probability1": np.nan},
        {"null_grid": []},
        {"null_grid": [0.3, 0.2]},
        {"null_grid": [0.5, 0.5]},
        {"null_grid": [-0.1]},
        {"reference": "invalid"},
        {"design": None},
    ],
)
def test_invalid_arguments(changes):
    args = dict(
        design=KStageTwoSampleBinomial([[2, 2]], [-1], []),
        stage=1,
        probability1=0.6,
        probability2=0.2,
    )
    args.update(changes)
    with pytest.raises(ValueError):
        ksbin2_boundary_table(**args)


def test_wrong_reference_size():
    with pytest.raises(ValueError):
        ksbin2_boundary_table(
            KStageTwoSampleBinomial([[2, 2]], [-1], []),
            1,
            0.6,
            0.2,
            reference=ksbin2_probability_table(3, 3, 0.6, 0.2).select(0.05),
        )


NATIVE_MIDP = json.loads((Path(__file__).parent / "fixtures/ksbin2_midp.json").read_text())["cases"]


@pytest.mark.parametrize("case", NATIVE_MIDP)
def test_native_multistage_midp_display(case):
    design = KStageTwoSampleBinomial(
        [[2, 2], [3, 3], [4, 4]], [0, 0, 1], [2, 2], criteria=(1,), alternative=case["alternative"]
    )
    table = ksbin2_boundary_table(design, case["stage"], 0.6, 0.2)
    assert_allclose(table.significance, case["input"], atol=2e-14)
    assert_allclose(table.midp_significance, case["midp"], atol=2e-14)


def test_first_group_preserves_prior_rejections_only_in_pointwise_method():
    d = KStageTwoSampleBinomial([[2, 2], [3, 3]], [0, -1], [2], criteria=(1,))
    t = ksbin2_boundary_table(d, 2, 0.6, 0.2, null_grid=[0.5])
    # Prior rejection (2,0) has probability 1/16. New first-group rejection
    # arrives via (1,0) or (2,1), then a success/failure increment: 1/16.
    assert_allclose(t.previous_null_rejection, [1 / 16])
    assert_allclose(t.stage_null_rejection[0, 0], 1 / 16)
    assert_allclose(t.midp_significance[0], 1 / 16)
    assert_allclose(t.null_midp[0, 0], 3 / 32)
    assert_allclose(t.significance[0], 1 / 8)


def test_multistage_midp_does_not_change_power_or_inclusive_regions():
    d = KStageTwoSampleBinomial([[2, 2], [3, 3]], [0, -1], [2])
    t = ksbin2_boundary_table(d, 2, [[0.5], [0.6]], [0.1, 0.2])
    before = t.cumulative_power.copy()
    _ = t.midp_significance, t.null_midp, t.pointwise_midp_significance
    assert_allclose(t.cumulative_power, before)
