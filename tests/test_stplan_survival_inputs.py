"""Focused checks for STPLAN survival-input conversions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.stplan_survival_inputs import (
    stplan_exponential_hazard,
    stplan_historical_control_hazard,
    stplan_piecewise_from_survival,
)


def test_exponential_and_historical_hazard_conversions_broadcast():
    assert_allclose(stplan_exponential_hazard([10, 20]), np.log(2) / [10, 20])
    assert_allclose(stplan_exponential_hazard([10, 20], parameter="mean"), [0.1, 0.05])
    assert_allclose(
        stplan_exponential_hazard([0.5, 0.25], parameter="survival", time=10),
        [np.log(2) / 10, np.log(4) / 10],
    )
    assert_allclose(
        stplan_historical_control_hazard([2, 9], [20, 30]),
        [0.1, 0.3],
    )
    smallest = np.nextafter(0.0, 1.0)
    assert_allclose(
        stplan_exponential_hazard(smallest, parameter="survival", time=1), -np.log(smallest)
    )


@pytest.mark.parametrize(
    ("times", "change", "hazards"),
    [
        ([2.0, 8.0], 5.0, (0.1, 0.2)),
        ([4.0, 8.0], 2.0, (0.1, 0.2)),
    ],
)
def test_piecewise_two_point_known_break_reconstructs_both_branches(times, change, hazards):
    before, after = hazards
    cumulative = before * min(times[0], change) + after * max(times[0] - change, 0)
    cumulative2 = before * min(times[1], change) + after * max(times[1] - change, 0)
    result = stplan_piecewise_from_survival(
        times, np.exp(-np.array([cumulative, cumulative2])), change_time=change
    )
    assert_allclose(result.hazard_before, before, rtol=0, atol=2e-15)
    assert_allclose(result.hazard_after, after, rtol=0, atol=2e-15)
    assert_allclose(result.change_time, change)
    assert result.change_time_identified.item()


def test_piecewise_three_point_infers_break_or_reports_exponential_ambiguity():
    times = np.array([2.0, 5.0, 8.0])
    true_break, before, after = 3.5, 0.1, 0.25
    cumulative = before * np.minimum(times, true_break) + after * np.maximum(times - true_break, 0)
    model = stplan_piecewise_from_survival(times, np.exp(-cumulative))
    assert_allclose(model.hazard_before, before, atol=2e-15)
    assert_allclose(model.hazard_after, after, atol=2e-15)
    assert_allclose(model.change_time, true_break, atol=2e-14)
    assert model.change_time_identified.item()

    constant = stplan_piecewise_from_survival(times, np.exp(-0.1 * times))
    assert_allclose(constant.hazard_before, 0.1, atol=2e-15)
    assert_allclose(constant.hazard_after, 0.1, atol=2e-15)
    assert_allclose(constant.change_time, 3.5)
    assert not constant.change_time_identified.item()


def test_piecewise_batches_are_bounded_immutable_and_reject_infeasible_curves():
    times = np.array([2.0, 5.0, 8.0])
    curves = np.stack([np.exp(-0.1 * times), np.exp(-0.2 * times)])
    model = stplan_piecewise_from_survival(times, curves)
    assert model.hazard_before.shape == (2,)
    assert model.change_time_identified.dtype == np.bool_
    with pytest.raises(ValueError):
        model.hazard_before.setflags(write=True)
    with pytest.raises(ValueError, match="nonincreasing"):
        stplan_piecewise_from_survival(times, [0.9, 0.95, 0.7])
    with pytest.raises(ValueError, match="feasible"):
        stplan_piecewise_from_survival(times, [0.8, 0.4, 0.3])
    with pytest.raises(ValueError, match="change_time"):
        stplan_piecewise_from_survival([2, 8], [0.8, 0.4], change_time=8)


def test_known_break_broadcasts_across_batches_and_marks_constant_curve_unidentified():
    times = np.array([2.0, 8.0])
    hazards = np.array([0.1, 0.2])
    breaks = np.array([3.0, 5.0])
    cumulative = hazards[:, None] * np.minimum(times, breaks[:, None]) + 0.2 * np.maximum(
        times - breaks[:, None], 0
    )
    model = stplan_piecewise_from_survival(times, np.exp(-cumulative), change_time=breaks)
    assert_allclose(model.hazard_before, hazards)
    assert_allclose(model.hazard_after, 0.2)
    assert_allclose(model.change_time, breaks)

    constant = stplan_piecewise_from_survival(
        [2.0, 8.0], np.exp(-0.1 * np.array([2.0, 8.0])), change_time=5.0
    )
    assert not constant.change_time_identified.item()
