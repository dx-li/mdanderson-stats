"""Focused tests for bounded historical-control allocation planning."""

import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.stplan_historical_planning import stplan_historical_allocation_plan
from mdanderson_stats.stplan_survival import stplan_historical_survival_power


def test_manual_allocation_plan_finds_interior_optimum_and_reusable_inputs():
    plan = stplan_historical_allocation_plan(
        0.0192540883488874,
        0.0287927162986077,
        3,
        0,
        10,
        40,
        target_power=0.8,
        accrual_bounds=(1e-4, 1e5),
    )
    assert_allclose(plan.control_allocation, 0.263054176248911, atol=2e-5)
    assert_allclose(plan.accrual_duration, 71.5013800757622, rtol=2e-7)
    assert_allclose(plan.achieved_power, 0.8, atol=1e-8)
    assert_allclose(plan.expected_enrollment, 3 * plan.accrual_duration)
    assert_allclose(stplan_historical_survival_power(**plan.inputs), plan.achieved_power)
    assert not plan.at_accrual_lower_bound


def test_allocation_endpoint_and_no_continued_followup_use_lower_bound():
    endpoint = stplan_historical_allocation_plan(
        0.028881132523331,
        0.0577367205542725,
        5,
        12,
        25,
        25,
        target_power=0.8,
        accrual_bounds=(1e-4, 1e5),
    )
    assert endpoint.control_allocation == 0

    no_followup = stplan_historical_allocation_plan(
        0.05,
        0.1,
        5,
        6,
        40,
        20,
        target_power=0.8,
        accrual_bounds=(1e-4, 1e5),
        continued_followup=False,
    )
    assert no_followup.control_allocation == 0


def test_boundary_adjacent_allocation_peak_and_narrow_feasible_accrual_bracket():
    args = (0.0192540883488874, 0.0287927162986077, 3, 0, 10, 40)
    constrained = stplan_historical_allocation_plan(
        *args,
        target_power=0.8,
        accrual_bounds=(1e-4, 1e5),
        allocation_bounds=(0.25, 0.99),
    )
    assert_allclose(constrained.control_allocation, 0.263054176248911, atol=2e-5)

    narrow = stplan_historical_allocation_plan(
        *args,
        target_power=0.8,
        accrual_bounds=(71.5, 71.502),
    )
    assert_allclose(narrow.accrual_duration, 71.5013800757622, atol=2e-4)


def test_lower_accrual_boundary_is_reported_when_it_already_meets_target():
    plan = stplan_historical_allocation_plan(
        0.05,
        0.1,
        10,
        6,
        1000,
        1000,
        target_power=0.8,
        accrual_bounds=(4, 8),
    )
    assert plan.at_accrual_lower_bound
    assert plan.accrual_duration == 4
    assert plan.achieved_power >= 0.8


def test_infeasible_target_and_invalid_bounds_fail_clearly():
    with pytest.raises(ValueError, match="not attained"):
        stplan_historical_allocation_plan(
            0.09,
            0.1,
            10,
            0,
            10,
            10,
            target_power=0.8,
            accrual_bounds=(1e-4, 1),
        )
    with pytest.raises(ValueError, match="allocation_bounds"):
        stplan_historical_allocation_plan(
            0.05,
            0.1,
            10,
            0,
            10,
            10,
            target_power=0.8,
            accrual_bounds=(1, 100),
            allocation_bounds=(0, 1),
        )
