import numpy as np
import pytest

from mdanderson_stats.tteconduct import (
    tteconduct_boundary_table,
    tteconduct_design,
    tteconduct_monitor,
)


def design():
    return tteconduct_design(60, 295, 3, 10, 1, 0.03, 40, max_total_time=1000)


def test_zero_margin_equal_inverse_gamma_prior_is_exchangeable():
    d = tteconduct_design(3, 10, 3, 10, 0, 0.5, 10, max_total_time=100)
    result = tteconduct_monitor(d, 0, 0, 0)
    assert result.probability == 0.5
    assert result.probability_error == 0


def test_guide_case_shifted_boundary_and_monitor_are_consistent():
    boundary = tteconduct_boundary_table(design(), 3).boundaries[0]
    assert boundary.minimum_total_time == pytest.approx(3.43772033817485, abs=2e-8)
    assert boundary.probability >= 0.03
    assert abs(boundary.residual) <= 2e-9
    monitor = tteconduct_monitor(design(), 3, 3, boundary.minimum_total_time)
    assert monitor.stop_for_futility is False
    assert monitor.probability >= 0.03


def test_boundary_table_default_is_one_row_per_possible_event_count():
    small = tteconduct_design(60, 295, 3, 10, 1, 0.03, 3, max_total_time=1000)
    table = tteconduct_boundary_table(small)
    assert len(table.boundaries) == 3
    assert table.boundaries[0].minimum_total_time == 0
    assert all(row.events == i for i, row in enumerate(table.boundaries, 1))


def test_maximum_patient_stop_and_validation_limits():
    d = design()
    result = tteconduct_monitor(d, 40, 0, 0)
    assert result.stop_for_maximum
    assert result.stop_accrual
    with pytest.raises(ValueError):
        tteconduct_design(1, 1, 1, 1, -1, 0.1, 10, max_total_time=10)
    with pytest.raises(ValueError):
        tteconduct_design(1, 1, 1, 1, 0, 0.1, 1001, max_total_time=10)
    with pytest.raises(ValueError):
        tteconduct_boundary_table(d, np.arange(1, 1002))


def test_beyond_cap_is_explicit():
    d = tteconduct_design(60, 295, 3, 10, 1, 0.999999, 40, max_total_time=1)
    row = tteconduct_boundary_table(d, 3).boundaries[0]
    assert row.beyond_cap
    assert np.isinf(row.minimum_total_time)
