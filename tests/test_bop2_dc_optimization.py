import numpy as np
import pytest

from mdanderson_stats.bop2_dc_optimization import BOP2DCInfeasibleError, optimize_bop2_dc


def _settings():
    return dict(
        lambda_lrv_grid=[0.7, 0.9],
        lambda_cmv_grid=[0.2, 0.5],
        gamma_lrv_grid=[0.0],
        gamma_cmv_grid=[0.0],
        looks=[2, 4],
        prior=[1, 1],
    )


def test_finite_grid_returns_metrics_and_grid_provenance():
    result = optimize_bop2_dc(4, 0.2, 0.5, 0.1, 0.7, **_settings())
    assert result.candidate_count == 4
    assert result.false_go_rate <= 0.1
    assert result.false_no_go_rate <= 0.1
    assert np.array_equal(result.grid["gamma_lrv"], [0.0])


def test_ess_objective_and_infeasible_constraints():
    result = optimize_bop2_dc(4, 0.2, 0.5, 0.1, 0.7, objective="ess_futile", **_settings())
    assert result.objective == "ess_futile"
    with pytest.raises(BOP2DCInfeasibleError):
        optimize_bop2_dc(4, 0.2, 0.5, 0.1, 0.7, false_go_limit=1e-8, **_settings())
