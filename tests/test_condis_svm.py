import numpy as np
import pytest

from mdanderson_stats.condis import condis_impute
from mdanderson_stats.condis_svm import (
    _estimate_sigma,
    _fit_scaled,
    _rbf_kernel,
    condis_svm_refine,
)


def test_svm_refinement_reconstructs_fit_and_restores_events():
    time = np.arange(1.0, 13.0)
    status = np.tile([1, 0, 1, 0], 3)
    imputation = condis_impute(time, status, horizon=12.0)
    covariates = np.column_stack((np.sin(time), time**2))
    pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    folds = np.tile([0, 1, 2], 4)

    result = condis_svm_refine(
        imputation,
        covariates,
        folds=3,
        fold_ids=folds,
        sigma_pair_indices=pairs,
        solver_tolerance=1e-8,
        max_iterations=10_000,
    )

    design = np.column_stack((status, covariates))
    if result.standardized:
        scaled = (
            design / result.predictor_magnitude - result.predictor_center
        ) / result.predictor_scale
        expected = result.response_center + result.response_scale * (
            _rbf_kernel(scaled, scaled, result.sigma) @ result.dual_coefficients + result.intercept
        )
    else:
        expected = (
            _rbf_kernel(design, design, result.sigma) @ result.dual_coefficients + result.intercept
        )
    np.testing.assert_allclose(result.fitted_time, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.fold_ids[0], folds)
    np.testing.assert_array_equal(result.refined_time[status == 1], time[status == 1])
    assert np.max(np.abs(result.dual_coefficients)) <= result.best_cost + 1e-10
    assert result.final_kkt_error <= 1e-7
    assert result.final_duality_gap >= -1e-10


def test_smo_all_bound_intercept_and_update_limit():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 4.0, 6.0])
    fit = _fit_scaled(x, y, x, 0.01, 0.5, 0.1, 1e-8, 2)
    np.testing.assert_allclose(fit.dual_coefficients, [-0.01, -0.01, 0.01, 0.01])
    assert fit.intercept == pytest.approx(3.0)
    np.testing.assert_allclose(
        fit.prediction,
        _rbf_kernel(x, x, 0.5) @ fit.dual_coefficients + fit.intercept,
    )
    assert fit.iterations == 2
    assert fit.kkt_error == pytest.approx(0.0)
    assert fit.duality_gap == pytest.approx(0.0)
    with pytest.raises(ArithmeticError, match="KKT working-set gap"):
        _fit_scaled(x, y, x, 0.01, 0.5, 0.1, 1e-8, 1)


def test_rbf_and_sigma_retain_extreme_raw_distance_information():
    left = np.array([[1e150]])
    right = np.array([[np.nextafter(1e150, np.inf)]])
    delta = float(right[0, 0] - left[0, 0])
    sigma = 1.0 / (delta * delta)
    kernel = _rbf_kernel(left, right, sigma)
    assert kernel[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-14)

    x = np.column_stack((np.ones(24), np.r_[0.0, 1e150, 0.0, 1e308, np.zeros(20)]))
    pairs = np.tile(np.array([[0, 1]], dtype=np.int64), (12, 1))
    pairs[-1] = [2, 3]
    estimated = _estimate_sigma(x, pairs)
    assert estimated == pytest.approx(1e-300, rel=2e-14)
    assert np.isfinite(_rbf_kernel(x[[0]], x[[1]], estimated)).all()

    tiny_x = np.column_stack((np.ones(4), [0.0, 1e-154, 0.0, 1e-154]))
    tiny_pairs = np.array([[0, 1], [2, 3]])
    assert _estimate_sigma(tiny_x, tiny_pairs) == pytest.approx(1e308, rel=2e-14)


def test_svm_rejects_single_row_training_folds():
    time = np.array([1.0, 2.0, 3.0])
    status = np.array([1, 0, 1])
    imputation = condis_impute(time, status, horizon=3.0)
    with pytest.raises(ValueError, match="at least two rows"):
        condis_svm_refine(
            imputation,
            np.array([[0.0], [1.0], [2.0]]),
            folds=2,
            fold_ids=np.array([0, 0, 1]),
            sigma=0.5,
        )
