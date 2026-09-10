import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    compare_predictive_survival,
    response_survival_posterior,
    simulate_response_survival,
)


def test_conjugate_updates_and_single_category_exact_probability():
    fit = response_survival_posterior(
        [[2], [3]],
        [[1], [2]],
        [[6], [10]],
        response_prior=[0.5],
        survival_shape=[3],
        survival_scale=[5],
        draws=100000,
        seed=82,
    )
    assert_allclose(fit.response_concentration, [[2.5], [3.5]])
    assert_allclose(fit.survival_shape, [[4], [5]])
    assert_allclose(fit.survival_scale, [[11], [15]])
    exact = compare_predictive_survival([1, 2], [6, 10], prior=[[3, 5], [3, 5]])
    assert (
        abs(fit.probability_b_superior - float(exact.arm_b_probability))
        < 5 * fit.monte_carlo_standard_error
    )
    scaled = response_survival_posterior(
        [[2], [3]],
        [[1], [2]],
        np.array([[6], [10]]) * 1e200,
        response_prior=[0.5],
        survival_shape=[3],
        survival_scale=[5e200],
        draws=100000,
        seed=82,
    )
    assert scaled.probability_b_superior == fit.probability_b_superior
    with pytest.raises(ValueError):
        fit.survival_scale[0, 0] = 1
    with pytest.raises(ValueError, match="events<=counts"):
        response_survival_posterior(
            [[0], [0]],
            [[1], [0]],
            [[1], [0]],
            response_prior=[1],
            survival_shape=[1],
            survival_scale=[1],
        )


def test_stop_before_enrollment_and_actual_accrual_followup():
    stopped = simulate_response_survival(
        [[1], [1]],
        [[1], [1]],
        response_prior=[1],
        survival_shape=[100],
        survival_scale=[[100], [100000]],
        max_patients=12,
        initial_patients=4,
        accrual_per_period=2,
        replicates=8,
        posterior_draws=1000,
        seed=82,
    )
    assert_allclose(stopped.enrollment.sum(axis=1), 4)
    assert np.all(stopped.selected_arm == 1) and np.all(stopped.stopped_early)
    assert_allclose(stopped.analysis_time, 3)
    complete = simulate_response_survival(
        [[1], [1]],
        [[1], [1]],
        response_prior=[1],
        survival_shape=[5],
        survival_scale=[5],
        max_patients=12,
        initial_patients=12,
        accrual_per_period=3,
        additional_followup=8,
        replicates=8,
        posterior_draws=1000,
        seed=82,
    )
    assert_allclose(complete.enrollment.sum(axis=1), 12)
    assert not np.any(complete.stopped_early)
    assert_allclose(complete.analysis_time, 12)
    assert_allclose(complete.mean_enrollment, complete.enrollment.mean(axis=0))
