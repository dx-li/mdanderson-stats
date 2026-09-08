"""Prior allocation objectives against independent prior evaluators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    single_design_precision,
    single_normal_criterion,
    single_optimize_allocations,
    single_optimize_prior_allocations,
    single_uniform_criterion,
)


@pytest.mark.parametrize("prior_kind", ["uniform", "normal"])
@pytest.mark.parametrize("measure", ["variance", "sd"])
@pytest.mark.parametrize("criterion", ["slope", "quantile"])
@pytest.mark.parametrize("model", ["logistic", "loglog"])
def test_optimized_objective_matches_prior_evaluator(prior_kind, measure, criterion, model):
    x = [-1, 2]
    options = dict(criterion=f"{criterion}_{measure}", model=model)

    def evaluate(n):
        if prior_kind == "uniform":
            return single_uniform_criterion(x, n, [0.1, 0.7], [0.5, 1.1], **options)
        return single_normal_criterion(x, n, [0.3, 0.9], [[0.01, 0.002], [0.002, 0.01]], **options)

    initial = evaluate([50, 50])
    aggregation = "arithmetic" if prior_kind == "uniform" else "harmonic"
    r = single_optimize_prior_allocations(
        x,
        initial.parameters,
        initial.weights,
        criterion=criterion,
        measure=measure,
        aggregation=aggregation,
        model=model,
    )
    assert_allclose(r.value, evaluate(r.subjects).value, rtol=1e-12)
    assert_allclose(r.initial_value, initial.value)
    assert r.value <= initial.value * (1 + 1e-8)
    # Independent finite-grid search brackets the optimum in a two-dose design.
    grid_best = min(evaluate([f, 100 - f]).value for f in np.linspace(1, 99, 51))
    assert r.value <= grid_best * (1 + 1e-8)
    assert r.optimality_gap < 1e-4


@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
@pytest.mark.parametrize("measure,power", [("sd", 0.5), ("variance", 1)])
def test_point_prior_reduction_and_scaling(aggregation, measure, power):
    point = single_optimize_allocations([-2, 2], [0, 1])
    r = single_optimize_prior_allocations(
        [-2, 2], [[0, 1]], [1], aggregation=aggregation, measure=measure, total_subjects=200
    )
    assert_allclose(r.subjects, 2 * point.subjects, atol=0.001)
    assert_allclose(r.value, (point.variance / 2) ** power, rtol=1e-8)


@pytest.mark.parametrize("form", ["linear", "centered"])
def test_symmetric_uncertain_slope_design(form):
    parameters = [[0, 0.8], [0, 1.2]] if form == "linear" else [[0.8, 0], [1.2, 0]]
    r = single_optimize_prior_allocations(
        [-2, -1, 0, 1, 2], parameters, [0.4, 0.6], criterion="slope", form=form
    )
    assert_allclose(r.subjects, [50, 0, 0, 0, 50], atol=1e-7)
    variance = single_design_precision(r.doses, r.subjects, parameters, form=form).slope_variance
    assert_allclose(r.value, np.dot([0.4, 0.6], variance))


def test_iteration_failure_is_explicit():
    with pytest.raises(RuntimeError, match="failed"):
        single_optimize_prior_allocations([-2, 2], [[0, 1]], [1], max_iterations=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"parameters": [0, 1]},
        {"prior_weights": [-1, 2]},
        {"prior_weights": [0.5]},
        {"prior_weights": [0, 1]},
        {"prior_weights": [np.nan, 1]},
        {"aggregation": "geometric"},
        {"measure": "power"},
    ],
)
def test_invalid_prior_request(changes):
    args = dict(doses=[-1, 2], parameters=[[0, 1], [0.2, 0.8]], prior_weights=[0.5, 0.5])
    args.update(changes)
    with pytest.raises(ValueError):
        single_optimize_prior_allocations(**args)
