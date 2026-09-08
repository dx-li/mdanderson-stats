"""Allocation optimization against printed examples and independent optima."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_design_precision, single_optimize_allocations


def test_original_optimized_example_allocations():
    r = single_optimize_allocations([-2.399239, 2.399495], [0, 1])
    assert_allclose(r.subjects, [90.741824, 9.258176], atol=0.001)
    assert_allclose(r.sd, 0.444280, atol=1e-6)
    assert r.variance < r.initial_variance
    assert r.optimality_gap < 1e-5


def test_symmetric_slope_optimum_discards_interior_doses():
    r = single_optimize_allocations([-2, -1, 0, 1, 2], [0, 1], criterion="slope")
    assert_allclose(r.subjects, [50, 0, 0, 0, 50], atol=1e-8)
    w = np.exp(-2) / (1 + np.exp(-2)) ** 2
    assert_allclose(r.variance, 1 / (400 * w))


@pytest.mark.parametrize("model", ["logistic", "loglog"])
@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("criterion", ["slope", "quantile"])
def test_two_point_closed_form_optimal_split(model, form, criterion):
    x, b = np.array([-1.0, 2.0]), np.array([0.3, 0.8])

    # With two dose points, variance = A/n1+B/n2. Recover A and B
    # independently from two fixed-design variances, then minimize analytically.
    def variance(n):
        r = single_design_precision(x, n, b, model=model, form=form)
        return float(r.slope_variance if criterion == "slope" else r.quantile_variance)

    ab = np.linalg.solve([[1, 1], [0.5, 1]], [variance([1, 1]), variance([2, 1])])
    root = np.sqrt(ab)
    expected = 100 * root / root.sum()
    r = single_optimize_allocations(x, b, model=model, form=form, criterion=criterion)
    assert_allclose(r.subjects, expected, atol=0.002)
    assert_allclose(r.variance, root.sum() ** 2 / 100, rtol=1e-8)


def test_initialization_total_scaling_and_permutation():
    r = single_optimize_allocations([-2, 2], [0, 1], initial_subjects=[20, 80])
    swapped = single_optimize_allocations([2, -2], [0, 1], total_subjects=200)
    assert_allclose(swapped.subjects[::-1], r.subjects * 2, atol=0.001)
    assert_allclose(swapped.variance, r.variance / 2, rtol=1e-8)
    actual = single_design_precision(r.doses, r.subjects, [0, 1])
    assert_allclose(actual.quantile_variance, r.variance)


def test_zero_slope_symmetric_optimum():
    r = single_optimize_allocations([-1, 0, 1], [0, 0], criterion="slope")
    assert_allclose(r.subjects, [50, 0, 50], atol=1e-8)
    assert_allclose(r.variance, 0.04)


def test_iteration_limit_is_reported():
    with pytest.raises(RuntimeError, match="failed"):
        single_optimize_allocations([-2, 2], [0, 1], max_iterations=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"total_subjects": 0},
        {"tolerance": 0},
        {"max_iterations": True},
        {"max_iterations": 0},
        {"criterion": "power"},
        {"parameters": [[0, 1]]},
        {"doses": [1, 1]},
        {"initial_subjects": [0, 100]},
        {"initial_subjects": [30, 60]},
        {"initial_subjects": [-1, 101]},
        {"quantile": 1},
    ],
)
def test_invalid_request(changes):
    args = dict(doses=[-1, 1], parameters=[0, 1])
    args.update(changes)
    with pytest.raises(ValueError):
        single_optimize_allocations(**args)
