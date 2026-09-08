"""Native profile-grid comparisons and independently solvable constrained fits."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import xlogy

from mdanderson_stats import StukelFitError, scan_stukel

CASES = json.loads((Path(__file__).parent / "fixtures/stukel_scan.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["dataset"])
def test_original_minim_grid(case):
    assert np.all(np.isin(case["status"], [3, 4, 5, 6]))
    actual = scan_stukel(
        case["x"], case["successes"], case["trials"], case["alpha1"], case["alpha2"]
    )
    assert actual.shape == (3, 4)
    assert_allclose(actual, case["objective"], rtol=0, atol=1e-7)


def test_intercept_only_profile_has_same_optimum_for_each_shape():
    y = np.array([2, 4, 5, 7])
    n = np.full(4, 10)
    p = y.sum() / n.sum()
    expected = -np.sum(xlogy(y, p) + xlogy(n - y, 1 - p))
    actual = scan_stukel(np.ones((4, 1)), y, n, [-0.5, 0, 0.5], [-0.3, 0, 0.7], intercept=False)
    assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_separated_data_profiles_at_requested_finite_bound():
    for bound in (0.5, 2.0):
        result = scan_stukel(
            [-1, 1], [0, 10], [10, 10], [0], [0], intercept=False, beta_bound=bound
        )
        assert_allclose(result, [[20 * np.logaddexp(0, -bound)]], rtol=1e-12)


def test_saturated_design_does_not_require_residual_degrees_of_freedom():
    y, n = np.array([2, 8]), np.array([10, 10])
    expected = -np.sum(xlogy(y, y / n) + xlogy(n - y, 1 - y / n))
    result = scan_stukel(np.eye(2), y, n, [0.2], [0.3], intercept=False)
    assert_allclose(result, [[expected]], atol=1e-10, rtol=0)


def test_single_observation_profile():
    result = scan_stukel([[1]], [3], [10], [0], [0], intercept=False)
    assert_allclose(result, [[-3 * np.log(0.3) - 7 * np.log(0.7)]], atol=1e-10, rtol=0)


def test_order_and_repeated_grid_values_preserved():
    case = CASES[0]
    first, second = np.array([0.2, -0.2, 0.2]), np.array([0.6, -0.3])
    result = scan_stukel(case["x"], case["successes"], case["trials"], first, second)
    expected = np.array(case["objective"])[np.ix_([2, 0, 2], [3, 0])]
    assert_allclose(result, expected, atol=1e-7, rtol=0)


def test_failed_cell_is_identified_without_returning_incomplete_grid():
    with pytest.raises(StukelFitError, match=r"Scan cell \(0, 0\).+optimizer failed"):
        scan_stukel([-1, 0, 1], [2, 3, 8], [10, 10, 10], [0], [0], max_iterations=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha1": []},
        {"alpha2": [[0]]},
        {"alpha1": [np.nan]},
        {"alpha2": 0},
        {"beta_bound": 0},
        {"beta_bound": np.inf},
        {"max_iterations": 0},
        {"gradient_tolerance": 0},
        {"tolerance": 1},
        {"intercept": 1},
        {"x": [1, 1, 1]},
        {"successes": [2, 3, 11]},
        {"trials": [10, 0, 10]},
    ],
)
def test_invalid_inputs(kwargs):
    arguments = dict(x=[-1, 0, 1], successes=[2, 3, 8], trials=[10, 10, 10], alpha1=[0], alpha2=[0])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        scan_stukel(**arguments)
