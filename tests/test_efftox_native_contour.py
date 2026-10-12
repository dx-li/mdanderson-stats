"""Original instruction execution fixtures, independently generated."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import fit_efftox_native_contour

CASES = json.loads((Path(__file__).parent / "fixtures/efftox-native-contour.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[v["case"] for v in CASES])
def test_original_solver(case):
    result = fit_efftox_native_contour(case["targets"])
    np.testing.assert_allclose(result.coefficients, case["coefficients"], rtol=2e-10, atol=2e-10)
    assert result.objective == pytest.approx(case["objective"], rel=2e-8, abs=2e-12)
    assert result.fallback == case["fallback"]
    for actual, expected in zip(result.runs, case["runs"], strict=True):
        np.testing.assert_allclose(actual.start, expected["start"], rtol=2e-10, atol=2e-10)
        # The unselected 200-step tiny-efficacy start amplifies roundoff.
        # Keep tight tolerances for converged runs and the selected fit above.
        rtol, atol = (2e-10, 2e-10) if actual.converged else (2e-6, 2e-8)
        np.testing.assert_allclose(
            actual.coefficients, expected["coefficients"], rtol=rtol, atol=atol
        )
        objective_rtol = 2e-8 if actual.converged else 3e-6
        assert actual.objective == pytest.approx(
            expected["objective"], rel=objective_rtol, abs=2e-12
        )
        assert actual.iterations == expected["iterations"]
        assert actual.evaluations == expected["evaluations"]


def test_radial_bridge_reproduces_curve_and_utility():
    result = fit_efftox_native_contour([[0.5, 0], [0.7, 0.25], [1, 0.65]])
    contour = result.as_contour()
    np.testing.assert_allclose(contour.coefficients, result.coefficients, atol=1e-12)
    np.testing.assert_allclose(contour.utility([0.5, 0.7, 1], [0, 0.25, 0.65]), 0, atol=1e-12)
    with pytest.raises(ValueError, match="fallback"):
        fit_efftox_native_contour([[0.2, 0], [0.4, 0.5], [1, 1]]).as_contour()


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[0.2, 0], [0.4, 0.1]],
        [[0, 0], [0.4, 0.1], [1, 0.2]],
        [[0.4, 0], [0.2, 0.1], [1, 0.2]],
        [[0.2, 0], [0.4, 0], [1, 0.2]],
        [[0.2, 0], [0.4, 0.1], [1, float("nan")]],
        [[1e-200, 0], [0.4, 0.1], [1, 0.2]],
        [["a", 0], [0.4, 0.1], [1, 0.2]],
    ],
)
def test_invalid_targets(points):
    with pytest.raises(ValueError):
        fit_efftox_native_contour(points)


def test_uninitialized_native_output_edge_is_an_explicit_error():
    points = [[0.2, 0], [0.4, 0.1], [1, 0.6]] + [[0.5, float(i % 2)] for i in range(997)]
    with pytest.raises(ArithmeticError, match="100"):
        fit_efftox_native_contour(points)


def test_objective_against_independent_exact_rational_arithmetic():
    from fractions import Fraction
    from random import Random

    from mdanderson_stats.efftox_native_contour import _objective

    points = [
        [Fraction(1, 5), 0],
        [Fraction(1, 2), Fraction(1, 10)],
        [Fraction(7, 10), Fraction(3, 10)],
        [1, Fraction(3, 5)],
    ]
    rng = Random(811)
    for _ in range(100):
        a, b, c = [Fraction(rng.randint(-30, 30), 10) for _ in range(3)]
        residuals = [(a + b / x + c / x**2 - y) ** 2 for x, y in points]
        expected = 4 * residuals[0] + sum(residuals[1:-1]) + 4 * residuals[-1]
        x = points[2][0]
        value = a + b / x + c / x**2
        slope = -b / x**2 - 2 * c / x**3
        expected += 1000 * (max(-slope, 0) + max(-value, 0) + max(value - 1, 0))
        discriminant = b * b - 4 * a * c
        if c > 0 and discriminant < 0:
            expected += 1000 * (c - discriminant)
        actual = _objective(np.array(points, dtype=float), np.array([a, b, c], dtype=float))
        assert actual == pytest.approx(float(expected), rel=1e-14, abs=1e-10)
