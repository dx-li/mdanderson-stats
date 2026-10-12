"""Recovered EffTox 5.2.3 inverse-quadratic contour fitting algorithm.

This is the legacy fitter exported by the native calculations library, not
its current GUI's Lp contour. See docs/efftox-native-contour.md for scope.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .efftox_legacy_contour import EffToxLegacyContour


@dataclass(frozen=True)
class EffToxContourFitRun:
    """One bounded native-profile simplex minimization."""

    start: tuple[float, ...]
    coefficients: tuple[float, ...]
    objective: float
    iterations: int
    evaluations: int
    converged: bool


@dataclass(frozen=True)
class EffToxNativeContourFit:
    """Legacy fitted coefficients and all optimizer diagnostics.

    A fallback is returned by the original solver without a validity recheck.
    Inspect ``fallback`` before using fitted coefficients scientifically.
    """

    targets: tuple[tuple[float, float], ...]
    coefficients: tuple[float, ...]
    objective: float
    runs: tuple[EffToxContourFitRun, ...]
    fallback: bool

    def as_contour(self) -> EffToxLegacyContour:
        """Use a physically valid fitted curve with published radial utility.

        This mathematical bridge supports Python decisions and simulations;
        it is not verification of the original legacy utility implementation.
        Fallback and curves without a monotone unit-square branch are rejected.
        """
        if self.fallback:
            raise ValueError("cannot use a fallback fit as an elicited contour")
        a, b, c = self.coefficients
        roots = np.roots([a, b, c])
        candidates = [float(v.real) for v in roots if v.imag == 0 and 0 < v.real < 1]
        if not candidates:
            raise ValueError("fit has no zero-toxicity endpoint inside the unit square")
        first = max(candidates)
        middle = (first + 1.0) / 2.0
        contour = EffToxLegacyContour(
            [first, middle, 1.0], [0.0, a + b / middle + c / middle**2, a + b + c]
        )
        if not np.allclose(contour.coefficients, self.coefficients, rtol=1e-9, atol=1e-11):
            raise ArithmeticError("radial contour reconstruction lost coefficient precision")
        return contour


def _objective(points: np.ndarray, coefficients: np.ndarray) -> float:
    a, b, c = (float(v) for v in coefficients)
    loss = 0.0
    for i, (x, y) in enumerate(points):
        residual = b / float(x) + a + c / (float(x) * float(x)) - float(y)
        loss += (4.0 if i in (0, len(points) - 1) else 1.0) * residual * residual
    x = float(points[2, 0])
    value = b / x + a + c / (x * x)
    derivative = -b / (x * x) - 2.0 * c / (x * x * x)
    discriminant = b * b - 4.0 * a * c
    loss += 1000.0 * (max(-derivative, 0.0) + max(-value, 0.0) + max(value - 1, 0.0))
    if c > 0 and discriminant < 0:
        loss += 1000.0 * (c - discriminant)
    return float(loss)


def _indices(values: list[float]) -> tuple[int, int, int]:
    # Preserve the native routine's unusual second-highest bookkeeping.
    high, low = (0, 1) if values[0] > values[1] else (1, 0)
    second = low
    for i in (2, 3):
        if values[i] > values[high]:
            second, high = high, i
        if values[i] < values[low]:
            low = i
    return high, low, second


def _minimize(start: np.ndarray, objective: Callable[[np.ndarray], float]) -> EffToxContourFitRun:
    vertices = [start.copy()] + [start + 0.05 * np.eye(3)[i] for i in range(3)]
    values = [objective(v) for v in vertices]
    evaluations = 4
    high, low, second = _indices(values)
    converged = False
    for iteration in range(1, 201):
        centroid = sum((vertices[i] for i in range(4) if i != high), np.zeros(3)) / 3.0
        reflected = 2.0 * centroid - vertices[high]
        fr = objective(reflected)
        evaluations += 1
        if fr < values[low]:
            expanded = 3.0 * reflected - 2.0 * centroid
            fe = objective(expanded)
            evaluations += 1
            vertices[high], values[high] = (expanded, fe) if fe < values[low] else (reflected, fr)
        elif fr > values[second]:
            if fr <= values[high]:
                vertices[high], values[high] = reflected, fr
            contracted = 0.5 * centroid + 0.5 * vertices[high]
            fc = objective(contracted)
            evaluations += 1
            if fc <= values[high]:
                vertices[high], values[high] = contracted, fc
            else:
                for i in range(4):
                    if i != low:
                        vertices[i] = 0.5 * (vertices[i] + vertices[low])
                        values[i] = objective(vertices[i])
                        evaluations += 1
        else:
            vertices[high], values[high] = reflected, fr
        high, low, second = _indices(values)
        distance = float(np.sqrt(sum(float(v) ** 2 for v in vertices[high] - vertices[low])))
        if distance < 0.001 and abs(values[high] - values[low]) < 0.001 * distance:
            converged = True
            break
    return EffToxContourFitRun(
        tuple(float(v) for v in start),
        tuple(float(v) for v in vertices[low]),
        values[low],
        iteration,
        evaluations,
        converged,
    )


def fit_efftox_native_contour(targets: ArrayLike) -> EffToxNativeContourFit:
    """Fit 3–1000 target (efficacy, toxicity) rows using the recovered solver.

    Probabilities must be finite, efficacy positive, and the first three rows
    strictly increasing in both coordinates. Later rows may be nonmonotone.
    Finite nonsingular inputs only are supported; native exception recovery
    and binary-identical floating-point behavior are not claimed.
    """
    raw = np.asarray(targets)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("targets must be a numeric matrix with two columns")
    points = np.asarray(raw, dtype=float)
    if not 3 <= len(points) <= 1000 or not np.isfinite(points).all():
        raise ValueError("targets must contain 3–1000 finite rows")
    if np.any(points < 0) or np.any(points > 1) or np.any(points[:, 0] <= 0):
        raise ValueError("targets must be probabilities with positive efficacy")
    if np.any(np.diff(points[:3], axis=0) <= 0):
        raise ValueError("the first three targets must increase in both coordinates")
    x = points[:3, 0]
    if np.any(points[:, 0] < 1e-100):
        raise ValueError("efficacy is too small for stable inverse-quadratic arithmetic")
    design = np.column_stack((np.ones(3), 1 / x, 1 / (x * x)))
    if not np.isfinite(design).all() or np.linalg.cond(design) > 1e12:
        raise ValueError("target interpolation is numerically ill-conditioned")
    fixed = np.array([0.5, 0.5, -0.5])
    starts = [fixed, np.linalg.solve(design, points[:3, 1])]
    if x[2] == 1:
        design[2] = [0, -1, -2]
        starts.append(np.linalg.solve(design, [points[0, 1], points[1, 1], 0]))
    runs = tuple(_minimize(start, lambda v: _objective(points, v)) for start in starts)
    best = min(runs, key=lambda run: run.objective)
    if best.objective >= 100:
        raise ArithmeticError("no native candidate improved the initial objective bound of 100")
    a, b, c = best.coefficients
    end = float(x[2])
    value = b / end + a + c / (end * end)
    derivative = -b / (end * end) - 2 * c / (end * end * end)
    valid = derivative > 0 and 0 < value < 1 and (c <= 0 or b * b - 4 * a * c >= 0)
    coefficients = best.coefficients if valid else tuple(float(v) for v in fixed)
    return EffToxNativeContourFit(
        tuple((float(e), float(t)) for e, t in points),
        coefficients,
        best.objective if valid else _objective(points, fixed),
        runs,
        not valid,
    )
