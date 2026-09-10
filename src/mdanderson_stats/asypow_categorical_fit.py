"""Concave categorical expected likelihood under fixed/equality constraints."""

from math import fsum

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ._validation import FloatArray
from .asypow_constraints import _components


def _fit_mass(a: FloatArray, b: FloatArray, count: FloatArray) -> FloatArray:
    size = b.shape[1]
    phase = linprog(
        np.r_[np.zeros(size), -1],
        A_ub=np.column_stack((-b, np.ones(len(a)))),
        b_ub=a,
        bounds=[(0, 1)] * (size + 1),
        method="highs",
    )
    if not phase.success:
        raise ValueError(f"categorical null feasibility failed: {phase.message}")
    y = np.asarray(phase.x[:-1], dtype=float)
    if np.any(a + b @ y <= 0):
        raise ValueError("categorical null has no numerically resolved positive interior")
    root_count = np.sqrt(count)
    for _ in range(200):
        mass = a + b @ y
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            features = (b / mass[:, None]) * root_count[:, None]
        if not np.all(np.isfinite(features)):
            raise ArithmeticError("categorical likelihood curvature is not representable")
        scale = np.max(np.abs(features), axis=0)
        if np.any(scale == 0):
            raise ArithmeticError("categorical likelihood has unresolved curvature")
        u, singular, vt = np.linalg.svd(features / scale, full_matrices=False)
        if singular[-1] <= np.finfo(float).eps * max(features.shape) * singular[0]:
            raise ArithmeticError("categorical likelihood is numerically ill-conditioned")
        projection = u.T @ root_count
        step = (vt.T @ (projection / singular)) / scale
        change = b @ step
        relative = change / mass
        if np.max(np.abs(relative)) <= 1e-10:
            # Apply the final tiny correction to avoid an avoidable error floor
            # in divergence for alternatives extremely close to the null.
            candidate = y + step
            return candidate if np.all(a + b @ candidate > 0) else y
        alpha = min(1.0, float(0.99 / -relative.min())) if relative.min() < 0 else 1.0
        slope = float(projection @ projection)
        for _ in range(60):
            if -count @ np.log1p(alpha * relative) <= -1e-4 * alpha * slope:
                candidate = y + alpha * step
                if np.all(a + b @ candidate > 0):
                    y = candidate
                    break
            alpha *= 0.5
        else:
            raise ArithmeticError("categorical likelihood line search failed")
    raise ArithmeticError("categorical likelihood did not converge in 200 iterations")


def _equality_categorical_null(
    p: FloatArray,
    mass: FloatArray,
    constraints: ArrayLike,
    ordinal: bool,
    log_weight: FloatArray,
) -> tuple[FloatArray, int]:
    components = _components(constraints, p.size)
    free = sum(value is None for _, value in components)
    fixed = np.zeros(p.size)
    design = np.zeros((p.size, free))
    column = 0
    already_null = True
    for indices, value in components:
        if value is None:
            design[indices, column] = 1
            column += 1
            already_null &= bool(np.all(p.ravel()[indices] == p.ravel()[indices[0]]))
        else:
            if not 0 < value < 1:
                raise ValueError("fixed categorical parameters must be strictly within (0,1)")
            fixed[indices] = value
            already_null &= bool(np.all(p.ravel()[indices] == value))
    df = p.size - free
    if already_null:
        return p.copy(), df
    groups, width = p.shape
    a = np.zeros((groups, width + 1))
    b = np.zeros((groups, width + 1, free))
    for g in range(groups):
        values = fixed[g * width : (g + 1) * width]
        rows = design[g * width : (g + 1) * width]
        if ordinal:
            a[g] = np.diff(np.r_[0, values, 1])
            b[g] = np.diff(np.vstack((np.zeros(free), rows, np.zeros(free))), axis=0)
        else:
            a[g] = np.r_[values, fsum([1.0, *(-values)])]
            b[g] = np.vstack((rows, -rows.sum(axis=0)))
    log_count = log_weight[:, None] + np.log(mass)
    with np.errstate(under="ignore"):
        count = np.exp(log_count.ravel() - log_count.max())
    if np.any(count == 0):
        raise ArithmeticError("categorical likelihood weights span an unresolved numerical range")
    y = _fit_mass(a.ravel(), b.reshape(-1, free), count)
    return (fixed + design @ y).reshape(p.shape), df
