"""Iterative WFMM method-of-moments initialization with explicit native differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve

from ._validation import FloatArray, scalar
from .wfmm_model import _finite_real, _strata
from .wfmm_variance_init import _frozen, _frozen_bool


@dataclass(frozen=True)
class WFMMMomentInitialization:
    random_variance: FloatArray
    residual_variance: FloatArray
    fixed_effect_estimates: FloatArray
    normal_equations: FloatArray
    floored: np.ndarray
    omitted_random_components: np.ndarray
    iterations: int
    converged: bool
    legacy_stopping_satisfied: bool
    maximum_change: float
    maximum_signed_change: float
    profile: str
    min_variance: float
    zero_threshold: float
    tolerance: float
    max_iterations: int


def _basis(design: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError(f"{name} has a zero/unrepresentable column")
    scaled = design / norms
    if np.linalg.cond(scaled) > 1e6:
        raise ValueError(f"{name} is rank deficient or ill conditioned")
    q, r = np.linalg.qr(scaled, mode="reduced")
    return q, r * norms


def wfmm_mom_initialization(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    profile: str = "stable",
    min_variance: float = 1e-6,
    zero_threshold: float = 1e-6,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> WFMMMomentInitialization:
    """Solve trace moment equations, alternating with joint GLS coefficient fits.

    Each random block supplies its column-space projector; each residual stratum
    supplies its diagonal indicator. Their covariance-component trace products
    form one constant moment matrix. Negative solutions are floored explicitly.

    Stable mode uses absolute changes and retains every floored component in GLS.
    Explicit legacy_3_1 uses the native signed maximum change and drops random
    components at/below zero_threshold from the GLS covariance. It also omits
    off-diagonal moment entries in all but the first residual-stratum column.
    A native stopping
    result is not marked converged unless the absolute-change criterion also holds.
    This is initialization, not profile ML/REML or a posterior uncertainty estimate.
    """
    y = _finite_real(coefficients, "coefficients")
    x = _finite_real(fixed_design, "fixed_design")
    if (
        y.ndim != 2
        or not 2 <= y.shape[0] <= 500
        or not 1 <= y.shape[1] <= 512
        or x.ndim != 2
        or x.shape[0] != y.shape[0]
        or not 1 <= x.shape[1] < y.shape[0]
        or x.shape[1] > 100
    ):
        raise ValueError("require bounded N-by-K coefficients and full-rank N-by-P fixed design")
    n, k = y.shape
    z = np.empty((n, 0)) if random_design is None else _finite_real(random_design, "random_design")
    if z.ndim != 2 or z.shape[0] != n or z.shape[1] > 500:
        raise ValueError("random_design must have N rows and at most 500 columns")
    rg = _strata(random_strata, z.shape[1], "random_strata")
    sg = _strata(residual_strata, n, "residual_strata")
    h = int(rg.max()) + 1 if rg.size else 0
    c = int(sg.max()) + 1
    floor, threshold, tol, maximum = (
        scalar(v, name)
        for v, name in (
            (min_variance, "min_variance"),
            (zero_threshold, "zero_threshold"),
            (tolerance, "tolerance"),
            (max_iterations, "max_iterations"),
        )
    )
    if (
        profile not in ("stable", "legacy_3_1")
        or floor <= 0
        or threshold < 0
        or tol <= 0
        or maximum != int(maximum)
        or not 1 <= maximum <= 1000
        or any(
            isinstance(v, (bool, np.bool_))
            for v in (min_variance, zero_threshold, tolerance, max_iterations)
        )
    ):
        raise ValueError("require known profile, positive floor/tolerance and 1..1000 iterations")
    maximum = int(maximum)
    if (
        2 * (h + c) * n**2 + 4 * (h + c) * k > 8000000
        or maximum * k * (n**3 + x.shape[1] ** 3) + (h + c) ** 2 * n**2 > 250000000
    ):
        raise ValueError("MOM initialization exceeds bounded cell/work budget")
    if (
        np.max(np.abs(y)) > 1e100
        or np.max(np.abs(x)) > 1e50
        or (z.size and np.max(np.abs(z)) > 1e50)
    ):
        raise ValueError("input scale exceeds bounded numerical range")
    qx, rx = _basis(x, "fixed_design")
    projectors, components = [], []
    for group in range(h):
        block = z[:, rg == group]
        if block.shape[1] > n:
            raise ValueError("random-effect block cannot have more columns than rows")
        qz, _ = _basis(block, "random-effect block")
        projectors.append(qz @ qz.T)
        components.append(block @ block.T)
    for group in range(c):
        diagonal = np.diag((sg == group).astype(float))
        projectors.append(diagonal)
        components.append(diagonal)
    moment_matrix = np.array(
        [[np.sum(p * component.T) for component in components] for p in projectors]
    )
    if profile == "legacy_3_1" and c > 1:
        # Native 0x488970..0x4889a8 visits off-diagonals only through H,
        # omitting later residual-stratum cross terms. Keep this defect explicit.
        for column in range(h + 1, h + c):
            diagonal = moment_matrix[column, column]
            moment_matrix[:, column] = 0.0
            moment_matrix[column, column] = diagonal
    if np.linalg.cond(moment_matrix) > 1e12:
        raise ValueError("variance-component moment equations are singular or ill conditioned")
    beta_q = qx.T @ y
    previous = np.zeros((h + c, k))
    variance = previous.copy()
    omitted = np.zeros((h, k), dtype=bool)
    floored = np.zeros_like(variance, dtype=bool)
    change = signed = float("inf")
    for iteration in range(1, maximum + 1):
        residual = y - qx @ beta_q
        moments = np.array([np.einsum("ij,ij->j", residual, p @ residual) for p in projectors])
        raw = np.linalg.solve(moment_matrix, moments)
        if not np.all(np.isfinite(raw)):
            raise ArithmeticError("MOM equations returned nonfinite variances")
        floored = raw < floor
        variance = np.maximum(raw, floor)
        change = float(np.max(np.abs(variance - previous)))
        signed = float(np.max(variance - previous))
        effective = variance.copy()
        if profile == "legacy_3_1":
            omitted = effective[:h] <= threshold
            effective[:h][omitted] = 0.0
        for column in range(k):
            covariance = sum(
                v * component for v, component in zip(effective[:, column], components, strict=True)
            )
            if not np.all(np.isfinite(covariance)) or np.linalg.cond(covariance) > 1e12:
                raise ArithmeticError("MOM covariance is nonfinite or ill conditioned")
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inv_x = cho_solve(factor, qx, check_finite=False)
            inv_y = cho_solve(factor, y[:, column], check_finite=False)
            information = qx.T @ inv_x
            if np.linalg.cond(information) > 1e12:
                raise ArithmeticError("MOM GLS information is ill conditioned")
            beta_q[:, column] = np.linalg.solve(information, qx.T @ inv_y)
        if not np.all(np.isfinite(beta_q)):
            raise ArithmeticError("MOM GLS coefficients are nonfinite")
        if (signed if profile == "legacy_3_1" else change) <= tol:
            break
        previous = variance
    beta = np.linalg.solve(rx, beta_q)
    if not np.all(np.isfinite(beta)):
        raise ArithmeticError("MOM restored coefficients are nonfinite")
    return WFMMMomentInitialization(
        _frozen(variance[:h]),
        _frozen(variance[h:]),
        _frozen(beta),
        _frozen(moment_matrix),
        _frozen_bool(floored),
        _frozen_bool(omitted),
        iteration,
        change <= tol,
        signed <= tol,
        change,
        signed,
        profile,
        floor,
        threshold,
        tol,
        maximum,
    )
