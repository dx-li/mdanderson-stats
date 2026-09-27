"""SURVAN multivariable Cox regression with Breslow tied-event likelihood."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc, ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .survan_cox_likelihood import _CoxLikelihood


@dataclass(frozen=True)
class SurvanCox:
    coefficients: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    one_sided_pvalues: FloatArray
    two_sided_pvalues: FloatArray
    negative_log_likelihood: float
    null_negative_log_likelihood: float
    likelihood_ratio: float
    degrees_of_freedom: int
    pvalue: float
    iterations: int
    column_scale: FloatArray
    column_center: FloatArray
    scaled_coefficients: FloatArray
    ties: str = "breslow"
    strata_labels: tuple[str | int, ...] = ()

    def log_relative_hazard(self, x: ArrayLike) -> FloatArray:
        """Log hazard relative to the training covariate means, not baseline survival."""
        if np.iscomplexobj(x):
            raise ValueError("x must be real")
        xx = finite(x, "x")
        if xx.ndim == 1:
            xx = xx[:, None]
        if xx.ndim != 2 or xx.shape[1] != self.column_scale.size:
            raise ValueError("x must have the fitted covariate count")
        with np.errstate(over="ignore", invalid="ignore"):
            eta = (xx / self.column_scale - self.column_center) @ self.scaled_coefficients
        if not np.isfinite(eta).all():
            raise ArithmeticError("log relative hazard exceeds numerical range")
        return _freeze(eta)


def survan_cox(
    time: ArrayLike,
    event: ArrayLike,
    x: ArrayLike,
    *,
    ties: str = "breslow",
    strata: ArrayLike | None = None,
) -> SurvanCox:
    """Fit a static-covariate proportional-hazards model without an intercept.

    Events=1 and right censors=0. Tied censors remain in risk sets. Optional
    string or integer ``strata`` labels fit shared covariate effects with separate baseline
    hazards. ``ties`` is
    ``"breslow"`` (the backward-compatible default) or ``"efron"``. Rank
    deficiency and monotone likelihood raise errors. Time-varying covariates,
    delayed entry, case weights and baseline survival estimation are not included.
    """
    if ties not in ("breslow", "efron"):
        raise ValueError("ties must be 'breslow' or 'efron'")
    if any(np.iscomplexobj(a) for a in (time, event, x)) or (
        strata is not None and np.iscomplexobj(strata)
    ):
        raise ValueError("time, event, x and strata must be real")
    t, e, xx = finite(time, "time"), count(event, "event"), finite(x, "x")
    if xx.ndim == 1:
        xx = xx[:, None]
    if (
        t.ndim != 1
        or e.shape != t.shape
        or xx.ndim != 2
        or xx.shape[0] != t.size
        or not 1 <= xx.shape[1] <= 100
        or not 2 <= t.size <= 100_000
        or xx.size > 2_000_000
        or np.any(t < 0)
        or np.any(e > 1)
    ):
        raise ValueError(
            "require nonnegative times, binary events and matching x; "
            "<=100000 rows, 1..100 covariates, <=2e6 design entries"
        )
    if not e.any():
        raise ValueError("Cox regression requires events")
    if strata is None:
        strata_codes = None
        strata_labels: tuple[str | int, ...] = ()
    else:
        strata_codes, strata_labels = _encode_strata(strata)
        if strata_codes.shape != t.shape:
            raise ValueError("strata must contain one label per row")
    p = xx.shape[1]
    scale = np.max(np.abs(xx), axis=0)
    if np.any(scale == 0):
        raise ValueError("Cox design is rank deficient")
    normalized = xx / scale
    center = normalized.mean(axis=0)
    design = normalized - center
    if np.linalg.matrix_rank(design) < p:
        raise ValueError("Cox design is rank deficient; do not include an intercept")
    model = _CoxLikelihood(t, e, design, ties=ties, strata=strata_codes)
    beta = np.zeros(p)
    null_nll, gradient, info, _ = model.evaluate(beta)
    if np.linalg.matrix_rank(info) < p:
        raise ValueError("Cox risk sets do not identify all coefficients")
    model.check_separation(gradient)
    nll = null_nll
    for iteration in range(101):
        eigenvalues = np.linalg.eigvalsh(info)
        if eigenvalues[0] <= 64 * np.finfo(float).eps * p * eigenvalues[-1]:
            raise ArithmeticError("Cox information is numerically singular")
        step = np.linalg.solve(info, gradient)
        decrement = float(gradient @ step)
        if decrement < 1e-14:
            break
        if iteration == 100:
            raise ArithmeticError("Cox optimization did not converge in 100 iterations")
        fraction = 1.0
        for _ in range(50):
            candidate = beta - fraction * step
            next_nll, next_gradient, next_info, _ = model.evaluate(candidate)
            slack = 8 * np.finfo(float).eps * max(1, abs(nll))
            if next_nll <= nll - 1e-4 * fraction * decrement + slack:
                beta, nll, gradient, info = candidate, next_nll, next_gradient, next_info
                break
            fraction /= 2
        else:
            raise ArithmeticError("Cox likelihood line search failed")
    scaled_covariance = np.linalg.solve(info, np.eye(p))
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = beta / scale
        covariance = scaled_covariance / scale[:, None] / scale[None, :]
    if not np.isfinite(coefficients).all() or not np.isfinite(covariance).all():
        raise ArithmeticError("Cox coefficient units exceed numerical range")
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ArithmeticError("Cox variance is not representable")
    se = np.sqrt(variances)
    one_sided = ndtr(-np.abs(coefficients / se))
    lr = max(0.0, 2 * (null_nll - nll))
    return SurvanCox(
        _freeze(coefficients),
        _freeze(covariance),
        _freeze(se),
        _freeze(one_sided),
        _freeze(2 * one_sided),
        nll,
        null_nll,
        lr,
        p,
        float(gammaincc(p / 2, lr / 2)),
        iteration,
        _freeze(scale),
        _freeze(center),
        _freeze(beta),
        ties,
        strata_labels,
    )


def _encode_strata(values: ArrayLike) -> tuple[np.ndarray, tuple[str | int, ...]]:
    if isinstance(values, (str, bytes)):
        raise ValueError("strata must be a label sequence, not a scalar string")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.size > 100_000:
            raise ValueError("strata must be a one-dimensional vector of at most 100,000 labels")
        raw = values.astype(object, copy=False)
    else:
        try:
            sequence = cast(Sequence[object], values)
            length = len(sequence)
        except TypeError as exc:
            raise ValueError("strata must be a bounded one-dimensional label sequence") from exc
        if length > 100_000:
            raise ValueError("strata must contain at most 100,000 labels")
        raw = np.fromiter(sequence, dtype=object, count=length)
    if raw.ndim != 1 or raw.size > 100_000:
        raise ValueError("strata must be a one-dimensional vector of at most 100,000 labels")
    labels: list[str | int] = []
    lookup: dict[str | int, int] = {}
    codes = np.empty(raw.size, dtype=np.int64)
    for index, value in enumerate(raw.tolist()):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("stratum labels cannot be booleans")
        if isinstance(value, (int, np.integer)):
            label: str | int = int(value)
        elif isinstance(value, (str, np.str_)) and str(value):
            label = str(value)
        else:
            raise ValueError("stratum labels must be nonempty strings or integers")
        if label not in lookup:
            lookup[label] = len(labels)
            labels.append(label)
        codes[index] = lookup[label]
    if not labels or len(labels) > 100:
        raise ValueError("require 1..100 strata")
    return codes, tuple(labels)
