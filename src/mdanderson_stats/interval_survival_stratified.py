"""Shared-coefficient stratified PH regression for interval-censored data.

Each stratum has a separate Turnbull support and baseline, while one shared
regression vector is estimated from the summed likelihood. This Python
extension does not claim parity with ``mets``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .interval_survival import (
    _MAX_COVARIATES,
    _MAX_ITERATIONS,
    _MAX_OUTPUT_CELLS,
    _MAX_ROWS,
    _MAX_SUPPORT,
    _MAX_WORK,
    IntervalSurvivalFit,
    IntervalSurvivalPrediction,
    _likelihood_terms,
    _maximal_intersections,
    _pava_increasing,
    _readonly_bool,
    predict_interval_survival,
)
from .survan_cox import _encode_strata


@dataclass(frozen=True)
class StratifiedIntervalBaseline:
    """One stratum's estimated support and baseline probability jumps."""

    label: str | int
    support_lower: FloatArray
    support_upper: FloatArray
    support_left_closed: np.ndarray
    support_mass: FloatArray
    log_cumulative_hazard: FloatArray
    log_likelihood: float
    kkt_error: float
    covariate_mean: FloatArray
    covariate_center_scaled: FloatArray


@dataclass(frozen=True)
class StratifiedIntervalSurvivalFit:
    """Joint fit with one shared coefficient vector and distinct baselines."""

    coefficients: FloatArray
    covariate_mean: FloatArray
    covariate_scale: FloatArray
    covariate_unit_scale: FloatArray
    covariate_center_scaled: FloatArray
    covariate_scale_scaled: FloatArray
    scaled_coefficients: FloatArray
    strata: tuple[StratifiedIntervalBaseline, ...]
    log_likelihood: float
    iterations: int
    score_error: float
    converged: bool

    @property
    def stratum_labels(self) -> tuple[str | int, ...]:
        return tuple(item.label for item in self.strata)

    def for_stratum(self, label: str | int) -> IntervalSurvivalFit:
        """Return a standard single-baseline fit view for existing predictors."""
        baseline = next((part for part in self.strata if part.label == label), None)
        if baseline is None or isinstance(label, (bool, np.bool_)):
            raise ValueError(f"unknown stratum label: {label!r}")
        return IntervalSurvivalFit(
            self.coefficients,
            baseline.covariate_mean,
            self.covariate_scale,
            self.covariate_unit_scale,
            baseline.covariate_center_scaled,
            self.covariate_scale_scaled,
            baseline.support_lower,
            baseline.support_upper,
            baseline.support_left_closed,
            baseline.support_mass,
            baseline.log_likelihood,
            self.iterations,
            baseline.kkt_error,
            self.converged,
            self.scaled_coefficients,
            baseline.log_cumulative_hazard,
        )


def _baseline(
    label: str | int,
    lower: FloatArray,
    upper: FloatArray,
    closed: np.ndarray,
    q: FloatArray,
    ll: float,
    kkt: float,
) -> StratifiedIntervalBaseline:
    boundaries = np.r_[-np.inf, q, np.inf]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_s = -np.exp(boundaries)
        survival = np.exp(log_s)
        mass = survival[:-1] * (-np.expm1(log_s[1:] - log_s[:-1]))
    if not np.isfinite(mass).all() or np.any(mass < -1e-12):
        raise ArithmeticError("stratified interval fit produced invalid baseline probabilities")
    mass = np.maximum(mass, 0.0)
    total = float(mass.sum())
    if not isfinite(ll):
        raise ArithmeticError("stratified interval log likelihood is not representable")
    if not isfinite(total) or total <= 0:
        raise ArithmeticError("stratified interval fit has no representable baseline mass")
    mass /= total
    return StratifiedIntervalBaseline(
        label,
        _freeze(lower),
        _freeze(upper),
        _readonly_bool(closed),
        _freeze(mass),
        _freeze(q),
        ll,
        kkt,
        _freeze(np.empty(0)),
        _freeze(np.empty(0)),
    )


def fit_stratified_interval_survival(
    lower: ArrayLike,
    upper: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    strata: ArrayLike,
    weights: ArrayLike | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    max_support: int = _MAX_SUPPORT,
    max_work: int = _MAX_WORK,
) -> StratifiedIntervalSurvivalFit:
    """Fit interval-censored PH regression with a shared slope and stratum baselines.

    Data use ``(lower, upper]`` intervals, exact events at equal positive
    endpoints, and ``upper=inf`` for right censoring. A stratum containing only
    right-censored rows is rejected because its finite maximum baseline is not
    identified by this implementation.
    """
    if any(
        np.iscomplexobj(v) for v in (lower, upper, covariates, strata, weights) if v is not None
    ):
        raise ValueError("interval, covariate, weight, and stratum data must be real")
    if isinstance(lower, np.ndarray) and (lower.ndim != 1 or lower.size > _MAX_ROWS):
        raise ValueError(f"lower must be a vector with at most {_MAX_ROWS} rows")
    if isinstance(upper, np.ndarray) and (upper.ndim != 1 or upper.size > _MAX_ROWS):
        raise ValueError(f"upper must be a vector with at most {_MAX_ROWS} rows")
    lo = finite(lower, "lower")
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 1 or not 1 <= lo.size <= _MAX_ROWS or hi.shape != lo.shape:
        raise ValueError(f"lower and upper must be aligned vectors with 1..{_MAX_ROWS} rows")
    if np.isnan(hi).any() or np.isneginf(hi).any():
        raise ValueError("upper may contain positive infinity for right censoring only")
    if np.any(lo < 0) or np.any(lo > hi) or np.any((lo == hi) & (lo == 0)):
        raise ValueError("intervals require 0 <= lower <= upper, with exact times positive")
    codes, labels = _encode_strata(strata)
    if codes.shape != lo.shape:
        raise ValueError("strata must contain one label per interval")
    if isinstance(covariates, np.ndarray) and (covariates.ndim > 2 or covariates.size > 2_000_000):
        raise ValueError("covariates exceed the two-million-cell design limit")
    x = np.empty((lo.size, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != lo.size or x.shape[1] > _MAX_COVARIATES:
        raise ValueError("covariates must have one row per interval and at most 100 columns")
    if x.size > 2_000_000:
        raise ValueError("covariate matrix exceeds the 2000000-cell limit")
    w = np.ones(lo.size) if weights is None else finite(weights, "weights")
    if w.shape != lo.shape or np.any(w <= 0):
        raise ValueError("weights must be positive, finite, and aligned with intervals")
    weight_scale = float(np.max(w))
    if not np.isfinite((w / weight_scale).sum()):
        raise ValueError("weights have an invalid total")
    tol = scalar(tolerance, "tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or int(max_iterations) != max_iterations
        or not 1 <= max_iterations <= _MAX_ITERATIONS
    ):
        raise ValueError(f"max_iterations must be in [1, {_MAX_ITERATIONS}]")
    if (
        isinstance(max_support, (bool, np.bool_))
        or int(max_support) != max_support
        or not 2 <= max_support <= _MAX_SUPPORT
    ):
        raise ValueError(f"max_support must be in [2, {_MAX_SUPPORT}]")
    if (
        isinstance(max_work, (bool, np.bool_))
        or int(max_work) != max_work
        or not 1 <= max_work <= _MAX_WORK
    ):
        raise ValueError(f"max_work must be in [1, {_MAX_WORK}]")

    groups: list[tuple[np.ndarray, FloatArray, FloatArray, np.ndarray, np.ndarray, np.ndarray]] = []
    support_total = work_total = 0
    for group, label in enumerate(labels):
        rows = np.flatnonzero(codes == group)
        if np.isposinf(hi[rows]).all():
            raise ValueError(
                f"stratum {label!r} is entirely right-censored; baseline maximum is unidentified"
            )
        sl, su, closed, (li, ri) = _maximal_intersections(lo[rows], hi[rows])
        support_total += sl.size
        work_total += rows.size * max(sl.size, 1)
        if support_total > max_support:
            raise ValueError(f"combined Turnbull support exceeds the {max_support}-interval limit")
        if work_total > max_work:
            raise ValueError("combined interval likelihood work estimate exceeds max_work")
        groups.append((rows, sl, su, closed, li, ri))
    if support_total > max_support:
        raise ValueError(
            f"combined Turnbull support has {support_total} intervals; limit is {max_support}"
        )
    if work_total > max_work:
        raise ValueError("combined interval likelihood work estimate exceeds max_work")

    # Baseline intercepts absorb stratum-specific covariate offsets; only the
    # within-stratum centered design identifies shared slopes.
    unit = np.max(np.abs(x), axis=0) if x.shape[1] else np.empty(0)
    if np.any(unit == 0):
        raise ValueError("constant covariates do not identify shared coefficients")
    normalized = x / unit
    within = np.empty_like(normalized)
    group_centers: list[FloatArray] = []
    for rows, *_ in groups:
        group_center = np.mean(normalized[rows], axis=0)
        group_centers.append(group_center)
        within[rows] = normalized[rows] - group_center
    center = np.mean(normalized, axis=0)
    scale_scaled = np.sqrt(np.mean(within * within, axis=0))
    if np.any(scale_scaled == 0) or not np.isfinite(scale_scaled).all():
        raise ValueError("shared covariates are not identified within strata")
    design = within / scale_scaled
    if x.shape[1] and np.linalg.matrix_rank(design) < x.shape[1]:
        raise ValueError("shared covariates are not identified within strata")
    if len(labels) == 1:
        from .interval_survival import fit_interval_survival

        fit = fit_interval_survival(
            lo,
            hi,
            x,
            weights=w,
            tolerance=tol,
            max_iterations=int(max_iterations),
            max_support=max_support,
            max_work=max_work,
        )
        baseline = StratifiedIntervalBaseline(
            labels[0],
            fit.support_lower,
            fit.support_upper,
            fit.support_left_closed,
            fit.support_mass,
            fit.log_cumulative_hazard,
            fit.log_likelihood,
            fit.score_error,
            fit.covariate_mean,
            fit.covariate_center_scaled,
        )
        return StratifiedIntervalSurvivalFit(
            fit.coefficients,
            fit.covariate_mean,
            fit.covariate_scale,
            fit.covariate_unit_scale,
            fit.covariate_center_scaled,
            fit.covariate_scale_scaled,
            fit.scaled_coefficients,
            (baseline,),
            fit.log_likelihood,
            fit.iterations,
            fit.score_error,
            fit.converged,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        mean = center * unit
        scale = scale_scaled * unit
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("covariate center or scale exceeds numerical range")

    opt_w = w / weight_scale
    if not np.isfinite(opt_w).all() or np.any(opt_w <= 0):
        raise ValueError("weight range is too wide for stable stratified optimization")
    q_list: list[FloatArray] = []
    for _, sl, *_ in groups:
        surv = 1 - np.arange(1, sl.size, dtype=np.float64) / (sl.size + 2)
        q_list.append(np.log(-np.log(surv)))
    beta = np.zeros(x.shape[1])
    previous = -np.inf
    error = np.inf
    converged = False

    def terms(qs: list[FloatArray], b: FloatArray, deriv: bool = True):
        total_ll = 0.0
        gb = np.zeros(b.size)
        hb = np.zeros((b.size, b.size))
        for q, (rows, _, _, _, li, ri) in zip(qs, groups, strict=True):
            part = _likelihood_terms(
                q,
                design[rows] @ b,
                li,
                ri,
                q.size + 1,
                opt_w[rows],
                design[rows],
                derivatives=deriv,
            )
            total_ll += part[0]
            if deriv:
                gb += part[1]
                hb += part[3]
        return total_ll, gb, hb

    for iteration in range(1, int(max_iterations) + 1):
        ll, gb, hb = terms(q_list, beta)
        if not np.isfinite(ll) or not np.isfinite(gb).all() or not np.isfinite(hb).all():
            raise ArithmeticError("joint interval PH likelihood became nonfinite")
        info = -0.5 * (hb + hb.T)
        if beta.size:
            eig = np.linalg.eigvalsh(info)
            if eig[0] <= 1e-12:
                raise ValueError(
                    "shared interval PH coefficients are not identified by the joint likelihood"
                )
            step = np.linalg.solve(info, gb)
        else:
            step = np.empty(0)
        fraction = 1.0
        for _ in range(30):
            candidate = beta + fraction * step
            candidate_ll = terms(q_list, candidate, False)[0]
            if np.isfinite(candidate_ll) and candidate_ll >= ll - 1e-10 * (1 + abs(ll)):
                beta = candidate
                break
            fraction *= 0.5
        else:
            if np.max(np.abs(gb), initial=0.0) > tol:
                raise ArithmeticError("joint regression Newton step failed to improve likelihood")

        q_error: list[float] = []
        for index, q in enumerate(q_list):
            rows, _, _, _, li, ri = groups[index]
            part = _likelihood_terms(
                q, design[rows] @ beta, li, ri, q.size + 1, opt_w[rows], design[rows]
            )
            gq, hq = part[2], part[4]
            negative = hq < -np.finfo(float).eps * float(opt_w[rows].sum())
            replacement = float(np.mean(hq[negative])) if np.any(negative) else -1.0
            diagonal = np.where(negative, hq, replacement)
            target = q - gq / diagonal
            direction = _pava_increasing(target, -diagonal) - q
            fraction = 1.0
            for _ in range(30):
                cand = q + fraction * direction
                cand_ll = _likelihood_terms(
                    cand,
                    design[rows] @ beta,
                    li,
                    ri,
                    q.size + 1,
                    opt_w[rows],
                    design[rows],
                    derivatives=False,
                )[0]
                if np.isfinite(cand_ll) and cand_ll >= part[0] - 1e-10 * (1 + abs(part[0])):
                    q_list[index] = cand
                    break
                fraction *= 0.5
            qnew = q_list[index]
            new_gq = _likelihood_terms(
                qnew, design[rows] @ beta, li, ri, qnew.size + 1, opt_w[rows], design[rows]
            )[2]
            projected = (
                _pava_increasing(qnew + new_gq / float(opt_w[rows].sum()), np.ones(qnew.size))
                - qnew
            )
            q_error.append(float(np.max(np.abs(projected), initial=0.0)))
        final_ll, final_gb, _ = terms(q_list, beta)
        error = max(float(np.max(np.abs(final_gb), initial=0.0)) / float(opt_w.sum()), *q_error)
        if (
            np.isfinite(previous)
            and final_ll - previous <= tol * (1 + abs(final_ll))
            and error <= max(5 * tol, 1e-7)
        ):
            converged = True
            break
        previous = final_ll
    if not converged:
        raise ArithmeticError(
            f"stratified interval PH fit did not converge; score error={error:.3g}"
        )

    final_ll, _, _ = terms(q_list, beta, False)
    result_strata = []
    for group_index, (q, (rows, sl, su, closed, li, ri)) in enumerate(
        zip(q_list, groups, strict=True)
    ):
        loglik_part = _likelihood_terms(
            q, design[rows] @ beta, li, ri, q.size + 1, opt_w[rows], design[rows], derivatives=False
        )[0]
        baseline = _baseline(
            labels[group_index],
            sl,
            su,
            closed,
            q,
            float(loglik_part * weight_scale),
            q_error[group_index],
        )
        result_strata.append(
            StratifiedIntervalBaseline(
                baseline.label,
                baseline.support_lower,
                baseline.support_upper,
                baseline.support_left_closed,
                baseline.support_mass,
                baseline.log_cumulative_hazard,
                baseline.log_likelihood,
                baseline.kkt_error,
                _freeze(group_centers[group_index] * unit),
                _freeze(group_centers[group_index]),
            )
        )
    coefficients = (beta / scale_scaled) / unit
    if not np.isfinite(coefficients).all():
        raise ArithmeticError("shared interval PH coefficients exceed numerical range")
    return StratifiedIntervalSurvivalFit(
        _freeze(coefficients),
        _freeze(mean),
        _freeze(scale),
        _freeze(unit),
        _freeze(center),
        _freeze(scale_scaled),
        _freeze(beta),
        tuple(result_strata),
        float(final_ll * weight_scale),
        iteration,
        error,
        converged,
    )


def predict_stratified_interval_survival(
    fit: StratifiedIntervalSurvivalFit,
    stratum: str | int,
    times: ArrayLike | None = None,
    profiles: ArrayLike | None = None,
    *,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
) -> IntervalSurvivalPrediction:
    """Return standard survival identification bounds for one stratum."""
    return predict_interval_survival(
        fit.for_stratum(stratum), times, profiles, max_output_cells=max_output_cells
    )
