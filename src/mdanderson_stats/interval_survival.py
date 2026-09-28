"""Semiparametric proportional-hazards regression for interval-censored data.

The baseline distribution is represented on Turnbull maximal-intersection
support intervals. Event time is not identified inside a non-singleton support
interval, so predictions are returned as survival identification bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar

_MAX_ROWS = 20_000
_MAX_COVARIATES = 100
_MAX_SUPPORT = 2_000
_MAX_WORK = 20_000_000
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_ITERATIONS = 5_000


@dataclass(frozen=True)
class IntervalSurvivalFit:
    """Interval-censored PH fit with Turnbull support and probability jumps."""

    coefficients: FloatArray
    covariate_mean: FloatArray
    covariate_scale: FloatArray
    covariate_unit_scale: FloatArray
    covariate_center_scaled: FloatArray
    covariate_scale_scaled: FloatArray
    support_lower: FloatArray
    support_upper: FloatArray
    support_left_closed: np.ndarray
    support_mass: FloatArray
    log_likelihood: float
    iterations: int
    score_error: float
    converged: bool
    scaled_coefficients: FloatArray
    log_cumulative_hazard: FloatArray


@dataclass(frozen=True)
class IntervalSurvivalPrediction:
    """Pointwise survival identification bounds, not confidence intervals."""

    profile: FloatArray
    times: FloatArray
    survival_lower: FloatArray
    survival_upper: FloatArray
    log_survival_lower: FloatArray
    log_survival_upper: FloatArray


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _interval_data(
    lower: ArrayLike,
    upper: ArrayLike,
    covariates: ArrayLike | None,
    weights: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if any(
        np.iscomplexobj(value) for value in (lower, upper, covariates, weights) if value is not None
    ):
        raise ValueError("interval survival data must be real")
    lo = finite(lower, "lower")
    if np.iscomplexobj(upper):
        raise ValueError("upper must be real")
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 1 or not 1 <= lo.size <= _MAX_ROWS or hi.shape != lo.shape:
        raise ValueError(f"lower and upper must be aligned vectors with 1..{_MAX_ROWS} rows")
    # Infinity is meaningful only as a right-censoring upper endpoint.
    if hi.shape != lo.shape or np.isnan(hi).any() or np.isneginf(hi).any():
        raise ValueError("upper may contain positive infinity for right censoring only")
    if np.any(~np.isfinite(lo)) or np.any(lo < 0) or np.any(lo > hi):
        raise ValueError("lower must be finite, nonnegative, and no greater than upper")
    if np.any((lo == hi) & (lo == 0)):
        raise ValueError("exact event times must be positive")
    x = np.empty((lo.size, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != lo.size or x.shape[1] > _MAX_COVARIATES:
        raise ValueError(
            f"covariates must have one row per observation and at most {_MAX_COVARIATES} columns"
        )
    if x.size > 2_000_000:
        raise ValueError("covariate matrix exceeds the 2000000-cell limit")
    if weights is None:
        w = np.ones(lo.size, dtype=np.float64)
    else:
        w = finite(weights, "weights")
        if w.shape != lo.shape or np.any(w <= 0):
            raise ValueError("weights must be positive and aligned with intervals")
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("weights have an invalid total")
    return lo, hi, x, w


def _maximal_intersections(
    lower: FloatArray, upper: FloatArray
) -> tuple[FloatArray, FloatArray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    endpoints = np.unique(np.r_[lower, upper[np.isfinite(upper)]])
    exact = set(map(float, lower[lower == upper]))
    open_starts = set(map(float, lower[lower < upper]))
    closed_ends = set(map(float, upper[np.isfinite(upper) & (lower < upper)]))
    lefts: list[float] = []
    rights: list[float] = []
    found_left = False
    last_left = -np.inf
    for value in endpoints:
        point = float(value)
        # Exact observations are closed singleton supports. They must be
        # registered before a non-degenerate right endpoint closes at the same
        # time; open starts are activated only after those closures.
        if point in exact:
            found_left = True
            last_left = point
        if found_left and (point in closed_ends or point in exact):
            lefts.append(last_left)
            rights.append(point)
            found_left = False
        if point in open_starts:
            found_left = True
            last_left = point
    if found_left:
        # A right-censored tail is represented by a final support interval.
        lefts.append(last_left)
        rights.append(np.inf)
    if not lefts:
        raise ValueError("intervals do not identify any maximal-intersection support")
    order = np.lexsort((np.asarray(rights), np.asarray(lefts)))
    sl = np.asarray(lefts, dtype=np.float64)[order]
    su = np.asarray(rights, dtype=np.float64)[order]
    # Exact observations identify point masses; all other support intervals
    # have the conventional open-left, closed-right interpretation.
    left_closed = sl == su
    return sl, su, left_closed, _endpoint_indices(lower, upper, sl, su)


def _endpoint_indices(
    lower: FloatArray, upper: FloatArray, support_lower: FloatArray, support_upper: FloatArray
) -> tuple[np.ndarray, np.ndarray]:
    left = np.empty(lower.size, dtype=np.int64)
    right = np.empty(upper.size, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        if lo == hi:
            exact = np.flatnonzero((support_lower == lo) & (support_upper == hi))
            if not exact.size:
                raise ValueError("an exact event has no singleton support interval")
            left[i] = right[i] = int(exact[0])
            continue
        starts = np.flatnonzero((support_lower <= lo) & (support_upper > lo))
        if starts.size:
            left[i] = int(starts[0])
        else:
            later = np.flatnonzero(support_lower > lo)
            left[i] = int(later[0]) if later.size else support_lower.size - 1
        if np.isposinf(hi):
            right[i] = support_lower.size - 1
        else:
            eligible = np.flatnonzero(
                (support_upper <= hi) | ((support_lower < hi) & (support_upper > hi))
            )
            right[i] = int(eligible[-1]) if eligible.size else 0
    if np.any(left > right):
        raise ValueError("an observation interval has no compatible maximal-intersection support")
    return left, right


def _pava_increasing(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted increasing isotonic regression using pooled adjacent blocks."""
    n = values.size
    if n == 0:
        return values.copy()
    means = np.empty(n, dtype=np.float64)
    masses = np.empty(n, dtype=np.float64)
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    blocks = 0
    for i in range(n):
        mass = max(float(weights[i]), np.finfo(np.float64).tiny)
        means[blocks] = values[i]
        masses[blocks] = mass
        starts[blocks] = ends[blocks] = i
        blocks += 1
        while blocks > 1 and means[blocks - 2] > means[blocks - 1]:
            total = masses[blocks - 2] + masses[blocks - 1]
            means[blocks - 2] = (
                means[blocks - 2] * masses[blocks - 2] + means[blocks - 1] * masses[blocks - 1]
            ) / total
            masses[blocks - 2] = total
            ends[blocks - 2] = ends[blocks - 1]
            blocks -= 1
    result = np.empty(n, dtype=np.float64)
    for block in range(blocks):
        result[starts[block] : ends[block] + 1] = means[block]
    return result


def _delta_values(delta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return log(1-exp(-delta)), delta/expm1(delta), h, and h*(1+h)."""
    log_fraction = np.full(delta.shape, -np.inf, dtype=np.float64)
    ratio = np.zeros(delta.shape, dtype=np.float64)
    h = np.zeros(delta.shape, dtype=np.float64)
    hc = np.zeros(delta.shape, dtype=np.float64)
    small = (delta > 0) & (delta < 1e-4)
    moderate = (delta >= 1e-4) & (delta <= 700)
    large = delta > 700
    d = delta[small]
    log_fraction[small] = np.log(d) - d / 2 + d * d / 24 - d**4 / 2880
    ratio[small] = 1 - d / 2 + d * d / 12 - d**4 / 720
    h[small] = 1 / d - 0.5 + d / 12 - d**3 / 720
    hc[small] = 1 / d**2 - 1 / 12 + d**2 / 240 - d**4 / 6048
    d = delta[moderate]
    em1 = np.expm1(d)
    log_fraction[moderate] = np.log(-np.expm1(-d))
    h[moderate] = 1 / em1
    ratio[moderate] = d / em1
    hc[moderate] = h[moderate] * (1 + h[moderate])
    log_fraction[large] = 0.0
    return log_fraction, ratio, h, hc


def _likelihood_terms(
    q: FloatArray,
    eta: FloatArray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    n_support: int,
    weights: FloatArray,
    design: FloatArray,
    *,
    derivatives: bool = True,
) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
    boundaries = np.r_[-np.inf, q, np.inf]
    ql = boundaries[left_index]
    qr = boundaries[right_index + 1]
    log_a = ql + eta
    log_b = qr + eta
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a = np.exp(log_a)
        b = np.exp(log_b)
        delta = b - a
    # For close finite boundaries, calculate the hazard difference without
    # subtracting rounded exponentials.
    close = np.isfinite(ql) & np.isfinite(qr) & ((qr - ql) < 0.5)
    if np.any(close):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            delta[close] = a[close] * np.expm1(qr[close] - ql[close])
    log_fraction, ratio, h, hc = _delta_values(delta)
    log_probability = -a + log_fraction
    if not np.isfinite(log_probability).all():
        return (
            -np.inf,
            np.full(design.shape[1], np.nan),
            np.full(q.size, np.nan),
            np.full((design.shape[1], design.shape[1]), np.nan),
            np.full(q.size, np.nan),
        )
    log_likelihood = float(np.dot(weights, log_probability))
    if not derivatives:
        return (
            log_likelihood,
            np.zeros(design.shape[1]),
            np.zeros(q.size),
            np.zeros((design.shape[1], design.shape[1])),
            np.zeros(q.size),
        )
    eta_score = -a + ratio
    eta_curvature = -a + ratio
    finite_delta = np.isfinite(delta)
    eta_curvature[finite_delta] -= hc[finite_delta] * delta[finite_delta] * delta[finite_delta]
    if not np.isfinite(eta_score).all() or not np.isfinite(eta_curvature).all():
        return (
            -np.inf,
            np.full(design.shape[1], np.nan),
            np.full(q.size, np.nan),
            np.full((design.shape[1], design.shape[1]), np.nan),
            np.full(q.size, np.nan),
        )
    beta_score = design.T @ (weights * eta_score)
    beta_hessian = (design.T * (weights * eta_curvature)) @ design
    q_score = np.zeros(q.size, dtype=np.float64)
    q_hessian = np.zeros(q.size, dtype=np.float64)
    # Boundary index j maps to q[j-1], for internal 1..K-1 only.
    left_finite = (left_index > 0) & np.isfinite(ql)
    if np.any(left_finite):
        a_l = a[left_finite]
        g = -a_l * (1 + h[left_finite])
        hdiag = -a_l * (1 + h[left_finite]) - a_l * a_l * hc[left_finite]
        np.add.at(q_score, left_index[left_finite] - 1, weights[left_finite] * g)
        np.add.at(q_hessian, left_index[left_finite] - 1, weights[left_finite] * hdiag)
    right_boundary = right_index + 1
    right_finite = (right_boundary < n_support) & np.isfinite(qr)
    if np.any(right_finite):
        b_r = b[right_finite]
        g = b_r * h[right_finite]
        hdiag = b_r * h[right_finite] - b_r * b_r * hc[right_finite]
        np.add.at(q_score, right_boundary[right_finite] - 1, weights[right_finite] * g)
        np.add.at(q_hessian, right_boundary[right_finite] - 1, weights[right_finite] * hdiag)
    return log_likelihood, beta_score, q_score, beta_hessian, q_hessian


def _objective(
    q: FloatArray,
    beta: FloatArray,
    design: FloatArray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    n_support: int,
    weights: FloatArray,
    *,
    derivatives: bool = True,
) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
    eta = design @ beta
    return _likelihood_terms(
        q, eta, left_index, right_index, n_support, weights, design, derivatives=derivatives
    )


def _one_step(
    q: FloatArray,
    beta: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    weights: FloatArray,
    design: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    n_support = lower.size
    ll, grad_b, grad_q, hess_b, hess_q = _objective(
        q, beta, design, left_index, right_index, n_support, weights
    )
    if not isfinite(ll):
        raise ArithmeticError("interval PH likelihood is nonfinite at its starting point")
    # Conditional regression Newton step with exact Hessian and likelihood
    # backtracking, matching icenReg's parameter block update.
    if beta.size:
        curvature = -0.5 * (hess_b + hess_b.T)
        try:
            eigmin = float(np.linalg.eigvalsh(curvature)[0])
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("interval PH regression information is singular") from exc
        if not np.isfinite(eigmin) or eigmin <= 1e-12:
            raise ValueError(
                "interval PH regression coefficients are not identified by these intervals"
            )
        try:
            step_b = np.linalg.solve(curvature, grad_b)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("interval PH regression information is singular") from exc
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + scale * step_b
            cand_ll = _objective(
                q, candidate, design, left_index, right_index, n_support, weights, derivatives=False
            )[0]
            if np.isfinite(cand_ll) and cand_ll >= ll - 1e-10 * (1 + abs(ll)):
                beta = candidate
                ll = cand_ll
                accepted = True
                break
            scale *= 0.5
        if not accepted and np.max(np.abs(grad_b), initial=0.0) > tolerance * weights.sum():
            raise ArithmeticError(
                "interval PH regression Newton step failed to improve the likelihood"
            )
    # Isotonic Newton baseline block. Equal adjacent q values are active
    # zero-mass support intervals, which a softmax parametrization cannot reach.
    if q.size:
        ll, _, grad_q, _, hess_q = _objective(
            q, beta, design, left_index, right_index, n_support, weights
        )
        negative = hess_q < -np.finfo(np.float64).eps * weights.sum()
        replacement = float(np.mean(hess_q[negative])) if np.any(negative) else -1.0
        diagonal = np.where(negative, hess_q, replacement)
        target = q - grad_q / diagonal
        projected = _pava_increasing(target, -diagonal)
        direction = projected - q
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = q + scale * direction
            cand_ll = _objective(
                candidate,
                beta,
                design,
                left_index,
                right_index,
                n_support,
                weights,
                derivatives=False,
            )[0]
            if np.isfinite(cand_ll) and cand_ll >= ll - 1e-10 * (1 + abs(ll)):
                q = candidate
                ll = cand_ll
                accepted = True
                break
            scale *= 0.5
        if not accepted and np.max(np.abs(grad_q), initial=0.0) > tolerance * weights.sum():
            raise ArithmeticError("interval PH baseline ICM step failed to improve the likelihood")
    ll, grad_b, grad_q, _, _ = _objective(
        q, beta, design, left_index, right_index, n_support, weights
    )
    if q.size:
        projected_gradient = _pava_increasing(q + grad_q / weights.sum(), np.ones(q.size)) - q
        baseline_error = float(np.max(np.abs(projected_gradient), initial=0.0))
    else:
        baseline_error = 0.0
    score_error = max(
        float(np.max(np.abs(grad_b), initial=0.0)) / weights.sum(),
        baseline_error,
    )
    return q, beta, ll, score_error


def fit_interval_survival(
    lower: ArrayLike,
    upper: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    weights: ArrayLike | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    max_support: int = _MAX_SUPPORT,
    max_work: int = _MAX_WORK,
) -> IntervalSurvivalFit:
    """Fit interval-censored proportional-hazards regression.

    Intervals use ``(lower, upper]`` semantics; ``upper=inf`` is right
    censoring and ``lower=upper>0`` is an exact event. Left censoring is
    represented by lower zero. No covariance or confidence limits are
    estimated because the native semiparametric fit reports covariance only
    through optional bootstrap resampling.
    """
    lo, hi, x, w = _interval_data(lower, upper, covariates, weights)
    if np.isposinf(hi).all():
        raise ValueError("all observations are right-censored; the baseline has no finite maximum")
    tol = scalar(tolerance, "tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or int(max_iterations) != max_iterations
        or max_iterations < 1
        or max_iterations > _MAX_ITERATIONS
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
        or max_work < 1
        or max_work > _MAX_WORK
    ):
        raise ValueError(f"max_work must be in [1, {_MAX_WORK}]")
    sl, su, left_closed, (left_index, right_index) = _maximal_intersections(lo, hi)
    if sl.size > max_support:
        raise ValueError(f"Turnbull support has {sl.size} intervals; limit is {max_support}")
    if lo.size * max(sl.size, 1) > max_work:
        raise ValueError("interval likelihood work estimate exceeds max_work")
    if x.shape[1]:
        unit_scale = np.max(np.abs(x), axis=0)
        if np.any(unit_scale == 0):
            raise ValueError("constant covariates do not have identified regression coefficients")
        x_scaled = x / unit_scale
        center_scaled = np.mean(x_scaled, axis=0)
        scale_scaled = np.std(x_scaled, axis=0)
        if np.any(scale_scaled == 0) or not np.isfinite(scale_scaled).all():
            raise ValueError("constant covariates do not have identified regression coefficients")
        design = (x_scaled - center_scaled) / scale_scaled
        with np.errstate(over="ignore", invalid="ignore"):
            mean = center_scaled * unit_scale
            scale = scale_scaled * unit_scale
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("covariate center or scale exceeds numerical range")
    else:
        unit_scale = center_scaled = scale_scaled = mean = scale = np.empty(0)
        design = np.empty((lo.size, 0))
    if design.shape[1] and np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("interval PH covariates are linearly dependent")
    n_support = sl.size
    # icenReg initializes survival linearly across the interval boundaries;
    # the final support keeps the remaining right-tail probability.
    survival = 1 - np.arange(1, n_support, dtype=np.float64) / (n_support + 2)
    q = np.log(-np.log(survival))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    weight_scale = float(np.max(w))
    opt_weights = w / weight_scale
    ll = -np.inf
    score_error = np.inf
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        previous = ll
        q, beta, ll, score_error = _one_step(
            q, beta, sl, su, left_index, right_index, opt_weights, design, tol
        )
        improvement = ll - previous
        if (
            np.isfinite(previous)
            and improvement <= tol * (1 + abs(ll))
            and score_error <= max(5 * tol, 1e-7)
        ):
            converged = True
            break
    if not converged:
        raise ArithmeticError(
            f"interval PH fit did not converge in {max_iterations} iterations; "
            f"score error={score_error:.3g}"
        )
    boundaries = np.r_[-np.inf, q, np.inf]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        base_log_survival = -np.exp(boundaries)
        base_survival = np.exp(base_log_survival)
    mass = base_survival[:-1] * (-np.expm1(base_log_survival[1:] - base_log_survival[:-1]))
    if np.any(mass < -1e-12) or not np.isfinite(mass).all():
        raise ArithmeticError("interval PH fit produced invalid support probabilities")
    mass = np.maximum(mass, 0.0)
    mass /= mass.sum()
    coefficients = (beta / scale_scaled) / unit_scale if beta.size else np.empty(0)
    if not np.isfinite(coefficients).all():
        raise ArithmeticError("interval PH coefficients exceed numerical range")
    return IntervalSurvivalFit(
        _freeze(coefficients),
        _freeze(mean),
        _freeze(scale),
        _freeze(unit_scale),
        _freeze(center_scaled),
        _freeze(scale_scaled),
        _freeze(sl),
        _freeze(su),
        _readonly_bool(left_closed),
        _freeze(mass),
        float(ll * weight_scale),
        iterations,
        float(score_error),
        converged,
        _freeze(beta),
        _freeze(q),
    )


def _prediction_inputs(
    fit: IntervalSurvivalFit,
    times: ArrayLike | None,
    profiles: ArrayLike | None,
    max_output_cells: int,
) -> tuple[FloatArray, FloatArray]:
    if profiles is None:
        profile = np.asarray(fit.covariate_mean)[None, :]
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile = finite(profiles, "profiles")
        if profile.ndim == 1:
            profile = profile[None, :]
        if profile.ndim != 2 or profile.shape[1] != fit.covariate_mean.size:
            raise ValueError("profiles must have one column per fitted covariate")
    if profile.shape[0] > 100_000 or profile.size > 2_000_000:
        raise ValueError("profiles exceed the interval survival prediction limit")
    if times is None:
        candidates = np.r_[
            fit.support_lower[np.isfinite(fit.support_lower)],
            fit.support_upper[np.isfinite(fit.support_upper)],
        ]
        time_values = np.unique(candidates)
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        time_values = finite(times, "times")
        if time_values.ndim == 0:
            time_values = time_values.reshape(1)
        if time_values.ndim != 1 or time_values.size == 0 or np.any(time_values < 0):
            raise ValueError("times must be a nonempty nonnegative scalar or vector")
    if time_values.size > 100_000:
        raise ValueError("too many prediction times (limit 100000)")
    if 8 * profile.shape[0] * time_values.size + profile.size + time_values.size > max_output_cells:
        raise ValueError("interval survival prediction exceeds the combined output-cell budget")
    return profile, time_values


def predict_interval_survival(
    fit: IntervalSurvivalFit,
    times: ArrayLike | None = None,
    profiles: ArrayLike | None = None,
    *,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
) -> IntervalSurvivalPrediction:
    """Predict PH survival identification bounds at times and covariate profiles."""
    if (
        isinstance(max_output_cells, (bool, np.bool_))
        or int(max_output_cells) != max_output_cells
        or max_output_cells < 1
        or max_output_cells > _MAX_OUTPUT_CELLS
    ):
        raise ValueError(f"max_output_cells must be in [1, {_MAX_OUTPUT_CELLS}]")
    profile, time_values = _prediction_inputs(fit, times, profiles, int(max_output_cells))
    # Maximum event probability by t: all mass in an interval may sit at its
    # earliest admissible point. Minimum event probability: only intervals
    # whose closed right edge has been passed must have occurred.
    boundaries = np.r_[-np.inf, fit.log_cumulative_hazard, np.inf]
    lower_boundary_index = np.searchsorted(fit.support_lower, time_values, side="left")
    at_lower = np.searchsorted(fit.support_lower, time_values, side="left")
    at_upper = np.searchsorted(fit.support_lower, time_values, side="right")
    for j, start in enumerate(at_lower):
        if at_upper[j] > start:
            lower_boundary_index[j] += int(np.sum(fit.support_left_closed[start : at_upper[j]]))
    upper_boundary_index = np.searchsorted(fit.support_upper, time_values, side="right")
    if profile.shape[1]:
        profile_scaled = profile / fit.covariate_unit_scale
        standardized_profile = (
            profile_scaled - fit.covariate_center_scaled
        ) / fit.covariate_scale_scaled
        eta = standardized_profile @ fit.scaled_coefficients
    else:
        eta = np.zeros(profile.shape[0])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_lower = -np.exp(boundaries[lower_boundary_index][None, :] + eta[:, None])
        log_upper = -np.exp(boundaries[upper_boundary_index][None, :] + eta[:, None])
        survival_lower = np.exp(log_lower)
        survival_upper = np.exp(log_upper)
    if np.isnan(log_lower).any() or np.isnan(log_upper).any():
        raise ArithmeticError("interval survival prediction exceeds numerical range")
    if np.any(survival_lower > survival_upper + 1e-12):
        raise ArithmeticError("interval survival bounds are inconsistent")
    return IntervalSurvivalPrediction(
        _freeze(profile),
        _freeze(time_values),
        _freeze(survival_lower),
        _freeze(survival_upper),
        _freeze(log_lower),
        _freeze(log_upper),
    )
