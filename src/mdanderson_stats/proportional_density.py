"""Shen–Qin–Costantino proportional-density estimation with right censoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit, logsumexp, ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .expsurv import exploratory_survival
from .survan_km import survan_km


@dataclass(frozen=True)
class ProportionalDensityFit:
    """Arm columns are control (0), treatment (1); masses share unique times.

    Incidence is the susceptible fraction, not the cure fraction. Conditional
    disease curves require adequate follow-up to identify that fraction.
    Bootstrap calibration/uncertainty is not supplied by this estimator.
    """

    alpha: float
    beta: float
    alpha_star: float
    incidence: FloatArray
    incidence_standard_error: FloatArray
    incidence_pvalue: float | None
    likelihood_ratio: float
    likelihood_pvalue: float | None
    equal_censoring: bool
    time: FloatArray
    observed_mass: FloatArray
    disease_mass: FloatArray
    disease_survival: FloatArray
    nonparametric_survival: FloatArray
    censoring_survival: FloatArray
    goodness_of_fit: float
    iterations: int
    score_error: float


def proportional_density_pepe(
    time: ArrayLike, estimated_survival: ArrayLike, nonparametric_survival: ArrayLike
) -> float:
    """Archive's right-endpoint squared-curve area, with unit weight.

    Times must be unique; both survival vectors must be nonincreasing in time.
    This is a statistic, not a calibrated test or a step-function integral.
    """
    arrays = (time, estimated_survival, nonparametric_survival)
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError("times and survival probabilities must be real")
    t, s, q = (finite(a, "curve") for a in arrays)
    if t.ndim != 1 or not t.size or s.shape != t.shape or q.shape != t.shape:
        raise ValueError("require three nonempty vectors of the same length")
    order = np.argsort(t, kind="stable")
    t, s, q = t[order], s[order], q[order]
    if t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be nonnegative and unique")
    if any(np.any((a < 0) | (a > 1)) or np.any(np.diff(a) > 0) for a in (s, q)):
        raise ValueError("survival must be nonincreasing and in [0,1]")
    return float(np.dot(np.diff(np.r_[0.0, t]), (s - q) ** 2))


def _fit_offset(
    time: FloatArray, group: FloatArray, offset: FloatArray
) -> tuple[float, float, FloatArray, int, float]:
    """Two-parameter convex likelihood; centered/scaled Newton with backtracking."""
    lo, hi = float(time.min()), float(time.max())
    if lo == hi:
        raise ValueError("failure times must vary to identify beta")
    a, b = time[group == 0], time[group == 1]
    if a.max() <= b.min() or b.max() <= a.min():
        raise ValueError("separated failure times: no finite proportional-density fit")
    center = lo / 2 + hi / 2
    scale = max(hi - center, center - lo)
    x = np.column_stack((np.ones(time.size), (time - center) / scale))
    coef = np.zeros(2)
    minority_count = min(np.sum(group == 0), np.sum(group == 1))
    for iteration in range(1, 101):
        eta = offset + x @ coef
        p, complement = expit(eta), expit(-eta)
        score = x.T @ np.where(group == 0, p, -complement) / time.size
        error = float(np.max(np.abs(score)) * time.size / minority_count)
        if error <= 1e-12:
            beta = float(coef[1] / scale)
            alpha = float(coef[0] - coef[1] * (center / scale))
            if not np.isfinite(alpha) or not np.isfinite(beta):
                raise ArithmeticError("coefficients exceed floating-point range; rescale time")
            return alpha, beta, eta, iteration, error
        information = (x.T * (p * complement)) @ x / time.size
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("singular proportional-density information") from exc
        loss = float(np.mean(np.logaddexp(0, (1 - 2 * group) * eta)))
        decrease = float(score @ step)
        if not np.all(np.isfinite(step)) or decrease <= 0:
            raise ArithmeticError("unstable proportional-density Newton step")
        fraction = 1.0
        for _ in range(50):
            trial = coef - fraction * step
            trial_eta = offset + x @ trial
            trial_loss = float(np.mean(np.logaddexp(0, (1 - 2 * group) * trial_eta)))
            if trial_loss <= loss - 1e-4 * fraction * decrease + 4 * np.finfo(float).eps:
                coef = trial
                break
            fraction *= 0.5
        else:
            raise ArithmeticError("proportional-density line search did not converge")
    raise ArithmeticError("proportional-density score did not converge in 100 iterations")


def proportional_density(
    time: ArrayLike,
    event: ArrayLike,
    treatment: ArrayLike,
    *,
    equal_censoring: bool = False,
) -> ProportionalDensityFit:
    """Fit g_c(t)/f_c(t) = exp(alpha_star + beta*t), event=1, censor=0.

    Independent censoring within each arm is assumed. Set equal_censoring=True
    only when a common censoring distribution is justified: this pools reverse
    KM estimates, sets the offset to zero and enables the asymptotic LR p-value.
    Default unequal-censoring estimation returns no LR p-value; calibration
    needs a bootstrap. Zero censoring support at any failure time is rejected.
    """
    arrays = (time, event, treatment)
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError("time, event and treatment must be real")
    t, d, z = (finite(a, name) for a, name in zip(arrays, ("time", "event", "treatment")))
    if t.ndim != 1 or not 4 <= t.size <= 1_000_000 or d.shape != t.shape or z.shape != t.shape:
        raise ValueError("require equal-length vectors with 4 to 1,000,000 observations")
    if np.any(t < 0) or np.any((d != 0) & (d != 1)) or np.any((z != 0) & (z != 1)):
        raise ValueError("time must be nonnegative; event and treatment must be 0 or 1")
    if not isinstance(equal_censoring, bool):
        raise ValueError("equal_censoring must be boolean")
    counts = np.array([np.sum((z == j) & (d == 1)) for j in (0, 1)])
    if np.any(counts == 0):
        raise ValueError("each arm must have observed failures")
    selected = np.flatnonzero(d == 1)
    selected = selected[np.argsort(t[selected], kind="stable")]
    failures, group = t[selected], z[selected]
    unique, starts = np.unique(failures, return_index=True)
    incidence, se = np.empty(2), np.empty(2)
    nonparametric, h = np.empty((unique.size, 2)), np.empty((failures.size, 2))
    pooled = exploratory_survival(t, 1 - d) if equal_censoring else None
    for j in (0, 1):
        mask = z == j
        km = survan_km(t[mask], d[mask])
        incidence[j], se[j] = 1 - km.survival[-1], km.standard_error[-1]
        index = np.searchsorted(km.time, unique, side="right") - 1
        survival = np.where(index < 0, 1.0, km.survival[np.maximum(index, 0)])
        nonparametric[:, j] = np.clip((survival - (1 - incidence[j])) / incidence[j], 0, 1)
        censor = pooled if pooled is not None else exploratory_survival(t[mask], 1 - d[mask])
        h[:, j] = censor.at(failures)
    if np.any(h <= 0):
        raise ValueError("censoring survival is zero at a pooled failure time; tail unsupported")
    log_h = np.log(h)
    offset = np.log(float(counts[1]) / counts[0]) + log_h[:, 1] - log_h[:, 0]
    alpha, beta, eta, iterations, error = _fit_offset(failures, group, offset)
    # Logistic complements avoid exp(alpha + beta*t + psi) overflow.
    log_observed = np.column_stack((-np.logaddexp(0, eta), -np.logaddexp(0, -eta)))
    log_observed -= np.log(counts)
    log_disease = log_observed - log_h
    normalizers = logsumexp(log_disease, axis=0)
    disease = np.exp(log_disease - normalizers)
    observed = np.exp(log_observed)
    disease_mass = np.add.reduceat(disease, starts, axis=0)
    observed_mass = np.add.reduceat(observed, starts, axis=0)
    # Reverse accumulation retains small upper-tail survival probabilities.
    survival = np.zeros_like(disease_mass)
    survival[:-1] = np.cumsum(disease_mass[:0:-1], axis=0)[::-1]
    survival = np.minimum(survival, 1)
    signed = 1 - 2 * group
    lr = max(
        0.0, float(2 * np.sum(np.logaddexp(0, signed * offset) - np.logaddexp(0, signed * eta)))
    )
    difference_se = float(np.hypot(*se))
    incidence_p = (
        float(2 * ndtr(-abs(incidence[0] - incidence[1]) / difference_se))
        if difference_se > 0
        else None
    )
    return ProportionalDensityFit(
        alpha,
        beta,
        float(alpha + normalizers[0] - normalizers[1]),
        _freeze(incidence),
        _freeze(se),
        incidence_p,
        lr,
        float(2 * ndtr(-np.sqrt(lr))) if equal_censoring else None,
        equal_censoring,
        _freeze(unique),
        _freeze(observed_mass),
        _freeze(disease_mass),
        _freeze(survival),
        _freeze(nonparametric),
        _freeze(h[starts]),
        proportional_density_pepe(unique, survival[:, 0], nonparametric[:, 0]),
        iterations,
        error,
    )
