"""Explicit PRT1.0 arithmetic diagnostics; never a replacement scientific default.

The named legacy functions retain recovered branch/projection defects and report
invalid output conditions. The paper APIs in prt.py/prt_fit.py remain distinct.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.special import betaincc

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .prt import _cutoffs, _dose_probabilities, _probability


@dataclass(frozen=True)
class PRTLegacyProjection:
    transformed: FloatArray
    covariance: FloatArray
    maximum_condition_number: float
    valid_probabilities: bool


def prt_legacy_projection(probability: ArrayLike) -> PRTLegacyProjection:
    """Native max-lower/min-upper projection with lower floor only.

    Values above one are retained as diagnostics and flagged invalid. Do not feed
    invalid transformed values into decisions or silently normalize/upper-clip.
    """
    p = finite(probability, "probability")
    if (
        p.ndim != 3
        or not 2 <= p.shape[2] <= 10
        or not 1 <= p.shape[1] <= 10
        or len(p) <= p.shape[2]
    ):
        raise ValueError("require (draws>doses, 1..10 intervals, 2..10 doses)")
    if np.any((p < 0) | (p > 1)) or p.size > 2_000_000:
        raise ValueError("require bounded raw probabilities with at most2000000 cells")
    _, h, k = p.shape
    transformed = np.empty_like(p)
    covariance = np.empty((h, k, k))
    maximum_condition = 0.0
    for interval in range(h):
        x = p[:, interval]
        cov = np.cov(x, rowvar=False, ddof=1)
        covariance[interval] = cov
        condition = float(np.linalg.cond(cov))
        if not np.isfinite(condition) or condition > 1e12:
            raise ArithmeticError("legacy covariance singular/ill-conditioned; no regularization")
        maximum_condition = max(maximum_condition, condition)
        means = {}
        for lo in range(k):
            means[lo, lo] = x[:, lo]
            for hi in range(lo + 1, k):
                try:
                    weights = cho_solve(
                        cho_factor(cov[lo : hi + 1, lo : hi + 1]), np.ones(hi - lo + 1)
                    )
                except np.linalg.LinAlgError as exc:
                    raise ArithmeticError("legacy covariance not positive definite") from exc
                denominator = float(weights.sum())
                if denominator <= 0 or not np.isfinite(denominator):
                    raise ArithmeticError("invalid legacy covariance weights")
                means[lo, hi] = x[:, lo : hi + 1] @ (weights / denominator)
        for dose in range(k):
            transformed[:, interval, dose] = np.maximum(
                0,
                np.maximum.reduce(
                    [
                        np.minimum.reduce([means[lo, hi] for hi in range(dose, k)])
                        for lo in range(dose + 1)
                    ]
                ),
            )
    if np.any(~np.isfinite(transformed)) or np.any(np.diff(transformed, axis=-1) < 0):
        raise ArithmeticError("legacy transformation is nonfinite or not ordered")
    return PRTLegacyProjection(
        _freeze(transformed), _freeze(covariance), maximum_condition, bool(np.all(transformed <= 1))
    )


@dataclass(frozen=True)
class PRTLegacyMoments:
    alpha: FloatArray
    beta: FloatArray
    fallback_applied: np.ndarray
    valid_moments: np.ndarray


def prt_legacy_beta_moments(mean: ArrayLike, standard_deviation: ArrayLike) -> PRTLegacyMoments:
    """Native arithmetic and strictly-negative .05 fallback, including invalid results.

    Zero, infinity and NaN outputs remain diagnostic values. Finite inputs outside
    proper beta moment bounds are flagged even if the fallback produces a beta.
    """
    m, s = finite(mean, "mean"), finite(standard_deviation, "standard_deviation")
    if m.ndim != 1 or m.shape != s.shape or not 1 <= len(m) <= 10 or np.any(s < 0):
        raise ValueError("require matching mean/nonnegative SD vectors of length1..10")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = s * s
        alpha = (m * (1 - m) - v) / (v * (1 - m) / m + v)
        beta = alpha * (1 - m) / m
        fallback = (alpha < 0) | (beta < 0)
        constant = float(np.float32(0.05))
        alpha = np.where(alpha < 0, constant, alpha)
        beta = np.where(beta < 0, constant, beta)
        valid = (
            (m > 0)
            & (m < 1)
            & (v > 0)
            & (v < m * (1 - m))
            & np.isfinite(alpha)
            & np.isfinite(beta)
            & (alpha > 0)
            & (beta > 0)
        )
    fallback = np.array(fallback, copy=True)
    fallback.setflags(write=False)
    valid = np.array(valid, copy=True)
    valid.setflags(write=False)
    return PRTLegacyMoments(_freeze(alpha), _freeze(beta), fallback, valid)


@dataclass(frozen=True)
class PRTLegacyPredictive:
    negligible: float
    excessive: float
    count_probability: FloatArray
    updated_exceedance: FloatArray
    pending: int


def prt_legacy_predictive(
    alpha: float,
    beta: float,
    pending_probability: ArrayLike,
    *,
    target: float = 0.3,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
) -> PRTLegacyPredictive:
    """Fixed patient-mean risk product with strict native category cutoffs.

    The 0/1 native jitter gives invalid probability masses; endpoints are rejected
    here, not silently jittered or normalized. At most10 pending risks are allowed.
    Empty-pending category arithmetic is returned, but native controllers bypass
    it and use branch-specific zero/no-pending conventions.
    """
    a, b, t = (scalar(v, n) for v, n in ((alpha, "alpha"), (beta, "beta"), (target, "target")))
    p = finite(pending_probability, "pending_probability")
    lo, hi = _cutoffs(negligible_cutoff, excessive_cutoff)
    if (
        a <= 0
        or b <= 0
        or not 0 < t < 1
        or p.ndim != 1
        or len(p) > 10
        or np.any((p <= 0) | (p >= 1))
    ):
        raise ValueError(
            "require positive beta parameters, target(0,1), up to10 strictly interior risks"
        )
    mass = np.zeros(len(p) + 1)
    mass[0] = 1.0
    for j, value in enumerate(p):
        mass[1 : j + 2] = mass[1 : j + 2] * (1 - value) + mass[: j + 1] * value
        mass[0] *= 1 - value
    k = np.arange(len(p) + 1)
    exceedance = betaincc(a + k, b + len(p) - k, t)
    if np.any(~np.isfinite(exceedance)) or abs(mass.sum() - 1) > 1e-12:
        raise ArithmeticError("legacy predictive arithmetic failed validity checks")
    return PRTLegacyPredictive(
        float(mass[exceedance < lo].sum()),
        float(mass[exceedance > hi].sum()),
        _freeze(mass),
        _freeze(exceedance),
        len(p),
    )


@dataclass(frozen=True)
class PRTLegacyDecision:
    action: str
    source_dose: int
    rule: str
    warnings: tuple[str, ...] = ()


def _criteria(exceedance: ArrayLike, current_dose: int, epsilon: float, lo: float, hi: float):
    xi = _dose_probabilities(exceedance, "exceedance")
    lo, hi = _cutoffs(lo, hi)
    k = scalar(current_dose, "current_dose")
    eps = scalar(epsilon, "epsilon")
    if k != np.floor(k) or not 0 <= k < len(xi) or not 0 <= eps <= 1:
        raise ValueError("invalid current dose or epsilon")
    return xi, int(k), eps, lo, hi


def prt_legacy_simulation_decision(
    exceedance: ArrayLike,
    distance: ArrayLike,
    current_dose: int,
    pending: int,
    predictive_negligible: float,
    predictive_excessive: float,
    next_excessive: float,
    *,
    epsilon: float = 0.05,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
) -> PRTLegacyDecision:
    """Native late simulation branch, from supplied criteria; screening is separate.

    source_dose retains the native register even on stop. These diagnostic rules
    include complemented cutoffs and defective ineligible-distance initialization.
    """
    xi, k, eps, lo, hi = _criteria(
        exceedance, current_dose, epsilon, negligible_cutoff, excessive_cutoff
    )
    d = finite(distance, "distance")
    m = count(pending, "pending")
    pn, pe, npe = _probability(
        [predictive_negligible, predictive_excessive, next_excessive], "predictive criteria"
    )
    if d.shape != xi.shape or np.any(d < 0) or m.ndim != 0 or m > 1000:
        raise ValueError("invalid distance or pending count")
    chosen, suspend = k, False
    rule = "6"
    warnings = []
    if pn + pe > 1 + 1e-12:
        warnings.append("supplied PN+PE exceeds one; abstract branch diagnostics only")
    if xi[k] >= hi:
        rule = "4"
        if k > 0:
            minimum = min([d[0]] + [d[i] for i in range(k) if xi[i] < hi])
            chosen = max(i for i in range(k) if d[i] == minimum)
    elif xi[k] >= lo or k == len(xi) - 1:
        rule = "5" if xi[k] >= lo else "6.1"
        suspend = bool(m > 0 and pe > 1 - eps)
    elif not m:
        if npe <= 1 - eps:
            chosen = k + 1
        else:
            suspend = True
    elif pn >= eps and npe <= 1 - eps:
        chosen = k + 1
    else:
        suspend = True
    if xi[k] < lo and k < len(xi) - 1 and xi[k + 1] >= hi:
        chosen, suspend = k, False
    if xi[0] > hi:
        return PRTLegacyDecision("stop", chosen, "3", tuple(warnings))
    if rule == "4" and xi[chosen] >= hi:
        warnings.append("native de-escalation selected a dose at/above excessive cutoff")
    action = (
        "suspend"
        if suspend
        else "escalate"
        if chosen > k
        else "deescalate"
        if chosen < k
        else "stay"
    )
    return PRTLegacyDecision(action, chosen, rule, tuple(warnings))


def prt_legacy_final_selection(
    assessed: ArrayLike,
    exceedance: ArrayLike,
    distance: ArrayLike,
    current_dose: int,
    *,
    excessive_cutoff: float = 0.9,
) -> PRTLegacyDecision:
    upper = scalar(excessive_cutoff, "excessive_cutoff")
    xi, k, _, _, hi = _criteria(exceedance, current_dose, 0.05, upper / 2, upper)
    a = count(assessed, "assessed")
    d = finite(distance, "distance")
    if a.shape != xi.shape or d.shape != xi.shape or np.any(d < 0):
        raise ValueError("matching assessed, exceedance and distance vectors required")
    if xi[0] > hi:
        return PRTLegacyDecision("stop", k, "7")
    eligible = np.flatnonzero((a >= 3) & (xi < hi))
    minimum = min([d[0]] + [d[i] for i in eligible if i > 0])
    chosen = k
    for i in eligible:
        if d[i] == minimum:
            chosen = int(i)
    valid = bool(chosen in eligible)
    return PRTLegacyDecision(
        "select" if valid else "legacy_invalid_selection",
        chosen,
        "7",
        () if valid else ("native fallback retained an ineligible current dose",),
    )


@dataclass(frozen=True)
class PRTLegacyScreening:
    infer: bool
    suspend: bool
    reason: str


def prt_legacy_screening(
    assigned: ArrayLike,
    current_assessed: int,
    current_observed: int,
    first_interval: int,
    current_dose: int,
    *,
    final: bool,
) -> PRTLegacyScreening:
    """Recovered pre-inference arithmetic; observed counts any positive interval.

    Supplied summary states are not a patient-history validator. Workflow callers
    must derive them from actual records (events and assessed are distinct).
    """
    a = count(assigned, "assigned")
    values = count(
        [current_assessed, current_observed, first_interval, current_dose], "screening values"
    )
    if a.ndim != 1 or not 2 <= len(a) <= 10 or values[-1] >= len(a) or type(final) is not bool:
        raise ValueError("invalid screening dimensions or final flag")
    assessed, observed, first, k = map(int, values)
    n = int(a[k])
    pending = n - assessed
    if pending >= 10:
        return PRTLegacyScreening(False, True, "ten_pending")
    if n == 3 and observed == 0:
        return PRTLegacyScreening(False, True, "initial_cohort_unobserved")
    if a[0] <= 3 and first == 0:
        return PRTLegacyScreening(False, True, "first_patient_unobserved")
    if n >= 3 and observed == 0:
        return PRTLegacyScreening(False, final, "current_dose_unobserved")
    return PRTLegacyScreening(True, False, "infer")


def prt_legacy_conduct_decision(
    exceedance: ArrayLike,
    assigned: ArrayLike,
    assessed: ArrayLike,
    current_dose: int,
    predictive_negligible: float,
    predictive_excessive: float,
    next_excessive: float,
    *,
    epsilon: float = 0.05,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
) -> PRTLegacyDecision:
    """Native conduct code from supplied criteria/counts, including missing output.

    Rule4 reports de-escalation but specifies no exact lower dose: source_dose
    remains current. Rule6.0 has no matching native text-output branch. Top-dose
    logic is hard-coded to dose10, so out-of-design escalations are flagged.
    """
    xi, k, eps, lo, hi = _criteria(
        exceedance, current_dose, epsilon, negligible_cutoff, excessive_cutoff
    )
    a, c = count(assigned, "assigned"), count(assessed, "assessed")
    pn, pe, npe = _probability(
        [predictive_negligible, predictive_excessive, next_excessive], "predictive criteria"
    )
    if a.shape != xi.shape or c.shape != xi.shape or np.any(c > a):
        raise ValueError("matching assigned/assessed counts with assessed<=assigned required")
    m = int(a[k] - c[k])
    exists = k + 1 < len(xi)
    next_assigned = int(a[k + 1]) if exists else 0
    next_pending = int(a[k + 1] - c[k + 1]) if exists else 0
    warnings = []
    if pn + pe > 1 + 1e-12:
        warnings.append("supplied PN+PE exceeds one; abstract branch diagnostics only")
    if xi[0] > hi:
        code = "3"
    elif xi[k] >= hi:
        code = "4"
    elif xi[k] >= lo:
        code = "5.11" if not m else "5.1" if pe <= 1 - eps else "5.2"
    elif k == 9:
        code = "6.1" if not m else "6.11" if pe <= 1 - eps else "6.12"
    elif m:
        if pn < eps:
            code = "6.4"
        elif next_assigned == 0:
            code = "6.21"
        elif not next_pending:
            code = "6.22"
        else:
            code = "6.2" if npe <= 1 - eps else "6.3"
    else:
        if next_assigned == 0:
            code = "6.0"
        elif not next_pending:
            code = "6.23" if npe <= 1 - eps else "6.33"
        else:
            code = "6.2" if npe <= eps else "6.3"
        if exists and xi[k + 1] > hi:
            code = "6.1001"
    chosen = k
    if code == "3":
        action = "stop"
    elif code == "4":
        action = "deescalate"
        warnings.append("native conduct does not specify an exact lower dose")
    elif code in ("5.11", "5.1", "6.1", "6.11", "6.1001"):
        action = "stay"
    elif code in ("5.2", "6.12", "6.3", "6.33", "6.4"):
        action = "suspend"
    elif code == "6.0":
        action = "legacy_no_output"
        warnings.append("native code6.0 has no matching decision-text output branch")
    else:
        chosen = k + 1
        action = "escalate"
        if chosen >= len(xi):
            action = "legacy_invalid_escalation"
            warnings.append("native hard-coded dose10 rule escalates outside the configured design")
    return PRTLegacyDecision(action, chosen, code, tuple(warnings))
