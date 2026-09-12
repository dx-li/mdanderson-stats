"""Posterior monitoring and allocation primitives for BARPO.

The functions here are deliberately stateless: a caller supplies the current
observed counts and assigned counts at each look.  Pending patients therefore
remain in ``assigned`` but do not alter the beta posterior.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincc

from ._cdflib import _freeze
from ._validation import count, finite, scalar
from .arand_posterior import arand_best_probability
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import BetaComparison, compare_beta_binomial


@dataclass(frozen=True)
class BarpoPosterior:
    successes: np.ndarray
    failures: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    best_probability: np.ndarray
    best_probability_error: np.ndarray
    variance: np.ndarray


@dataclass(frozen=True)
class BarpoMonitoring:
    posterior: BarpoPosterior
    futility_probability: np.ndarray | None
    efficacy_probability: np.ndarray | None
    final_efficacy_probability: np.ndarray | None
    futile: np.ndarray | None
    efficacious: np.ndarray | None
    final_efficacious: np.ndarray | None


def _vectors(value: ArrayLike, name: str, k: int, *, integer: bool = False) -> np.ndarray:
    result = count(value, name) if integer else finite(value, name)
    if result.ndim != 1 or result.shape != (k,):
        raise ValueError(f"{name} must have one entry per arm")
    return result


def _prior(prior: ArrayLike) -> np.ndarray:
    p = finite(prior, "prior")
    if p.ndim != 2 or p.shape[1] != 2 or not 1 <= len(p) <= 10 or np.any(p <= 0):
        raise ValueError("prior must have 1 to 10 positive rows of (alpha, beta)")
    with np.errstate(over="ignore"):
        if not np.all(np.isfinite(p.sum(axis=1))):
            raise ValueError("prior shape sums must be finite")
    return p


def barpo_posterior(
    successes: ArrayLike,
    failures: ArrayLike,
    *,
    prior: ArrayLike,
    absolute_tolerance: float = 1e-9,
) -> BarpoPosterior:
    """Return beta posteriors and independent posterior best-arm probabilities."""
    p = _prior(prior)
    k = len(p)
    successes = _vectors(successes, "successes", k, integer=True)
    failures = _vectors(failures, "failures", k, integer=True)
    with np.errstate(over="ignore"):
        posterior = p + np.column_stack((successes, failures))
    if not np.all(np.isfinite(posterior.sum(axis=1))):
        raise ArithmeticError("posterior beta shape sums overflow")
    best = arand_best_probability(posterior, absolute_tolerance=absolute_tolerance)
    alpha, beta = posterior.T
    total = alpha + beta
    variance = alpha / total * beta / total / (total + 1)
    return BarpoPosterior(
        _freeze(successes),
        _freeze(failures),
        _freeze(alpha),
        _freeze(beta),
        best.probability,
        best.absolute_error,
        _freeze(variance),
    )


def _threshold(value: float | None, name: str, *, strict: bool = False) -> float | None:
    if value is None:
        return None
    value = scalar(value, name)
    if not 0 <= value <= 1 or (strict and value in (0, 1)):
        raise ValueError(f"{name} must be in [0,1]" + (" (exclusive)" if strict else ""))
    return value


def barpo_monitor(
    successes: ArrayLike,
    failures: ArrayLike,
    *,
    prior: ArrayLike,
    control: bool = False,
    theta_fut: float | None = None,
    pfut: float | None = None,
    theta_eff: float | None = None,
    peff: float | None = None,
    theta_final: float | None = None,
    pfinal: float | None = None,
    absolute_tolerance: float = 1e-9,
) -> BarpoMonitoring:
    """Compute BARPO futility, early efficacy, and final efficacy probabilities.

    With ``control=True`` arm zero is the control and each other arm uses the
    direct posterior probability ``P(theta_i > theta_0)``.  Without a control,
    reference values are fixed response rates.  The final rule is independent
    of the early efficacy rule.  A threshold pair must be supplied together.
    """
    if not isinstance(control, (bool, np.bool_)):
        raise ValueError("control must be boolean")
    theta_fut = _threshold(theta_fut, "theta_fut")
    theta_eff = _threshold(theta_eff, "theta_eff")
    theta_final = _threshold(theta_final, "theta_final")
    pfut, peff, pfinal = (
        _threshold(x, n) for x, n in ((pfut, "pfut"), (peff, "peff"), (pfinal, "pfinal"))
    )
    if control and any(x is not None for x in (theta_fut, theta_eff, theta_final)):
        raise ValueError("theta thresholds are not used in control monitoring")
    if not control and (theta_fut is None) != (pfut is None):
        raise ValueError("theta_fut and pfut must be supplied together")
    if not control and (theta_eff is None) != (peff is None):
        raise ValueError("theta_eff and peff must be supplied together")
    if not control and (theta_final is None) != (pfinal is None):
        raise ValueError("theta_final and pfinal must be supplied together")

    posterior = barpo_posterior(
        successes, failures, prior=prior, absolute_tolerance=absolute_tolerance
    )
    k = len(posterior.alpha)
    if control and k < 2:
        raise ValueError("control monitoring requires at least two arms")

    control_ordering: list[BetaComparison] | None = None
    if control:
        control_post = BetaBinomialPosterior(posterior.alpha[0], posterior.beta[0])
        comparisons = []
        for i in range(1, k):
            treatment = BetaBinomialPosterior(posterior.alpha[i], posterior.beta[i])
            comparisons.append(
                compare_beta_binomial(
                    control_post, treatment, absolute_tolerance=absolute_tolerance
                )
            )
        control_ordering = comparisons

    def probabilities(theta: float, *, upper: bool) -> np.ndarray:
        if control:
            out = np.zeros(k)
            assert control_ordering is not None
            for i in range(1, k):
                comparison = control_ordering[i - 1]
                out[i] = comparison.treatment_greater if upper else comparison.control_greater
            return out
        if upper:
            return np.asarray(betaincc(posterior.alpha, posterior.beta, theta))
        return np.asarray(betainc(posterior.alpha, posterior.beta, theta))

    fut = probabilities(theta_fut or 0.0, upper=False) if pfut is not None else None
    eff = probabilities(theta_eff or 0.0, upper=True) if peff is not None else None
    final = probabilities(theta_final or 0.0, upper=True) if pfinal is not None else None
    futile = None if fut is None else (fut > pfut)
    efficacious = None if eff is None else (eff >= peff)
    final_efficacious = None if final is None else (final >= pfinal)
    if control:
        if futile is not None:
            futile[0] = False
        if efficacious is not None:
            efficacious[0] = False
        if final_efficacious is not None:
            final_efficacious[0] = False
    return BarpoMonitoring(
        posterior,
        None if fut is None else _freeze(fut),
        None if eff is None else _freeze(eff),
        None if final is None else _freeze(final),
        None if futile is None else _freeze_bool(futile),
        None if efficacious is None else _freeze_bool(efficacious),
        None if final_efficacious is None else _freeze_bool(final_efficacious),
    )


def _freeze_bool(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _floors(raw: np.ndarray, floor: ArrayLike | None, stopped: np.ndarray) -> np.ndarray:
    k = len(raw)
    f = np.zeros(k) if floor is None else _vectors(floor, "minimum_probability", k)
    if np.any(f < 0) or np.sum(f) > 1 or np.any(f[stopped] != 0):
        raise ValueError(
            "minimum probabilities must be nonnegative, feasible, and zero for stopped arms"
        )
    active = ~stopped
    if not np.any(active) or np.sum(f[active]) > 1 + 1e-12:
        raise ValueError("at least one arm must remain eligible")
    result = np.zeros(k)
    remaining = 1.0
    open_arms = active.copy()
    while np.any(open_arms):
        required = f[open_arms].sum()
        if required > remaining + 1e-12:
            raise ValueError("minimum probabilities are infeasible")
        share = raw[open_arms]
        if share.sum() <= 0:
            raise ArithmeticError("allocation weights vanish on eligible arms")
        proposed = remaining * share / share.sum()
        below = proposed < f[open_arms]
        indices = np.flatnonzero(open_arms)
        if not np.any(below):
            result[indices] = proposed
            break
        fixed = indices[below]
        result[fixed] = f[fixed]
        remaining -= result[fixed].sum()
        open_arms[fixed] = False
    if not np.isclose(result.sum(), 1.0, rtol=0, atol=2e-15):
        raise ArithmeticError("allocation floor water-filling failed")
    return result


def barpo_allocation(
    posterior: BarpoPosterior,
    assigned: ArrayLike,
    *,
    method: str = "barcp",
    tau: float = 0.5,
    tau1: float = 1.0,
    max_n: int | None = None,
    target_probability: ArrayLike | None = None,
    stopped: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
) -> np.ndarray:
    """Compute one BARPO randomization vector from a posterior summary.

    ``assigned`` includes pending assignments.  DBCD requires its desired
    target vector explicitly; zero assigned proportions are rejected because
    the published allocation function is undefined there.
    """
    if not isinstance(posterior, BarpoPosterior):
        raise TypeError("posterior must be a BarpoPosterior")
    k = len(posterior.alpha)
    assigned = _vectors(assigned, "assigned", k, integer=True)
    if np.any(assigned < posterior.successes + posterior.failures):
        raise ValueError("assigned counts cannot be below observed counts")
    if stopped is None:
        stopped_array: np.ndarray = np.zeros(k, dtype=bool)
    else:
        raw_stopped = np.asarray(stopped)
        if raw_stopped.dtype.kind != "b":
            raise ValueError("stopped must contain booleans")
        stopped_array = raw_stopped.astype(bool, copy=False)
    if stopped_array.shape != (k,):
        raise ValueError("stopped must have one entry per arm")
    if not np.any(~stopped_array):
        raise ValueError("at least one arm must remain eligible")
    if method not in ("barcp", "barn2n", "barmtv", "dbcd"):
        raise ValueError("method must be barcp, barn2n, barmtv, or dbcd")
    tau, tau1 = scalar(tau, "tau"), scalar(tau1, "tau1")
    if tau < 0 or tau1 < 0:
        raise ValueError("tau and tau1 must be nonnegative")
    if method == "barn2n":
        if max_n is None or isinstance(max_n, bool) or np.ndim(max_n) != 0:
            raise ValueError("max_n must be a positive integer for barn2n")
        max_n_value = scalar(max_n, "max_n")
        if max_n_value <= 0 or int(max_n_value) != max_n_value:
            raise ValueError("max_n must be a positive integer for barn2n")
        if np.sum(assigned) > max_n_value:
            raise ValueError("assigned total cannot exceed max_n")
        exponent = float(np.sum(assigned) / (2 * max_n_value))
    else:
        exponent = tau
    p = np.asarray(posterior.best_probability)
    if np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=2e-7):
        raise ValueError("posterior best probabilities must be a valid partition")
    active = ~stopped_array
    if method in ("barcp", "barn2n", "barmtv") and np.any(
        p[active] <= posterior.best_probability_error[active]
    ):
        raise ArithmeticError("best-arm probability tail is unresolved for allocation")
    if method == "barmtv":
        variance = np.asarray(posterior.variance)
        if np.any(~np.isfinite(variance[active])) or np.any(variance[active] <= 0):
            raise ArithmeticError("posterior variance is invalid for barmtv allocation")
        log_raw = 0.5 * (np.log(p) + np.log(variance) - np.log(assigned + 1))
        log_raw -= np.max(log_raw[active])
        raw = np.zeros(k)
        raw[active] = np.exp(log_raw[active])
    elif method == "dbcd":
        if target_probability is None:
            raise ValueError("target_probability is required for dbcd")
        target = _vectors(target_probability, "target_probability", k)
        if np.any(target <= 0) or not np.isclose(target.sum(), 1, rtol=0, atol=2e-12):
            raise ValueError("target_probability must be positive and sum to one")
        total = float(assigned.sum())
        if total <= 0:
            raise ValueError("dbcd requires at least one assigned patient")
        x = assigned / total
        if np.any(x <= 0):
            raise ValueError("dbcd rejects zero assigned proportions")
        raw = np.zeros(k)
        if tau1 == 0:
            raw[active] = 1.0
        else:
            with np.errstate(divide="ignore", invalid="raise", over="ignore"):
                log_target = np.log(target)
                delta = log_target - np.log(x)
                anchor = np.max(delta[active])
                score = (
                    log_target - log_target[np.argmax(np.where(active, delta, -np.inf))]
                ) + tau * (delta - anchor)
            score -= np.max(score[active])
            raw[active] = np.exp(tau1 * score[active])
    else:
        if np.any(p[active] <= 0):
            raise ArithmeticError("best-arm probability is unresolved for allocation")
        raw = np.zeros(k)
        if exponent == 0:
            raw[active] = 1.0
        else:
            with np.errstate(divide="ignore", invalid="raise", over="ignore"):
                log_raw = exponent * (np.log(p[active]) - np.max(np.log(p[active])))
                raw[active] = np.exp(log_raw)
    if np.any(~np.isfinite(raw)) or raw.sum() <= 0:
        raise ArithmeticError("allocation weights are unresolved")
    return _freeze(_floors(raw, minimum_probability, stopped_array))
