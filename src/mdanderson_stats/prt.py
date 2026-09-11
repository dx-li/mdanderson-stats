"""Predicted-risk toxicity criteria and published PRT cohort decision rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaincc, log_ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar


def _cutoffs(lower: float, upper: float) -> tuple[float, float]:
    lo, hi = scalar(lower, "negligible_cutoff"), scalar(upper, "excessive_cutoff")
    if not 0 < lo < hi < 1:
        raise ValueError("require 0 < negligible_cutoff < excessive_cutoff < 1")
    return lo, hi


def _probability(value: ArrayLike, name: str) -> FloatArray:
    p = finite(value, name)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return p


def prt_conditional_toxicity(beta: ArrayLike) -> FloatArray:
    """Raw probit-hazard risk after 0,...,H completed intervals, ending in zero.

    beta has final axes (H assessment intervals, doses). Leading draw axes are
    preserved. This is equation (2.1)'s model, BEFORE Bayesian isotonic projection.
    """
    b = finite(beta, "beta")
    if b.ndim < 2 or min(b.shape[-2:]) < 1:
        raise ValueError("beta must have interval and dose axes")
    log_survival = log_ndtr(-b)
    remaining = np.cumsum(log_survival[..., ::-1, :], axis=-2)[..., ::-1, :]
    risk = -np.expm1(remaining)
    return _freeze(np.concatenate((risk, np.zeros((*b.shape[:-2], 1, b.shape[-1]))), axis=-2))


def prt_interval_loglikelihood(
    beta: ArrayLike, survived: ArrayLike, events: ArrayLike
) -> FloatArray:
    """Probit discrete-hazard log likelihood from interval-by-dose event/survival counts.

    survived counts patients completing that interval without toxicity. Events
    count patients with toxicity in that interval. A pending, incomplete interval
    contributes neither count. Leading beta axes may contain posterior draws.
    """
    b = finite(beta, "beta")
    s, e = count(survived, "survived"), count(events, "events")
    if (
        b.ndim < 2
        or s.ndim != 2
        or s.shape != e.shape
        or b.shape[-2:] != s.shape
        or min(s.shape) < 1
    ):
        raise ValueError("counts must match beta's final interval-by-dose axes")
    positive = np.multiply(log_ndtr(b), e, out=np.zeros_like(b), where=e > 0)
    negative = np.multiply(log_ndtr(-b), s, out=np.zeros_like(b), where=s > 0)
    return _freeze(np.sum(positive + negative, axis=(-2, -1)))


@dataclass(frozen=True)
class PRTPredictiveRisk:
    posterior_mean: float
    posterior_variance: float
    beta_alpha: float
    beta_beta: float
    posterior_excessive: float
    predictive_negligible: float
    predictive_acceptable: float
    predictive_excessive: float
    decision_negligible: float
    decision_excessive: float
    count_probability: FloatArray
    updated_exceedance: FloatArray
    pending: int
    draws: int


def prt_predictive_risk(
    total_probability: ArrayLike,
    pending_probability: ArrayLike,
    *,
    target: float = 0.3,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
    chunk_size: int = 256,
) -> PRTPredictiveRisk:
    """Compute equations (4.1)-(4.3) for one dose from aligned isotonic posterior draws.

    total_probability is (draws,), pending_probability is (draws,pending patients).
    Each pending column contains that patient's conditional remaining toxicity
    probability. Shared rows preserve posterior-induced dependence between patients.
    Counts are marginalized by dynamic programming, not 2**pending enumeration.
    """
    total = _probability(total_probability, "total_probability")
    future = _probability(pending_probability, "pending_probability")
    if total.ndim != 1 or len(total) < 2 or future.ndim != 2 or len(future) != len(total):
        raise ValueError(
            "require >=2 total-risk draws and aligned (draws,pending) conditional risks"
        )
    if future.shape[1] > 1000:
        raise ValueError("at most 1000 pending patients are supported")
    lo, hi = _cutoffs(negligible_cutoff, excessive_cutoff)
    target = scalar(target, "target")
    chunk = scalar(chunk_size, "chunk_size")
    if not 0 < target < 1 or chunk != np.floor(chunk) or not 1 <= chunk <= 10000:
        raise ValueError("target must be in (0,1); chunk_size an integer in [1,10000]")
    mean = float(total.mean())
    variance = float(np.mean((total - mean) ** 2))
    if not 0 < mean < 1 or not 0 < variance < mean * (1 - mean):
        raise ValueError(
            "posterior moments must identify a proper nondegenerate beta approximation"
        )
    concentration = mean * (1 - mean) / variance - 1
    alpha, beta = mean * concentration, (1 - mean) * concentration
    if not np.all(np.isfinite([alpha, beta])) or min(alpha, beta) <= 0:
        raise ArithmeticError("moment-matched beta parameters are not representable")
    m = future.shape[1]
    count_probability = np.zeros(m + 1)
    for start in range(0, len(total), int(chunk)):
        probabilities = future[start : start + int(chunk)]
        mass = np.zeros((len(probabilities), m + 1))
        mass[:, 0] = 1
        for j in range(m):
            p = probabilities[:, j, None]
            mass[:, 1 : j + 2] = mass[:, 1 : j + 2] * (1 - p) + mass[:, : j + 1] * p
            mass[:, 0] *= 1 - p[:, 0]
        count_probability += mass.sum(axis=0) / len(total)
    if abs(count_probability.sum() - 1) > 1e-10:
        raise ArithmeticError("predictive count probabilities failed the mass check")
    successes = np.arange(m + 1)
    exceedance = betaincc(alpha + successes, beta + m - successes, target)
    if np.any(~np.isfinite(exceedance)):
        raise ArithmeticError("predictive beta exceedance failed")
    negligible, excessive = exceedance <= lo, exceedance >= hi
    pn = float(count_probability[negligible].sum())
    pe = float(count_probability[excessive].sum())
    pa = float(count_probability[~(negligible | excessive)].sum())
    return PRTPredictiveRisk(
        mean,
        variance,
        alpha,
        beta,
        float(np.mean(total > target)),
        pn,
        pa,
        pe,
        1.0 if m == 0 else pn,
        0.0 if m == 0 else pe,
        _freeze(count_probability),
        _freeze(exceedance),
        m,
        len(total),
    )


@dataclass(frozen=True)
class PRTDecision:
    action: str
    dose: int | None
    rule: str


def _dose_probabilities(value: ArrayLike, name: str) -> FloatArray:
    p = _probability(value, name)
    if p.ndim != 1 or not 2 <= len(p) <= 10 or np.any(np.diff(p) < 0):
        raise ValueError(f"{name} must have 2 to 10 nondecreasing dose probabilities")
    return p


def prt_decision(
    exceedance: ArrayLike,
    predictive_negligible: ArrayLike,
    predictive_excessive: ArrayLike,
    pending: ArrayLike,
    current_dose: int,
    *,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
    epsilon: float = 0.05,
) -> PRTDecision:
    """Published rules 3-6, applied at cohort decisions. Dose indices are zero-based.

    exceedance is posterior Pr(isotonic total toxicity > target), not predictive
    PE. Pending-zero doses override predictive criteria to PN=1 and PE=0.
    """
    xi = _dose_probabilities(exceedance, "exceedance")
    pn = _probability(predictive_negligible, "predictive_negligible")
    pe = _probability(predictive_excessive, "predictive_excessive")
    m = count(pending, "pending")
    if (
        pn.shape != xi.shape
        or pe.shape != xi.shape
        or m.shape != xi.shape
        or np.any(pn + pe > 1 + 1e-12)
    ):
        raise ValueError("criteria/pending must match doses, with PN+PE <= 1")
    lo, hi = _cutoffs(negligible_cutoff, excessive_cutoff)
    eps, current = scalar(epsilon, "epsilon"), scalar(current_dose, "current_dose")
    if not 0 <= eps <= 1 or current != np.floor(current) or not 0 <= current < len(xi):
        raise ValueError("epsilon must be in [0,1] and current_dose a valid zero-based index")
    k = int(current)
    pn, pe = np.where(m == 0, 1, pn), np.where(m == 0, 0, pe)
    if xi[0] > hi:
        return PRTDecision("stop", None, "3")
    if xi[k] > hi:
        dose = int(np.flatnonzero(xi[:k] <= hi)[-1])
        return PRTDecision("deescalate", dose, "4")
    if xi[k] >= lo:
        return (
            PRTDecision("stay", k, "5.1") if pe[k] <= eps else PRTDecision("suspend", None, "5.2")
        )
    if k == len(xi) - 1 or xi[k + 1] > hi:
        return (
            PRTDecision("stay", k, "6.1") if pe[k] <= eps else PRTDecision("suspend", None, "6.1")
        )
    if pn[k] < 1 - eps:
        return PRTDecision("suspend", None, "6.4")
    if pe[k + 1] > eps:
        return PRTDecision("suspend", None, "6.3")
    return PRTDecision("escalate", k + 1, "6.2")


def prt_final_selection(
    exceedance: ArrayLike,
    posterior_mean: ArrayLike,
    *,
    target: float = 0.3,
    excessive_cutoff: float = 0.9,
) -> int | None:
    """Rule 7: closest posterior mean among doses not excessively toxic; lower-dose ties."""
    xi = _dose_probabilities(exceedance, "exceedance")
    means = _dose_probabilities(posterior_mean, "posterior_mean")
    t, upper = scalar(target, "target"), scalar(excessive_cutoff, "excessive_cutoff")
    if xi.shape != means.shape or not 0 < t < 1 or not 0 < upper < 1:
        raise ValueError(
            "matching dose vectors and target/cutoff strictly between zero and one required"
        )
    eligible = np.flatnonzero(xi <= upper)
    return int(eligible[np.argmin(abs(means[eligible] - t))]) if len(eligible) else None
