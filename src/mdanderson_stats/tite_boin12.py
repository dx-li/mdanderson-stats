"""Approximated-likelihood TITE-BOIN12 calculations.

This module accepts patient-level endpoint histories.  Pending endpoints are
represented by ``-1`` and are converted to effective binomial observations
using their follow-up fractions.  The implementation deliberately stops at
interim conduct; final OBD selection and BDA are separate methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import scalar
from .boin12 import BOIN12Design, BOIN12Posterior

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, dtype: type[np.generic] = np.float64) -> NDArray:
    out = np.asarray(value, dtype=dtype).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class TITEBOIN12Posterior:
    N: IntArray
    observed_events: IntArray
    pending_counts: IntArray
    ESS: FloatArray
    MLE: FloatArray
    patient_conditional_means: FloatArray
    expected_joint_counts: FloatArray
    posterior: BOIN12Posterior
    admissible: NDArray[np.bool_]


@dataclass(frozen=True)
class TITEBOIN12Decision:
    action: str
    next_dose: int | None
    eliminated: NDArray[np.bool_]
    pending_counts: IntArray
    posterior: TITEBOIN12Posterior | None


def _inputs(
    doses: ArrayLike,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    toxicity_followup: ArrayLike,
    efficacy_followup: ArrayLike,
    toxicity_window: float,
    efficacy_window: float,
    n_doses: int,
) -> tuple[IntArray, IntArray, IntArray, FloatArray, FloatArray, float, float, int]:
    if isinstance(n_doses, (bool, np.bool_)) or not isinstance(n_doses, (int, np.integer)):
        raise ValueError("n_doses must be an integer")
    k = int(n_doses)
    if not 1 <= k <= 100:
        raise ValueError("n_doses must lie in [1,100]")
    d = np.asarray(doses)
    t = np.asarray(toxicity)
    e = np.asarray(efficacy)
    tf = np.asarray(toxicity_followup, dtype=float)
    ef = np.asarray(efficacy_followup, dtype=float)
    if any(x.ndim != 1 for x in (d, t, e, tf, ef)) or not (
        d.size == t.size == e.size == tf.size == ef.size
    ):
        raise ValueError("patient arrays must be matching one-dimensional arrays")
    if d.size > 1000:
        raise ValueError("at most 1000 patients are supported")
    if np.any(~np.isfinite(d)) or np.any(d != np.floor(d)) or np.any((d < 1) | (d > k)):
        raise ValueError("doses must be integer values in [1,n_doses]")
    if np.any(~np.isin(t, (-1, 0, 1))) or np.any(~np.isin(e, (-1, 0, 1))):
        raise ValueError("endpoint outcomes must be -1, 0, or 1")
    tw = scalar(toxicity_window, "toxicity_window")
    ew = scalar(efficacy_window, "efficacy_window")
    if tw <= 0 or ew <= 0:
        raise ValueError("assessment windows must be positive")
    if np.any(~np.isfinite(tf)) or np.any(~np.isfinite(ef)) or np.any(tf < 0) or np.any(ef < 0):
        raise ValueError("follow-up times must be finite and nonnegative")
    for y, f, window, name in ((t, tf, tw, "toxicity"), (e, ef, ew, "efficacy")):
        pending = y == -1
        if np.any(f[pending] >= window):
            raise ValueError(f"pending {name} follow-up must be strictly below its window")
        if np.any((y == 0) & (f < window)):
            raise ValueError(f"completed non-event {name} follow-up must reach its window")
    return d.astype(np.int64), t.astype(np.int64), e.astype(np.int64), tf, ef, tw, ew, k


def _endpoint(
    y: IntArray, f: FloatArray, doses: IntArray, window: float, k: int
) -> tuple[IntArray, IntArray, IntArray, IntArray, FloatArray, FloatArray, FloatArray]:
    n: IntArray = np.bincount(doses, minlength=k + 1)[1:].astype(np.int64)
    observed: IntArray = np.bincount(doses[y >= 0], minlength=k + 1)[1:].astype(np.int64)
    events: IntArray = np.bincount(doses[y == 1], minlength=k + 1)[1:].astype(np.int64)
    pending: IntArray = np.bincount(doses[y == -1], minlength=k + 1)[1:].astype(np.int64)
    weights = np.zeros(y.size, dtype=float)
    pending_mask = y == -1
    weights[pending_mask] = f[pending_mask] / window
    ess = np.bincount(doses, weights=np.where(y >= 0, 1.0, weights), minlength=k + 1)[1:]
    failure = np.bincount(doses, weights=np.where(y == 0, 1.0, weights), minlength=k + 1)[1:]
    mle = np.full(k, np.nan)
    treated = n > 0
    valid = treated & (ess > 0)
    mle[valid] = events[valid] / ess[valid]
    return n, observed, events, pending, ess, failure, mle


def tite_boin12_posterior(
    design: BOIN12Design,
    doses: ArrayLike,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    toxicity_followup: ArrayLike,
    efficacy_followup: ArrayLike,
    *,
    toxicity_window: float,
    efficacy_window: float,
    n_doses: int,
) -> TITEBOIN12Posterior:
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    ct = scalar(design.toxicity_cutoff, "toxicity_cutoff")
    ce = scalar(design.efficacy_cutoff, "efficacy_cutoff")
    if not 0 < ct < 1 or not 0 < ce < 1:
        raise ValueError("toxicity_cutoff and efficacy_cutoff must lie in (0,1)")
    d, t, e, tf, ef, tw, ew, k = _inputs(
        doses,
        toxicity,
        efficacy,
        toxicity_followup,
        efficacy_followup,
        toxicity_window,
        efficacy_window,
        n_doses,
    )
    nt, _, yt, pt, et, qt, mt = _endpoint(t, tf, d, tw, k)
    ne, _, ye, pe, ee, qe, me = _endpoint(e, ef, d, ew, k)
    if np.any((nt > 0) & (et <= 0)) or np.any((ne > 0) & (ee <= 0)):
        raise ValueError("a treated endpoint has zero effective sample size")
    n: IntArray = np.bincount(d, minlength=k + 1)[1:].astype(np.int64)
    cond = np.full((d.size, 2), np.nan)
    cond_zero = np.full((d.size, 2), np.nan)
    for j, (y, f, window, p, q) in enumerate(((t, tf, tw, mt, qt), (e, ef, ew, me, qe))):
        pending = y == -1
        p_i = p[d[pending] - 1]
        endpoint_ess = (et if j == 0 else ee)[d[pending] - 1]
        q_i = q[d[pending] - 1] / endpoint_ess
        w_i = f[pending] / window
        denom = q_i + p_i * (1.0 - w_i)
        cond[pending, j] = p_i * (1.0 - w_i) / denom
        cond_zero[pending, j] = q_i / denom
        cond[~pending, j] = y[~pending]
        cond_zero[~pending, j] = 1.0 - y[~pending]
    pt_i, pe_i = cond[:, 0], cond[:, 1]
    pt_zero, pe_zero = cond_zero[:, 0], cond_zero[:, 1]
    cells: FloatArray = np.zeros((k, 4), dtype=float)
    for j, values in enumerate((pt_zero * pe_i, pt_zero * pe_zero, pt_i * pe_i, pt_i * pe_zero)):
        np.add.at(cells[:, j], d - 1, values)
    utilities = np.asarray(design.utilities, dtype=float)
    x = cells @ utilities / 100.0
    alpha = 1.0 + x
    beta = 1.0 + (n - x)
    utility_mean = 100.0 * alpha / (alpha + beta)
    u = (
        utilities[0] * (1 - design.toxicity_limit) * design.efficacy_limit
        + utilities[1] * (1 - design.toxicity_limit) * (1 - design.efficacy_limit)
        + utilities[2] * design.toxicity_limit * design.efficacy_limit
        + utilities[3] * design.toxicity_limit * (1 - design.efficacy_limit)
    )
    benchmark = u + (100 - u) / 2
    util_prob = betaincc(alpha, beta, benchmark / 100.0)
    tox_over = betaincc(1 + yt, 1 + qt, design.toxicity_limit)
    eff_futile = betainc(1 + ye, 1 + qe, design.efficacy_limit)
    posterior = BOIN12Posterior(
        _readonly(tox_over),
        _readonly(eff_futile),
        _readonly(utility_mean),
        _readonly(util_prob),
        _readonly(x),
    )
    admissible = (tox_over < ct) & (eff_futile < ce)
    if not np.all(np.isfinite(np.concatenate((tox_over, eff_futile, util_prob)))):
        raise ValueError("posterior probabilities are not finite")
    pending_both = np.column_stack((pt, pe)).astype(np.int64)
    events_both = np.column_stack((yt, ye)).astype(np.int64)
    ess_both = np.column_stack((et, ee))
    mle_both = np.column_stack((mt, me))
    return TITEBOIN12Posterior(
        _readonly(n, np.int64),
        _readonly(events_both, np.int64),
        _readonly(pending_both, np.int64),
        _readonly(ess_both),
        _readonly(mle_both),
        _readonly(cond),
        _readonly(cells),
        posterior,
        _readonly(admissible, np.bool_),
    )


def tite_boin12_decision(
    design: BOIN12Design,
    doses: ArrayLike,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    toxicity_followup: ArrayLike,
    efficacy_followup: ArrayLike,
    *,
    toxicity_window: float,
    efficacy_window: float,
    n_doses: int,
    current_dose: int,
    eliminated: ArrayLike | None = None,
    max_pending_toxicity: float = 0.5,
    max_pending_efficacy: float = 0.5,
) -> TITEBOIN12Decision:
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    ct = scalar(design.toxicity_cutoff, "toxicity_cutoff")
    ce = scalar(design.efficacy_cutoff, "efficacy_cutoff")
    if not 0 < ct < 1 or not 0 < ce < 1:
        raise ValueError("toxicity_cutoff and efficacy_cutoff must lie in (0,1)")
    current_value = scalar(current_dose, "current_dose")
    if current_value != int(current_value):
        raise ValueError("current_dose must be an integer dose index in [1,n_doses]")
    d, t, e, tf, ef, tw, ew, k = _inputs(
        doses,
        toxicity,
        efficacy,
        toxicity_followup,
        efficacy_followup,
        toxicity_window,
        efficacy_window,
        n_doses,
    )
    if not 1 <= int(current_value) <= k:
        raise ValueError("current_dose must be an integer dose index in [1,n_doses]")
    prior = np.zeros(k, dtype=bool) if eliminated is None else np.asarray(eliminated)
    if prior.shape != (k,) or not np.all(np.isin(prior, (False, True, 0, 1))):
        raise ValueError("eliminated must match n_doses")
    prior = prior.astype(bool)
    nt, _, _, pt, et, _, _ = _endpoint(t, tf, d, tw, k)
    ne, _, _, pe, ee, _, _ = _endpoint(e, ef, d, ew, k)
    current = int(current_value) - 1
    if np.all(prior):
        return TITEBOIN12Decision(
            "stop_safety",
            None,
            _readonly(prior, np.bool_),
            _readonly(np.column_stack((pt, pe)), np.int64),
            None,
        )
    if nt[current] == 0 or ne[current] == 0:
        raise ValueError("current_dose must identify a treated dose")
    frac_t = pt[current] / nt[current]
    frac_e = pe[current] / ne[current]
    max_pending_toxicity = scalar(max_pending_toxicity, "max_pending_toxicity")
    max_pending_efficacy = scalar(max_pending_efficacy, "max_pending_efficacy")
    zero = np.any((nt > 0) & (et <= 0)) or np.any((ne > 0) & (ee <= 0))
    if not 0 <= max_pending_toxicity <= 1 or not 0 <= max_pending_efficacy <= 1:
        raise ValueError("pending thresholds must lie in [0,1]")
    if frac_t > max_pending_toxicity or frac_e > max_pending_efficacy or zero:
        action = "suspend_pending" if not zero else "suspend_no_information"
        return TITEBOIN12Decision(
            action,
            None,
            _readonly(prior, np.bool_),
            _readonly(np.column_stack((pt, pe)), np.int64),
            None,
        )
    result = tite_boin12_posterior(
        design, d, t, e, tf, ef, toxicity_window=tw, efficacy_window=ew, n_doses=k
    )
    allowed = result.admissible & ~prior
    if not np.any(allowed):
        return TITEBOIN12Decision(
            "stop_safety", None, _readonly(~allowed, np.bool_), result.pending_counts, result
        )
    rate = float(result.MLE[current, 0])
    boin = design._boin
    if design.early_stop_patients is not None and result.N[current] >= design.early_stop_patients:
        return TITEBOIN12Decision(
            "stop_precision", None, _readonly(~allowed, np.bool_), result.pending_counts, result
        )
    if rate >= boin.deescalation_boundary:
        target = max(current - 1, 0)
        if allowed[target]:
            return TITEBOIN12Decision(
                "deescalate",
                target + 1,
                _readonly(~allowed, np.bool_),
                result.pending_counts,
                result,
            )
        return TITEBOIN12Decision(
            "stop_no_admissible_neighbor",
            None,
            _readonly(~allowed, np.bool_),
            result.pending_counts,
            result,
        )
    if (
        result.N[current] >= design.exploration_patients
        and current + 1 < k
        and result.N[current + 1] == 0
        and allowed[current + 1]
        and rate < boin.deescalation_boundary
    ):
        return TITEBOIN12Decision(
            "explore_escalate",
            current + 2,
            _readonly(~allowed, np.bool_),
            result.pending_counts,
            result,
        )
    candidates = np.arange(max(0, current - 1), min(k, current + 2))
    candidates = candidates[allowed[candidates]]
    if rate > boin.escalation_boundary and result.N[current] >= design.stay_patients:
        candidates = candidates[candidates <= current]
    if candidates.size == 0:
        return TITEBOIN12Decision(
            "stop_no_admissible_neighbor",
            None,
            _readonly(~allowed, np.bool_),
            result.pending_counts,
            result,
        )
    best = int(candidates[np.argmax(result.posterior.utility_probability[candidates])])
    action = "stay" if best == current else "escalate" if best > current else "deescalate"
    return TITEBOIN12Decision(
        action, best + 1, _readonly(~allowed, np.bool_), result.pending_counts, result
    )
