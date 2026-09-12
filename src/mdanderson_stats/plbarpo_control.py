"""Concurrent and entire-trial control monitoring for PLBARPO.

This module contains no trial scheduler.  It only aggregates dated records and
computes the posterior ordering probabilities used by platform monitoring.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_binomial


def _readonly_bool(value: ArrayLike) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _readonly_counts(value: ArrayLike) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _counts(value: ArrayLike, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = count(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def plbarpo_control_counts(
    enrollment_time: ArrayLike,
    outcomes: ArrayLike,
    observation_time: ArrayLike,
    windows: ArrayLike,
    *,
    as_of: float,
) -> np.ndarray:
    """Aggregate observed binary outcomes into half-open enrollment windows.

    Each window is ``[open, close)``.  Enrollment exactly at ``close`` belongs
    to the next window (or no window).  ``close`` may be positive infinity;
    window starts and all record enrollment times must be finite.  A record is
    counted only when enrolled by ``as_of`` and its outcome was observed by
    ``as_of``.  Pending outcomes are represented by infinite observation time.
    """
    enrollment = finite(enrollment_time, "enrollment_time")
    if enrollment.ndim != 1:
        raise ValueError("enrollment_time must be one-dimensional")
    if len(enrollment) > 100_000:
        raise ValueError("at most 100000 records are supported")
    outcome = np.asarray(outcomes, dtype=np.float64)
    observation = np.asarray(observation_time, dtype=np.float64)
    if outcome.shape != enrollment.shape:
        raise ValueError("enrollment_time and outcomes must be one-dimensional with equal length")
    if observation.shape != enrollment.shape:
        raise ValueError("observation_time must have one entry per record")
    if np.any(np.isnan(observation)) or np.any(observation < enrollment):
        raise ValueError("observation_time must be >= enrollment_time or positive infinity")
    if np.any(~np.isfinite(enrollment)):
        raise ValueError("enrollment_time must be finite")
    cutoff = scalar(as_of, "as_of")
    missing_outcome = np.isnan(outcome)
    if np.any(missing_outcome & (observation <= cutoff)):
        raise ValueError("missing outcomes are allowed only for observations after as_of")
    if np.any(~missing_outcome & ((outcome != 0) & (outcome != 1))):
        raise ValueError("outcomes must be binary 0/1 when observed")
    raw_window = np.asarray(windows, dtype=np.float64)
    if raw_window.ndim != 2 or raw_window.shape[1] != 2 or not 1 <= raw_window.shape[0] <= 100:
        raise ValueError("windows must have shape (K, 2), with 1 <= K <= 100")
    if np.any(np.isnan(raw_window)) or np.any(np.isinf(raw_window[:, 0])):
        raise ValueError("window starts must be finite and window closes may only be +inf")
    if np.any(np.isneginf(raw_window[:, 1])):
        raise ValueError("window closes may only be positive infinity")
    if np.any(raw_window[:, 0] >= raw_window[:, 1]):
        raise ValueError("each window must satisfy open < close")

    eligible = (enrollment <= cutoff) & (observation <= cutoff)
    successes: np.ndarray = np.zeros(len(raw_window), dtype=np.float64)
    failures: np.ndarray = np.zeros(len(raw_window), dtype=np.float64)
    for i, (opening, closing) in enumerate(raw_window):
        in_window = eligible & (enrollment >= opening) & (enrollment < closing)
        successes[i] = np.sum(in_window & (outcome == 1))
        failures[i] = np.sum(in_window & (outcome == 0))
    return _readonly_counts(np.column_stack((successes, failures)))


@dataclass(frozen=True)
class PLBarpoControlMonitoring:
    treatment_alpha: np.ndarray
    treatment_beta: np.ndarray
    control_alpha: np.ndarray
    control_beta: np.ndarray
    futility_probability: np.ndarray | None
    efficacy_probability: np.ndarray | None
    final_efficacy_probability: np.ndarray | None
    absolute_error: np.ndarray
    futile: np.ndarray | None
    efficacious: np.ndarray | None
    final_efficacious: np.ndarray | None


def _prior(prior: ArrayLike, name: str, k: int | None = None) -> np.ndarray:
    value = finite(prior, name)
    if value.ndim == 1 and value.shape == (2,) and k is None:
        value = value.reshape(1, 2)
    if value.ndim != 2 or value.shape[1] != 2 or (k is not None and value.shape != (k, 2)):
        expected = f"({k}, 2)" if k is not None else "(2,)"
        raise ValueError(f"{name} must have shape {expected}")
    if np.any(value <= 0) or not np.all(np.isfinite(value.sum(axis=1))):
        raise ValueError(f"{name} must contain positive finite beta shapes")
    return value


def _cutoff(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = scalar(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def plbarpo_control_monitor(
    successes: ArrayLike,
    failures: ArrayLike,
    *,
    prior: ArrayLike,
    control_prior: ArrayLike,
    control_counts: ArrayLike,
    control_mode: str = "entire",
    pfut: float | None = None,
    peff: float | None = None,
    pfinal: float | None = None,
    absolute_tolerance: float = 1e-9,
) -> PLBarpoControlMonitoring:
    """Monitor treatment arms against an entire-trial or concurrent control.

    ``prior`` is a K-by-2 treatment prior.  ``control_counts`` is a single
    ``(successes, failures)`` pair for ``entire`` mode, or K-by-2 pairs in
    ``concurrent`` mode.  Treatment vectors exclude the control arm.
    """
    if control_mode not in ("entire", "concurrent"):
        raise ValueError("control_mode must be entire or concurrent")
    pfut, peff, pfinal = (
        _cutoff(v, n) for v, n in ((pfut, "pfut"), (peff, "peff"), (pfinal, "pfinal"))
    )
    k_prior = finite(prior, "prior")
    if k_prior.ndim != 2 or k_prior.shape[1] != 2 or not 1 <= k_prior.shape[0] <= 100:
        raise ValueError("prior must have shape (K, 2), with 1 <= K <= 100")
    k = k_prior.shape[0]
    p = _prior(k_prior, "prior", k)
    control_shape = finite(control_prior, "control_prior")
    if control_shape.shape != (2,) or np.any(control_shape <= 0):
        raise ValueError("control_prior must have shape (2,) with positive shapes")
    s = _counts(successes, "successes", (k,))
    f = _counts(failures, "failures", (k,))
    if control_mode == "entire":
        cc = _counts(control_counts, "control_counts", (2,))
        control_s = np.full(k, cc[0])
        control_f = np.full(k, cc[1])
    else:
        cc = _counts(control_counts, "control_counts", (k, 2))
        control_s, control_f = cc.T

    treatment = BetaBinomialPosterior(p[:, 0] + s, p[:, 1] + f)
    control = BetaBinomialPosterior(control_shape[0] + control_s, control_shape[1] + control_f)
    comparison = compare_beta_binomial(control, treatment, absolute_tolerance=absolute_tolerance)
    fut = comparison.control_greater if pfut is not None else None
    eff = comparison.treatment_greater if peff is not None else None
    final = comparison.treatment_greater if pfinal is not None else None
    futile = None if fut is None else fut > pfut
    efficacious = None if eff is None else eff >= peff
    final_efficacious = None if final is None else final >= pfinal
    return PLBarpoControlMonitoring(
        _freeze(treatment.alpha),
        _freeze(treatment.beta),
        _freeze(control.alpha),
        _freeze(control.beta),
        None if fut is None else _freeze(fut),
        None if eff is None else _freeze(eff),
        None if final is None else _freeze(final),
        _freeze(comparison.absolute_error),
        None if futile is None else _readonly_bool(futile),
        None if efficacious is None else _readonly_bool(efficacious),
        None if final_efficacious is None else _readonly_bool(final_efficacious),
    )
