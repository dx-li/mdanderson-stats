"""Recovered desktop DSF contracts with explicit scientific and legacy profiles."""

from dataclasses import dataclass
from math import exp, isfinite, log, log1p

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtri

from ._cdflib import _freeze
from .dose_schedule_decision import DoseScheduleDecision, _freeze_bool, _scalar
from .dose_schedule_trial import _count


def _profile(value: str) -> str:
    if value not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    return value


def dose_schedule_desktop_decision(
    mean_risk: ArrayLike,
    overdose_probability: ArrayLike,
    treated: ArrayLike,
    *,
    target: float,
    upper_probability: float,
    final: bool = False,
    profile: str = "safe",
) -> DoseScheduleDecision:
    """Allocate from supplied posterior summaries, not a posterior fit itself.

    Legacy accepts cutoff equality, propagates lower-dose exclusions within a
    schedule, and retains no-skip at final selection. Safe evaluates each risk
    independently, uses a strict cutoff and removes no-skip at final selection.
    Both start at (0,0). Exact distance ties use dose then schedule order.
    """
    _profile(profile)
    mean = np.asarray(mean_risk)
    tail = np.asarray(overdose_probability)
    counts = np.asarray(treated)
    if mean.ndim != 2 or not (1 <= mean.shape[0] <= 7 and 1 <= mean.shape[1] <= 5):
        raise ValueError("posterior summaries require 1..7 doses and 1..5 schedules")
    for values in (mean, tail):
        if values.shape != mean.shape or values.dtype.kind not in "iuf":
            raise ValueError("posterior summaries must be matching real matrices")
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("posterior summaries must be finite probabilities")
    if counts.shape != mean.shape or counts.dtype.kind not in "iu":
        raise ValueError("treated must be a matching integer matrix")
    if np.any(counts < 0) or np.sum(counts, dtype=object) > 200:
        raise ValueError("treated total must be at most 200 and counts nonnegative")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    target = _scalar(target, "target")
    cutoff = _scalar(upper_probability, "upper_probability")
    if not 0 < target < 1 or not 0 < cutoff < 1:
        raise ValueError("target and upper_probability must be in (0,1)")
    legacy = profile == "desktop_legacy"
    acceptable = tail <= cutoff if legacy else tail < cutoff
    eligible = acceptable.copy()
    if not np.any(counts):
        if final:
            raise ValueError("final selection requires at least one patient")
        if legacy:
            eligible[:] = False  # Native first-assignment bypass never computes this mask.
        return DoseScheduleDecision(
            "start",
            (0, 0),
            _freeze(mean),
            _freeze(tail),
            _freeze_bool(acceptable),
            _freeze_bool(eligible),
            None,
            "first patient follows lowest dose and shortest schedule",
        )
    if legacy or not final:
        maximum = np.max(np.argwhere(counts > 0), axis=0) + 1
        dose, schedule = np.indices(mean.shape)
        eligible &= (dose <= maximum[0]) & (schedule <= maximum[1])
    if legacy:
        eligible = np.logical_and.accumulate(eligible, axis=0)
    distance = np.where(eligible, abs(mean - target), np.inf)
    if not np.any(eligible):
        pair = None
        action = "stop"
        selected_distance = None
    else:
        indices = np.unravel_index(int(np.argmin(distance)), mean.shape)
        pair = (int(indices[0]), int(indices[1]))
        action = "select" if final else "treat"
        selected_distance = float(distance[pair])
    return DoseScheduleDecision(
        action,
        pair,
        _freeze(mean),
        _freeze(tail),
        _freeze_bool(acceptable),
        _freeze_bool(eligible),
        selected_distance,
        f"{profile}: nearest eligible posterior mean risk with lexicographic ties",
    )


@dataclass(frozen=True)
class DoseScheduleDesktopObservation:
    """Likelihood time and event status visible at a specified calendar time."""

    included: bool
    time: float
    event: bool
    pending: bool
    final_ready: bool


def dose_schedule_desktop_observation(
    *,
    outcome: int,
    outcome_time: float,
    entry_time: float,
    study_time: float,
    horizon: float,
    low_grade_delay: float = 14.0,
    profile: str = "safe",
) -> DoseScheduleDesktopObservation:
    """Observe a native-style outcome (0 censored, 1 toxicity, 2 delayed toxicity).

    Safe uses the requested delay and caps censored follow-up at the horizon.
    Legacy uses the native fixed 14 days, even when the GUI requested another
    delay, and retains native uncapped pending follow-up. A confirmed delayed
    event is backdated to onset. Pending status blocks a safe final analysis.
    """
    _profile(profile)
    outcome = _count(outcome, "outcome", 0, 2)
    values = [outcome_time, entry_time, study_time, horizon, low_grade_delay]
    values = [
        _scalar(v, n)
        for v, n in zip(
            values,
            ("outcome_time", "entry_time", "study_time", "horizon", "low_grade_delay"),
            strict=True,
        )
    ]
    onset, entry, study, horizon, delay = values
    if min(values) < 0 or horizon == 0:
        raise ValueError("times must be nonnegative and horizon positive")
    if profile == "desktop_legacy":
        delay = 14.0
    elapsed = study - entry
    if elapsed < 0:
        return DoseScheduleDesktopObservation(False, 0.0, False, False, False)
    deadline = onset + (delay if outcome == 2 else 0)
    if not isfinite(deadline):
        raise ValueError("qualification deadline is not representable")
    if profile == "safe" and study >= entry + horizon and elapsed < horizon:
        elapsed = horizon  # Avoid cancellation at an exactly completed calendar horizon.
    reached = study >= entry + deadline if profile == "safe" else elapsed >= deadline
    qualifies = outcome != 0 and reached
    if profile == "safe":
        qualifies = qualifies and onset <= horizon
    pending = outcome == 2 and onset <= min(elapsed, horizon) and not reached
    if qualifies:
        time = onset
    elif profile == "desktop_legacy" and outcome != 0:
        time = elapsed
    else:
        time = min(elapsed, horizon, onset if outcome == 0 else horizon)
    ready = qualifies or (elapsed >= horizon and not pending and (outcome != 0 or onset >= horizon))
    return DoseScheduleDesktopObservation(True, time, qualifies, pending, ready)


def dose_schedule_desktop_lognormal(probability: float, horizon: float) -> tuple[float, float]:
    """Calibrate lognormal quantile and variance used by the desktop simulator.

    P(T <= horizon)=probability and Var(T) equals the squared mean of the
    corresponding exponential distribution. Uses an accurate normal inverse
    and bounded scalar solve; native inverse-normal approximation differs.
    """
    p = _scalar(probability, "probability")
    tau = _scalar(horizon, "horizon")
    if not 0 < p < 1 or not 0 < tau <= 1e6:
        raise ValueError("probability must be in (0,1), horizon in (0,1e6]")
    z = float(ndtri(p))
    target = -2 * log(-log1p(-p))

    def residual(sigma):
        v = sigma * sigma
        log_expm1 = log(np.expm1(v)) if v < 50 else v + log1p(-exp(-v))
        return -2 * sigma * z + v + log_expm1 - target

    sigma = float(brentq(residual, 1e-12, max(10.0, 2 * abs(z) + 10), xtol=1e-13))
    return log(tau) - sigma * z, sigma


def dose_schedule_desktop_event_time(
    uniform: float,
    probability: float,
    horizon: float,
    *,
    distribution: str = "exponential",
    shape: float = 1.0,
) -> float:
    """Generate a censored event time from an explicit CDF-oriented uniform.

    Returns horizon for no event; native simulation treats the horizon tie as
    censored. Python's supplied uniform convention does not claim native RNG
    stream equivalence. Extreme probabilities zero/one have exact boundaries.
    """
    u = _scalar(uniform, "uniform")
    p = _scalar(probability, "probability")
    tau = _scalar(horizon, "horizon")
    shape = _scalar(shape, "shape")
    if not 0 < u < 1 or not 0 <= p <= 1 or not 0 < tau <= 1e6:
        raise ValueError("invalid uniform, probability or horizon")
    if distribution not in ("exponential", "weibull", "lognormal"):
        raise ValueError("unknown event distribution")
    if not 0 < shape <= 100:
        raise ValueError("shape must be in (0,100]")
    if p == 0 or u >= p:
        return tau
    if p == 1:
        return 0.0
    if distribution == "lognormal":
        mu, sigma = dose_schedule_desktop_lognormal(p, tau)
        return min(tau, exp(mu + sigma * float(ndtri(u))))
    power = shape if distribution == "weibull" else 1.0
    ratio = log1p(-u) / log1p(-p)
    return tau * exp(log(ratio) / power)
