"""Bayesian phase-II monitoring with delayed binary outcomes.

The delayed outcomes are treated as missing and multiply imputed from a
correlated piecewise-exponential event-time model (Cai, Liu and Yuan, 2014).
The paper's Gamma notation is ambiguous about scale/rate: this module uses
``Gamma(shape, rate)`` throughout.  In particular the martingale prior is
``Gamma(c_j, rate=c_j / lambda_previous)``, which has conditional mean equal
to ``lambda_previous``.
"""

from dataclasses import dataclass
from math import exp, log

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc, expm1, logsumexp

from ._validation import scalar


@dataclass(frozen=True)
class Phase2DelayResult:
    """Posterior monitoring result and Monte Carlo diagnostics."""

    decision: str
    stopped: bool
    posterior_probability: float
    threshold: float
    cutoff: float
    endpoint: str
    n_subjects: int
    n_observed: int
    n_pending: int
    observed_events: int
    posterior_probability_sd: float
    posterior_probability_mc_se: float
    imputation_draws: int
    hazard_draws: int
    hazard_trace: NDArray[np.float64]
    log_hazard_trace: NDArray[np.float64]
    probability_trace: NDArray[np.float64]


def _beta_tail(alpha: float, beta: float, threshold: float, upper: bool) -> float:
    value = float((betaincc if upper else betainc)(alpha, beta, threshold))
    if not np.isfinite(value):
        raise ArithmeticError("Beta posterior tail is not finite")
    return value


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    number = scalar(value, name)
    if not lower <= number <= upper or number != int(number):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(number)


def _exp_penalty(value: float) -> float:
    # Infinite penalties mean zero conditional density, not a clipped hazard.
    return float("inf") if value > 709.782712893384 else exp(value)


def _validate_inputs(
    event: ArrayLike,
    event_time: ArrayLike,
    followup: ArrayLike,
    window: float,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    e = np.asarray(event)
    t = np.asarray(event_time, dtype=float)
    r = np.asarray(followup, dtype=float)
    if e.ndim != 1 or t.ndim != 1 or r.ndim != 1 or not (e.shape == t.shape == r.shape):
        raise ValueError(
            "event, event_time and followup must be one-dimensional arrays of equal length"
        )
    if not 1 <= e.size <= 1000:
        raise ValueError("between 1 and 1000 subjects are supported")
    if not np.all(np.isin(e, [0, 1])):
        raise ValueError("event must contain only 0/1 indicators")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)):
        raise ValueError("event_time and followup must be finite")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be positive and finite")
    if np.any(r < 0) or np.any(r > window):
        raise ValueError("followup must lie in [0, window]")
    if np.any(t < 0) or np.any(t > r):
        raise ValueError("event_time must lie in [0, followup]")
    if np.any((e == 0) & (t != 0)):
        raise ValueError("non-events must have event_time=0")
    if endpoint not in {"response", "toxicity", "progression"}:
        raise ValueError("endpoint must be response, toxicity, or progression")
    return e.astype(bool), t, r, float(window), endpoint


def _exposure_counts(
    event: np.ndarray, event_time: np.ndarray, followup: np.ndarray, window: float, intervals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, window, intervals + 1)
    exposure = np.zeros(intervals)
    events: NDArray[np.int64] = np.zeros(intervals, dtype=int)
    pending: NDArray[np.bool_] = np.zeros(event.size, dtype=bool)
    for i, (ev, ti, ri) in enumerate(zip(event, event_time, followup)):
        pending[i] = not ev and ri < window
        x = ti if ev else ri
        for j in range(intervals):
            exposure[j] += max(0.0, min(x, edges[j + 1]) - edges[j])
        if ev:
            j = min(intervals - 1, int(np.searchsorted(edges, ti, side="right") - 1))
            events[j] += 1
    return exposure, events, pending


def _log_density(
    z: float,
    j: int,
    log_hazards: np.ndarray,
    exposure: np.ndarray,
    events: np.ndarray,
    c: np.ndarray,
    initial_log_hazard: float,
) -> float:
    previous_z = initial_log_hazard if j == 0 else log_hazards[j - 1]
    value = (events[j] + c[j] - (c[j + 1] if j + 1 < c.size else 0.0)) * z
    if exposure[j] > 0:
        value -= _exp_penalty(log(exposure[j]) + z)
    value -= _exp_penalty(log(c[j]) + z - previous_z)
    if j + 1 < c.size:
        value -= _exp_penalty(log(c[j + 1]) + log_hazards[j + 1] - z)
    if np.isnan(value):
        raise ArithmeticError("log-hazard conditional density is indeterminate")
    return float(value)


def _slice_log_hazard(
    current: float,
    j: int,
    log_hazards: np.ndarray,
    exposure: np.ndarray,
    events: np.ndarray,
    c: np.ndarray,
    initial_log_hazard: float,
    rng: np.random.Generator,
) -> float:
    z0 = current
    f0 = _log_density(z0, j, log_hazards, exposure, events, c, initial_log_hazard)
    if not np.isfinite(f0):
        raise ArithmeticError("current log-hazard state has nonfinite density")
    level = f0 - rng.exponential()
    width = 1.0
    left = z0 - rng.random() * width
    right = left + width
    left_ok = right_ok = False
    step = width
    for _ in range(64):
        if _log_density(left, j, log_hazards, exposure, events, c, initial_log_hazard) <= level:
            left_ok = True
            break
        left -= step
        step *= 2
    step = width
    for _ in range(64):
        if _log_density(right, j, log_hazards, exposure, events, c, initial_log_hazard) <= level:
            right_ok = True
            break
        right += step
        step *= 2
    if not (left_ok and right_ok):
        raise ArithmeticError("could not bracket log-hazard slice; increase numerical range")
    for _ in range(100):
        z = rng.uniform(left, right)
        if _log_density(z, j, log_hazards, exposure, events, c, initial_log_hazard) >= level:
            return z
        if z < z0:
            left = z
        else:
            right = z
    raise ArithmeticError("log-hazard slice shrinkage failed")


def phase2_delay_monitor(
    event: ArrayLike,
    event_time: ArrayLike,
    followup: ArrayLike,
    *,
    window: float,
    endpoint: str = "response",
    threshold: float,
    cutoff: float = 0.95,
    intervals: int = 6,
    prior_alpha: float,
    prior_beta: float,
    hazard_c: float | ArrayLike,
    lambda0: float,
    burn_in: int = 200,
    hazard_draws: int = 500,
    imputations_per_draw: int = 1,
    seed: int | None = None,
) -> Phase2DelayResult:
    """Monitor one interim snapshot using posterior multiple imputation.

    ``hazard_draws`` posterior hazard states are generated after ``burn_in``
    Gibbs sweeps; each state yields ``imputations_per_draw`` completed binary
    datasets. Numerical defaults are Python implementation choices, not app
    defaults. Bounds prevent accidental allocation of unreasonably large runs.
    """
    window = scalar(window, "window")
    threshold = scalar(threshold, "threshold")
    cutoff = scalar(cutoff, "cutoff")
    prior_alpha = scalar(prior_alpha, "prior_alpha")
    prior_beta = scalar(prior_beta, "prior_beta")
    lambda0 = scalar(lambda0, "lambda0")
    intervals = _integer(intervals, "intervals", 1, 100)
    burn_in = _integer(burn_in, "burn_in", 0, 100_000)
    hazard_draws = _integer(hazard_draws, "hazard_draws", 1, 100_000)
    imputations_per_draw = _integer(imputations_per_draw, "imputations_per_draw", 1, 1000)
    event, event_time, followup, window, endpoint = _validate_inputs(
        event, event_time, followup, window, endpoint
    )
    if not 0 < threshold < 1 or not 0 < cutoff <= 1:
        raise ValueError("threshold must lie in (0,1) and cutoff in (0,1]")
    if prior_alpha <= 0 or prior_beta <= 0 or not np.isfinite(prior_alpha + prior_beta):
        raise ValueError("prior_alpha and prior_beta must be positive and finite")
    n = event.size
    if (burn_in + hazard_draws) * intervals > 200_000 or hazard_draws * intervals > 2_000_000:
        raise ValueError("requested Gibbs run is too large")
    if hazard_draws * n * intervals > 5_000_000:
        raise ValueError("requested remaining-exposure work is too large")
    if hazard_draws * imputations_per_draw * n > 2_000_000:
        raise ValueError("requested Monte Carlo run is too large")
    c = np.broadcast_to(np.asarray(hazard_c, dtype=float), (intervals,)).copy()
    if np.any(~np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("hazard_c must be positive and finite")
    if not np.isfinite(lambda0) or lambda0 <= 0:
        raise ValueError("lambda0 must be positive and finite")
    exposure, events, pending = _exposure_counts(
        event, event_time / window, followup / window, 1.0, intervals
    )
    observed_events = int(event.sum())
    n_pending = int(pending.sum())
    if n_pending == 0:
        a = prior_alpha + observed_events
        b = prior_beta + (event.size - observed_events)
        probability = _beta_tail(a, b, threshold, endpoint in {"toxicity", "progression"})
        stopped = probability > cutoff
        decision = (
            (
                "stop_futility"
                if endpoint == "response"
                else "stop_safety"
                if endpoint == "toxicity"
                else "stop_futility"
            )
            if stopped
            else "continue"
        )
        empty_h: NDArray[np.float64] = np.empty((0, intervals), dtype=float)
        empty_lh: NDArray[np.float64] = np.empty((0, intervals), dtype=float)
        empty_p: NDArray[np.float64] = np.empty(0, dtype=float)
        empty_h.flags.writeable = False
        empty_lh.flags.writeable = False
        empty_p.flags.writeable = False
        return Phase2DelayResult(
            decision,
            stopped,
            probability,
            float(threshold),
            float(cutoff),
            endpoint,
            event.size,
            event.size,
            0,
            observed_events,
            0.0,
            0.0,
            0,
            0,
            empty_h,
            empty_lh,
            empty_p,
        )
    rng = np.random.default_rng(seed)
    log_window = log(window)
    initial_log_hazard = log(lambda0) + log_window
    log_hazards: NDArray[np.float64] = np.full(intervals, initial_log_hazard, dtype=float)
    total_sweeps = burn_in + hazard_draws
    probability_trace: NDArray[np.float64] = np.empty(
        hazard_draws * imputations_per_draw, dtype=float
    )
    beta_tails = np.array(
        [
            _beta_tail(
                prior_alpha + observed_events + k,
                prior_beta + (n - observed_events - k),
                threshold,
                endpoint in {"toxicity", "progression"},
            )
            for k in range(n_pending + 1)
        ]
    )
    draw_means: NDArray[np.float64] = np.empty(hazard_draws, dtype=float)
    hazard_trace: NDArray[np.float64] = np.empty((hazard_draws, intervals), dtype=float)
    log_hazard_trace: NDArray[np.float64] = np.empty((hazard_draws, intervals), dtype=float)
    edges = np.linspace(0.0, 1.0, intervals + 1)
    pending_idx = np.flatnonzero(pending)
    remaining_exposure = np.maximum(
        0.0,
        edges[1:][None, :] - np.maximum(followup[pending_idx, None] / window, edges[:-1][None, :]),
    )
    with np.errstate(divide="ignore"):
        log_remaining = np.log(remaining_exposure)
    pos = 0
    for sweep in range(total_sweeps):
        for j in range(intervals):
            log_hazards[j] = _slice_log_hazard(
                log_hazards[j], j, log_hazards, exposure, events, c, initial_log_hazard, rng
            )
        if sweep < burn_in:
            continue
        draw = sweep - burn_in
        log_hazard_trace[draw] = log_hazards - log_window
        with np.errstate(over="raise"):
            hazard_trace[draw] = np.exp(log_hazard_trace[draw])
        omega: NDArray[np.float64] = np.empty(n_pending)
        for k, remaining in enumerate(log_remaining):
            log_h = float(logsumexp(log_hazards + remaining))
            omega[k] = 1.0 if log_h > 709 else -expm1(-exp(log_h))
        draw_start = pos
        for _ in range(imputations_per_draw):
            imputed = int(np.sum(rng.random(n_pending) < omega)) if n_pending else 0
            p = float(beta_tails[imputed])
            probability_trace[pos] = p
            pos += 1
        draw_means[draw] = float(np.mean(probability_trace[draw_start:pos]))
    mean = float(np.mean(probability_trace))
    sd = float(np.std(probability_trace, ddof=1)) if probability_trace.size > 1 else 0.0
    # Imputations sharing one hazard state are nested, so estimate MC error
    # from the per-hazard means. Batch means also allow modest Gibbs autocorrelation.
    if draw_means.size < 2:
        mc_se = float("nan")
    else:
        batches = min(20, draw_means.size)
        means = np.array([x.mean() for x in np.array_split(draw_means, batches)])
        mc_se = float(np.std(means, ddof=1) / np.sqrt(means.size))
    stopped = mean > cutoff
    if endpoint == "response":
        decision = "stop_futility" if stopped else "continue"
    elif endpoint == "toxicity":
        decision = "stop_safety" if stopped else "continue"
    else:
        decision = "stop_futility" if stopped else "continue"
    hazard_trace.flags.writeable = False
    log_hazard_trace.flags.writeable = False
    probability_trace.flags.writeable = False
    return Phase2DelayResult(
        decision,
        stopped,
        mean,
        float(threshold),
        float(cutoff),
        endpoint,
        event.size,
        event.size - n_pending,
        n_pending,
        observed_events,
        sd,
        mc_se,
        probability_trace.size,
        hazard_draws,
        hazard_trace,
        log_hazard_trace,
        probability_trace,
    )
