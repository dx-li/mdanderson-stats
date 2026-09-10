"""Posterior predictive survival trials with competing accrual termination rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _integer
from .beta_binomial import _owned
from .predictive_survival import _survival_prior, compare_predictive_survival


@dataclass(frozen=True)
class SurvivalPredictiveProbabilities:
    """Conclusion order: A superior, B superior, inconclusive, not evaluable."""

    probability: FloatArray
    mcse: FloatArray
    patients: FloatArray
    events: FloatArray
    exposure: FloatArray
    accrual_stop_time: FloatArray
    final_time: FloatArray
    decision: NDArray[np.str_]


def _stop_time(
    arrivals: FloatArray,
    event_times: FloatArray,
    pending: FloatArray,
    remaining_events: int | None,
    remaining_patients: int | None,
    deadline: float,
) -> float:
    stop = deadline
    if remaining_patients == 0 or remaining_events == 0:
        return 0.0
    if remaining_patients is not None and arrivals.size >= remaining_patients:
        stop = min(stop, float(arrivals[remaining_patients - 1]))
    if remaining_events is not None:
        possible = np.r_[pending, event_times]
        if possible.size >= remaining_events:
            stop = min(
                stop, float(np.partition(possible, remaining_events - 1)[remaining_events - 1])
            )
    return stop


def _future_summary(
    arrivals: FloatArray,
    arms: NDArray[np.int64],
    durations: FloatArray,
    pending: FloatArray,
    pending_arms: NDArray[np.int64],
    stop: float,
    followup: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    final = stop + followup
    if not np.isfinite(final) or (followup > 0 and final <= stop):
        raise ArithmeticError("final follow-up time is not representable")
    keep = arrivals <= stop
    arm, observed = arms[keep], final - arrivals[keep]
    d = np.bincount(arm, weights=(arrivals[keep] + durations[keep] <= final), minlength=2).astype(
        float
    )
    d += np.bincount(pending_arms, weights=(pending <= final), minlength=2)
    e = np.bincount(arm, weights=np.minimum(durations[keep], observed), minlength=2).astype(float)
    e += np.bincount(pending_arms, weights=np.minimum(pending, final), minlength=2)
    return np.bincount(arm, minlength=2).astype(float), d, e, final


def predictive_survival(
    patients: ArrayLike,
    events: ArrayLike,
    exposure: ArrayLike,
    *,
    prior: ArrayLike,
    accrual_rate: float,
    followup: float,
    max_patients: int | None = None,
    max_duration: float | None = None,
    elapsed_time: float = 0,
    max_events: int | None = None,
    allocation_probability: float = 0.5,
    method: str = "bayesian",
    posterior_cutoff: float = 0.95,
    significance_level: float = 0.025,
    n_simulations: int = 10000,
    max_simulated_patients: int = 10000,
    rng: np.random.Generator | int | None = None,
) -> SurvivalPredictiveProbabilities:
    """Simulate exponential outcomes from the current inverse-gamma mean posterior.

    Accrual ends at the first enabled total-patient, total-duration or total-event
    limit. Additional follow-up begins then; events may exceed the stopping limit.
    Current non-event patients remain at risk; exponential memorylessness permits
    use of aggregate current exposure. At least one accrual limit is required.
    """
    n, d, e = count(patients, "patients"), count(events, "events"), finite(exposure, "exposure")
    if (
        n.shape != (2,)
        or d.shape != (2,)
        or e.shape != (2,)
        or np.any(d > n)
        or np.any((n == 0) & (e != 0))
    ):
        raise ValueError(
            "require two-arm counts, events<=patients, and zero exposure for empty arms"
        )
    initial = compare_predictive_survival(
        d,
        e,
        prior=prior,
        method=method,
        posterior_cutoff=posterior_cutoff,
        significance_level=significance_level,
    )
    pair = _survival_prior(prior)
    rate, extra, elapsed, allocation = (
        scalar(x, name)
        for x, name in (
            (accrual_rate, "accrual_rate"),
            (followup, "followup"),
            (elapsed_time, "elapsed_time"),
            (allocation_probability, "allocation_probability"),
        )
    )
    if rate <= 0 or extra < 0 or elapsed < 0 or not 0 <= allocation <= 1:
        raise ValueError(
            "require positive accrual rate, nonnegative follow-up/elapsed time, allocation in [0,1]"
        )
    simulations, guard = (
        _integer(n_simulations, "n_simulations"),
        _integer(max_simulated_patients, "max_simulated_patients"),
    )
    if not 1 <= simulations <= 100000 or not 1 <= guard <= 100000 or n.sum() > guard:
        raise ValueError(
            "simulation count and patient guard must lie in [1,100000], guard>=current patients"
        )
    cap = None if max_patients is None else _integer(max_patients, "max_patients")
    target = None if max_events is None else _integer(max_events, "max_events")
    duration = None if max_duration is None else scalar(max_duration, "max_duration")
    if cap is None and target is None and duration is None:
        raise ValueError("at least one accrual termination limit is required")
    if (
        (cap is not None and (cap < n.sum() or cap > guard))
        or (target is not None and target < d.sum())
        or (duration is not None and duration < elapsed)
    ):
        raise ValueError(
            "limits must not precede current data, and max_patients must not exceed the guard"
        )
    remaining_n = None if cap is None else int(cap - n.sum())
    remaining_d = None if target is None else int(target - d.sum())
    deadline = np.inf if duration is None else duration - elapsed
    interval = 1 / rate
    if not np.isfinite(interval):
        raise ArithmeticError("accrual interval is not representable")
    pending_arms = np.repeat(np.arange(2), (n - d).astype(np.int64))
    generator = np.random.default_rng(rng)
    ns, ds, es = np.empty((simulations, 2)), np.empty((simulations, 2)), np.empty((simulations, 2))
    stops, finals = np.empty(simulations), np.empty(simulations)
    for trial in range(simulations):
        gamma = generator.gamma(initial.posterior_shape, 1)
        with np.errstate(divide="ignore", over="ignore"):
            means = initial.posterior_scale / gamma
        if np.any(~np.isfinite(means)) or np.any(means <= 0):
            raise ArithmeticError("posterior mean-survival draw is not representable")
        pending = generator.exponential(size=pending_arms.size) * means[pending_arms]
        arrivals, durations = np.empty(0), np.empty(0)
        arms = np.empty(0, dtype=np.int64)
        event_times = np.empty(0)
        stop = _stop_time(arrivals, event_times, pending, remaining_d, remaining_n, deadline)
        while stop > (float(arrivals[-1]) if arrivals.size else 0):
            available = guard - int(n.sum()) - arrivals.size
            if remaining_n is not None:
                available = min(available, remaining_n - arrivals.size)
            if available <= 0:
                if remaining_n is not None and arrivals.size == remaining_n:
                    break
                raise ArithmeticError(
                    "patient simulation guard reached before resolving accrual stop"
                )
            size = min(64, available)
            gaps = generator.exponential(interval, size=size)
            times = (float(arrivals[-1]) if arrivals.size else 0) + gaps.cumsum()
            assigned = (generator.random(size) >= allocation).astype(np.int64)
            delays = generator.exponential(size=size) * means[assigned]
            if (
                np.any(~np.isfinite(times))
                or np.any(~np.isfinite(delays))
                or np.any(np.diff(np.r_[arrivals[-1:] if arrivals.size else [0], times]) <= 0)
            ):
                raise ArithmeticError("simulated arrival/event durations are not representable")
            arrivals, durations, arms = (
                np.r_[arrivals, times],
                np.r_[durations, delays],
                np.r_[arms, assigned],
            )
            event_times = arrivals + durations
            if np.any(~np.isfinite(event_times)) or np.any(
                (durations > 0) & (event_times <= arrivals)
            ):
                raise ArithmeticError("simulated event calendar time is not representable")
            stop = _stop_time(arrivals, event_times, pending, remaining_d, remaining_n, deadline)
        if not np.all(np.isfinite(pending)):
            raise ArithmeticError("pending event times overflow")
        future_n, future_d, future_e, final = _future_summary(
            arrivals, arms, durations, pending, pending_arms, stop, extra
        )
        ns[trial], ds[trial], es[trial] = n + future_n, d + future_d, e + future_e
        stops[trial], finals[trial] = stop, final
    comparison = compare_predictive_survival(
        ds,
        es,
        prior=pair,
        method=method,
        posterior_cutoff=posterior_cutoff,
        significance_level=significance_level,
    )
    codes = ("arm_a_superior", "arm_b_superior", "inconclusive", "not_evaluable")
    probability = np.array([np.mean(comparison.decision == code) for code in codes])
    return SurvivalPredictiveProbabilities(
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / simulations)),
        _owned(ns),
        _owned(ds),
        _owned(es),
        _owned(stops),
        _owned(finals),
        comparison.decision,
    )
