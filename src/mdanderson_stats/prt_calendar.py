"""Explicit-input calendar replay for predicted-risk toxicity trials.

The replay implements published cohort/suspension conduct rules but does not
invent the unavailable Appendix-B event-time generator or arrival law. Callers
supply arrival times and a potential-toxicity-delay table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .prt import PRTDecision, prt_decision, prt_final_selection, prt_predictive_risk
from .prt_fit import PRTIsotonicProjection, PRTModelFit, fit_prt_model, prt_isotonic_projection

_MAX_CANDIDATES = 500
_MAX_ANALYSES = 200
_MAX_RESULT_CELLS = 2_000_000
_MAX_FIT_CELLS = 2_000_000
_MAX_WORK = 20_000_000


@dataclass(frozen=True)
class PRTCalendarAnalysis:
    """Immutable summary of one cohort, suspension, or final analysis."""

    time: float
    reason: str
    current_dose: int
    action: str
    next_dose: int | None
    rule: str
    log_likelihood_at_posterior_mean: float
    likelihood_evaluations: int
    max_split_rhat: float
    survived: np.ndarray
    events: np.ndarray
    pending_by_interval: np.ndarray
    posterior_mean: FloatArray
    posterior_exceedance: FloatArray
    predictive_negligible: FloatArray
    predictive_excessive: FloatArray
    final_selection: int | None


@dataclass(frozen=True)
class PRTCalendarResult:
    """Calendar path and compact decision history from one explicit replay."""

    interval_endpoints: FloatArray
    arrival_times: FloatArray
    potential_toxicity_delays: FloatArray
    patient_candidate_index: np.ndarray
    patient_arrival_time: FloatArray
    patient_enrollment_time: FloatArray
    patient_dose: np.ndarray
    patient_toxicity_delay: FloatArray
    patient_event_time: FloatArray
    patient_event_interval: np.ndarray
    patient_completed_intervals: np.ndarray
    patient_event_observed: np.ndarray
    patient_enrolled: np.ndarray
    analyses: tuple[PRTCalendarAnalysis, ...]
    status: str
    selected_dose: int | None
    waiting_policy: str
    candidate_count: int
    enrolled_count: int
    declined_count: int
    queued_not_enrolled_count: int
    duration: float
    enrollment_end_reason: str
    enrollment_end_time: float | None
    stopping_time: float | None
    likelihood_evaluations: int
    work_units: int
    fit_count: int
    target: float
    negligible_cutoff: float
    excessive_cutoff: float
    epsilon: float
    prior_mean: FloatArray
    prior_variance: FloatArray
    draws: int
    warmup: int
    chains: int
    max_likelihood_evaluations: int
    max_work: int


def _freeze_int(value: ArrayLike) -> np.ndarray:
    out = np.array(value, dtype=np.int64, copy=True)
    out.setflags(write=False)
    return out


def _freeze_bool(value: ArrayLike) -> np.ndarray:
    out = np.array(value, dtype=np.bool_, copy=True)
    out.setflags(write=False)
    return out


def _bounded_finite(
    value: ArrayLike,
    name: str,
    maximum: int,
    ndim: int | tuple[int, ...],
    *,
    allow_positive_infinity: bool = False,
) -> FloatArray:
    allowed_ndim = (ndim,) if isinstance(ndim, int) else ndim
    if isinstance(value, np.ndarray):
        if value.ndim not in allowed_ndim or value.size > maximum:
            raise ValueError(f"{name} exceeds its bounded {ndim}D input size")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a numeric sequence")
        if not (0 in allowed_ndim and np.isscalar(value)):
            try:
                if len(value) > maximum:  # type: ignore[arg-type]
                    raise ValueError(f"{name} exceeds its bounded input size")
            except TypeError as exc:
                raise ValueError(f"{name} must be an array-like value") from exc
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        if 2 in allowed_ndim:
            rows = cast(Iterable[object], value)
            for row in rows:
                try:
                    if len(row) > 10:  # type: ignore[arg-type]
                        raise ValueError(f"{name} rows may contain at most 10 doses")
                    if any(not np.isscalar(item) for item in cast(Iterable[object], row)):
                        raise ValueError(f"{name} must be a flat rectangular matrix")
                except TypeError as exc:
                    raise ValueError(f"{name} must be a rectangular matrix") from exc
        elif 1 in allowed_ndim and not np.isscalar(value):
            if any(not np.isscalar(item) for item in cast(Iterable[object], value)):
                raise ValueError(f"{name} must be a flat vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    invalid = np.isnan(result) | np.isneginf(result)
    if not allow_positive_infinity:
        invalid |= np.isposinf(result)
    if result.ndim not in allowed_ndim or result.size > maximum or np.any(invalid):
        raise ValueError(f"{name} must be valid and within its bounded input size")
    return result


def _interval_counts(
    enrolled: list[dict[str, float | int | bool]],
    now: float,
    endpoints: FloatArray,
    dose_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    h = endpoints.size
    survived = np.zeros((h, dose_count), dtype=np.int64)
    events = np.zeros((h, dose_count), dtype=np.int64)
    pending = np.zeros((dose_count, h), dtype=np.int64)
    pending_patients: list[tuple[int, int]] = []
    for patient_index, patient in enumerate(enrolled):
        dose = int(patient["dose"])
        delay = float(patient["delay"])
        entry = float(patient["entry"])
        event_time = entry + delay if delay < endpoints[-1] else float("inf")
        event = delay < endpoints[-1] and now >= event_time
        if event:
            interval = int(np.searchsorted(endpoints, delay, side="right"))
            if interval >= h:
                raise ArithmeticError(
                    "in-window toxicity was mapped beyond the assessment intervals"
                )
            if interval:
                survived[:interval, dose] += 1
            events[interval, dose] += 1
            patient["event_seen"] = True
            patient["event_interval"] = interval
            patient["completed"] = interval
        else:
            boundaries = np.asarray([entry + float(value) for value in endpoints])
            complete = int(np.searchsorted(boundaries, now, side="right"))
            if complete:
                survived[:complete, dose] += 1
            patient["completed"] = complete
            if complete < h:
                pending[dose, complete] += 1
                pending_patients.append((patient_index, complete))
    return survived, events, pending, pending_patients


def run_prt_calendar(
    arrival_times: ArrayLike,
    potential_toxicity_delays: ArrayLike,
    *,
    interval_endpoints: ArrayLike,
    starting_dose: int,
    cohort_size: int,
    max_patients: int,
    waiting_policy: str,
    rng: np.random.Generator,
    target: float = 0.3,
    negligible_cutoff: float = 0.3,
    excessive_cutoff: float = 0.9,
    epsilon: float = 0.05,
    prior_mean: ArrayLike = -14.0,
    prior_variance: ArrayLike = 28.0,
    draws: int,
    warmup: int,
    chains: int,
    max_analyses: int = 100,
    max_likelihood_evaluations: int = 1000,
    max_work: int = _MAX_WORK,
) -> PRTCalendarResult:
    """Replay one PRT trial from explicit arrivals and dose-specific event delays.

    ``arrival_times`` contains one time per candidate in nondecreasing order.
    The delay matrix has shape ``(candidates,doses)``; infinity means no toxicity
    during the assessment window. Delays restart at enrollment. Arrivals during
    suspension are either queued FIFO or declined according to the required
    ``waiting_policy``. Follow-up/toxicity updates at a tied time are processed
    before arrivals; rules are evaluated after the full arriving cohort.

    Intervals are ``[0,t1), [t1,t2), ...``. A delay of zero is an event in the
    first interval; an event exactly at the final endpoint is outside the modeled
    toxicity window, following paper Section 2's half-open interval definition.
    On input exhaustion, a partial final cohort is analyzed. Final selection is
    made after every enrolled patient has an observed event or completes follow-up.
    This is a replay convention; the native executable's arrival, waiting and
    event-time simulation distributions are not reproduced.
    """
    if waiting_policy not in ("queue", "decline"):
        raise ValueError("waiting_policy must be 'queue' or 'decline'")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")

    def bounded_integer(name: str, value: int, lo: int, hi: int) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        try:
            number = scalar(value, name)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]") from exc
        if number != np.floor(number) or not lo <= number <= hi:
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        return int(number)

    cohort_size = bounded_integer("cohort_size", cohort_size, 1, _MAX_CANDIDATES)
    max_patients = bounded_integer("max_patients", max_patients, 1, _MAX_CANDIDATES)
    draws = bounded_integer("draws", draws, 8, 100_000)
    warmup = bounded_integer("warmup", warmup, 0, 100_000)
    chains = bounded_integer("chains", chains, 2, 16)
    max_analyses = bounded_integer("max_analyses", max_analyses, 1, _MAX_ANALYSES)
    max_likelihood_evaluations = bounded_integer(
        "max_likelihood_evaluations", max_likelihood_evaluations, 1, _MAX_WORK
    )
    max_work = bounded_integer("max_work", max_work, 1, _MAX_WORK)
    if max_patients % cohort_size:
        raise ValueError("max_patients must be divisible by cohort_size")

    # Reject oversized ndarray inputs before conversion/copying.
    arrivals = _bounded_finite(arrival_times, "arrival_times", _MAX_CANDIDATES, 1)
    if arrivals.ndim != 1 or not 1 <= arrivals.size <= _MAX_CANDIDATES:
        raise ValueError(f"arrival_times must contain 1..{_MAX_CANDIDATES} values")
    n_candidates = arrivals.size
    delays = _bounded_finite(
        potential_toxicity_delays,
        "potential_toxicity_delays",
        2_000_000,
        2,
        allow_positive_infinity=True,
    )
    if delays.shape[0] != n_candidates or not 2 <= delays.shape[1] <= 10:
        raise ValueError("delay matrix must be candidates x 2..10 doses")
    endpoints = _bounded_finite(interval_endpoints, "interval_endpoints", 10, 1)
    if (
        delays.ndim != 2
        or delays.shape[0] != n_candidates
        or not 2 <= delays.shape[1] <= 10
        or delays.size > 2_000_000
    ):
        raise ValueError("delay matrix must be candidates x 2..10 doses, within two million cells")
    doses = delays.shape[1]
    if np.any(np.isnan(delays)) or np.any(np.isneginf(delays)) or np.any(delays < 0):
        raise ValueError("toxicity delays must be nonnegative finite values or +infinity")
    if (
        endpoints.ndim != 1
        or not 1 <= endpoints.size <= 10
        or endpoints[0] <= 0
        or np.any(endpoints[1:] <= endpoints[:-1])
    ):
        raise ValueError(
            "interval_endpoints must be a strictly increasing positive vector of length 1..10"
        )
    if np.any(np.diff(arrivals) < 0) or arrivals[0] < 0:
        raise ValueError("arrival_times must be nonnegative and nondecreasing")
    start = scalar(starting_dose, "starting_dose")
    if start != np.floor(start) or not 0 <= start < doses:
        raise ValueError("starting_dose must be a valid zero-based dose index")
    target, lo, hi, eps = (
        scalar(target, "target"),
        scalar(negligible_cutoff, "negligible_cutoff"),
        scalar(excessive_cutoff, "excessive_cutoff"),
        scalar(epsilon, "epsilon"),
    )
    if not 0 < target < 1 or not 0 < lo < hi < 1 or not 0 <= eps <= 1:
        raise ValueError("invalid PRT target/cutoff settings")
    pri_mean = _bounded_finite(prior_mean, "prior_mean", 10, (0, 1))
    pri_var = _bounded_finite(prior_variance, "prior_variance", 10, (0, 1))
    if pri_mean.ndim > 1 or pri_var.ndim > 1:
        raise ValueError("prior mean and variance must be scalars or interval vectors")
    try:
        pri_mean = np.broadcast_to(pri_mean, (endpoints.size,))
        pri_var = np.broadcast_to(pri_var, (endpoints.size,))
    except ValueError as exc:
        raise ValueError("prior mean/variance must be scalar or one value per interval") from exc
    if np.any(pri_var <= 0):
        raise ValueError("prior_variance must be positive")
    fit_cells = 8 * chains * draws * endpoints.size * doses
    if fit_cells > _MAX_FIT_CELLS:
        raise ValueError("posterior, risk and projection arrays exceed the fit cell budget")
    maximum_enrolled = min(max_patients, n_candidates)
    analysis_bound = 2 * n_candidates + n_candidates * endpoints.size + maximum_enrolled + 1
    if analysis_bound > max_analyses:
        raise ValueError(
            f"max_analyses is below the conservative calendar update bound ({analysis_bound})"
        )
    possible_analyses = analysis_bound
    history_cells = possible_analyses * (10 * doses + 3 * endpoints.size * doses)
    input_cells = 2 * arrivals.size + 2 * delays.size + endpoints.size
    if fit_cells + history_cells + input_cells > _MAX_RESULT_CELLS:
        raise ValueError(
            "calendar inputs, fit and analysis history exceed the combined cell budget"
        )
    sample_count = chains * draws
    if sample_count * max_patients > _MAX_FIT_CELLS:
        raise ValueError("pending predictive-risk matrix exceeds the fit cell budget")
    minimum_fit_work = (
        chains * (draws + warmup) * endpoints.size * doses
        + sample_count * endpoints.size * doses**3
    )
    minimum_prediction_work = sample_count * doses
    if minimum_fit_work + minimum_prediction_work > max_work:
        raise ValueError("max_work is too small for one fit, projection and analysis")

    survived = np.zeros((endpoints.size, doses), dtype=np.int64)
    events = np.zeros_like(survived)
    enrolled: list[dict[str, float | int | bool]] = []
    ledger: list[tuple[int, float, float, int, float]] = []
    analyses: list[PRTCalendarAnalysis] = []
    queue: list[int] = []
    arrival_index = 0
    current_dose = int(start)
    cohort_count = 0
    suspended = False
    status = "arrival_tape_exhausted"
    selected: int | None = None
    fit_count = total_evaluations = total_work = 0
    cached_counts: tuple[np.ndarray, np.ndarray] | None = None
    cached_fit: PRTModelFit | None = None
    cached_projection: PRTIsotonicProjection | None = None
    cached_total: FloatArray | None = None
    time_origin = float(arrivals[0])
    last_time = time_origin
    declined_count = 0
    enrollment_reason = "arrival_tape_exhausted"
    enrollment_end_time: float | None = None
    stop_time: float | None = None

    def analyze(now: float, reason: str) -> PRTDecision:
        nonlocal current_dose, stop_time, enrollment_end_time, enrollment_reason
        nonlocal cached_counts, cached_fit, cached_projection, cached_total
        nonlocal fit_count, total_evaluations, total_work, status, selected, suspended
        if len(analyses) >= max_analyses:
            raise ArithmeticError("calendar analysis-count budget exceeded")
        s, e, pending, pending_patients = _interval_counts(enrolled, now, endpoints, doses)
        same = (
            cached_counts is not None
            and np.array_equal(cached_counts[0], s)
            and np.array_equal(cached_counts[1], e)
        )
        prediction_work = (
            chains * draws * sum((int(np.sum(pending[dose])) + 1) ** 2 for dose in range(doses))
        )
        if total_work + prediction_work > max_work:
            raise ArithmeticError("calendar predictive-risk work budget exhausted before analysis")
        if not same:
            cached_fit = None
            cached_projection = None
            cached_total = None
            fixed_fit_work = (
                chains * (draws + warmup) * endpoints.size * doses
                + chains * draws * endpoints.size * doses**3
            )
            available_work = max_work - total_work - fixed_fit_work - prediction_work
            if available_work < 0:
                raise ArithmeticError("calendar fit/projection work budget exhausted before refit")
            available_evaluations = max_likelihood_evaluations - total_evaluations
            per_fit_eval = min(
                available_evaluations,
                max(0, available_work // (endpoints.size * doses)),
            )
            if per_fit_eval < 1 and np.any(s + e):
                raise ArithmeticError("calendar cumulative fit-work budget exhausted before refit")
            per_fit_eval = max(1, per_fit_eval)
            try:
                fit = fit_prt_model(
                    s,
                    e,
                    prior_mean=pri_mean,
                    prior_variance=pri_var,
                    draws=int(draws),
                    warmup=int(warmup),
                    chains=int(chains),
                    rng=rng,
                    max_likelihood_evaluations=int(per_fit_eval),
                )
                raw = fit.conditional_toxicity[..., :-1, :].reshape(-1, endpoints.size, doses)
                projection = prt_isotonic_projection(raw)
            except (ValueError, ArithmeticError) as exc:
                raise ArithmeticError(
                    f"PRT posterior/projection failed at time {now:g} ({reason}): {exc}"
                ) from exc
            fit_count += 1
            total_evaluations += fit.likelihood_evaluations
            fit_work = (
                chains * (draws + warmup) * endpoints.size * doses
                + fit.likelihood_evaluations * endpoints.size * doses
                + chains * draws * endpoints.size * doses**3
            )
            total_work += fit_work
            cached_fit = fit
            cached_projection = projection
            cached_total = projection.probability[:, 0, :]
            cached_counts = (s.copy(), e.copy())
        assert cached_fit is not None and cached_projection is not None and cached_total is not None
        posterior_mean = cached_total.mean(axis=0)
        exceedance = (cached_total > target).mean(axis=0)
        predictive_negligible = np.empty(doses)
        predictive_excessive = np.empty(doses)
        for dose in range(doses):
            columns = [
                cached_projection.probability[:, completed, int(enrolled[index]["dose"])]
                for index, completed in pending_patients
                if int(enrolled[index]["dose"]) == dose
            ]
            future = np.column_stack(columns) if columns else np.empty((cached_total.shape[0], 0))
            risk = prt_predictive_risk(
                cached_total[:, dose],
                future,
                target=target,
                negligible_cutoff=lo,
                excessive_cutoff=hi,
            )
            predictive_negligible[dose] = risk.decision_negligible
            predictive_excessive[dose] = risk.decision_excessive
        total_work += prediction_work
        decision = (
            PRTDecision("stay", current_dose, "7")
            if reason == "final"
            else prt_decision(
                exceedance,
                predictive_negligible,
                predictive_excessive,
                np.sum(pending, axis=1),
                current_dose,
                negligible_cutoff=lo,
                excessive_cutoff=hi,
                epsilon=eps,
            )
        )
        total_evaluations_here = cached_fit.likelihood_evaluations
        reported_rhat = cached_fit.beta_summary.split_rhat
        nonmissing_rhat = reported_rhat[~np.isnan(reported_rhat)]
        rhat = float(np.max(nonmissing_rhat)) if nonmissing_rhat.size else float("nan")
        mean_beta = cached_fit.beta_summary.mean
        analysis = PRTCalendarAnalysis(
            now,
            reason,
            current_dose,
            decision.action,
            decision.dose,
            decision.rule,
            _fit_loglik(s, e, mean_beta),
            total_evaluations_here,
            rhat,
            _freeze_int(s),
            _freeze_int(e),
            _freeze_int(pending),
            _freeze(posterior_mean),
            _freeze(exceedance),
            _freeze(predictive_negligible),
            _freeze(predictive_excessive),
            None,
        )
        analyses.append(analysis)
        if decision.action == "stop":
            status = "early_stop"
            stop_time = now
            enrollment_end_time = now
            enrollment_reason = "early_stop"
            selected = None
            suspended = True
        else:
            current_dose = decision.dose if decision.dose is not None else current_dose
            suspended = decision.action == "suspend"
        return decision

    def final_analysis(now: float) -> None:
        nonlocal selected, status
        # Final assessment is after all currently enrolled records resolve.
        s, e, _, _ = _interval_counts(enrolled, now, endpoints, doses)
        decision = analyze(now, "final")
        if decision.action == "stop":
            status = "early_stop"
            selected = None
            return
        assert cached_total is not None
        means = cached_total.mean(axis=0)
        selected = prt_final_selection(
            (cached_total > target).mean(axis=0), means, target=target, excessive_cutoff=hi
        )
        last = analyses[-1]
        analyses[-1] = PRTCalendarAnalysis(
            last.time,
            last.reason,
            last.current_dose,
            "final",
            last.next_dose,
            "7",
            last.log_likelihood_at_posterior_mean,
            last.likelihood_evaluations,
            last.max_split_rhat,
            last.survived,
            last.events,
            last.pending_by_interval,
            last.posterior_mean,
            last.posterior_exceedance,
            last.predictive_negligible,
            last.predictive_excessive,
            selected,
        )
        status = "completed" if selected is not None else "no_acceptable_dose"

    def enroll(candidate: int, now: float, dose: int) -> bool:
        nonlocal cohort_count, status, enrollment_reason, enrollment_end_time
        if len(enrolled) >= max_patients:
            status = "maximum_enrollment"
            enrollment_reason = "maximum_enrollment"
            enrollment_end_time = now
            return False
        delay = float(delays[candidate, dose])
        for endpoint in endpoints:
            if not np.isfinite(now + float(endpoint)) or now + float(endpoint) <= now:
                raise ArithmeticError("assessment endpoint is outside calendar-time resolution")
        if (
            delay < endpoints[-1]
            and delay > 0
            and (not np.isfinite(now + delay) or now + delay <= now)
        ):
            raise ArithmeticError("positive toxicity delay is outside calendar-time resolution")
        entry = {
            "entry": now,
            "dose": dose,
            "delay": delay,
            "event_seen": False,
            "event_interval": -1,
            "completed": 0,
        }
        enrolled.append(entry)
        ledger.append((candidate, float(arrivals[candidate]), now, dose, delay))
        cohort_count += 1
        return True

    def close_cohort(now: float, reason: str) -> None:
        nonlocal cohort_count, status, stop_time, enrollment_reason, enrollment_end_time
        if cohort_count:
            cohort_count = 0
            decision = analyze(now, reason)
            if decision.action == "stop":
                stop_time = now
                return
            if len(enrolled) >= max_patients:
                status = "maximum_enrollment"
                enrollment_reason = "maximum_enrollment"
                enrollment_end_time = now

    def release_queue(now: float) -> None:
        """Enroll queued patients in FIFO cohorts while accrual remains open."""
        nonlocal cohort_count
        while not suspended and queue and len(enrolled) < max_patients:
            candidate = queue.pop(0)
            if not enroll(candidate, now, current_dose):
                break
            if cohort_count >= cohort_size or len(enrolled) >= max_patients:
                close_cohort(now, "reopened_cohort")
                if suspended or status == "early_stop":
                    break
        if arrival_index >= n_candidates and queue == [] and cohort_count:
            close_cohort(now, "partial_final_cohort")

    def refresh(now: float) -> bool:
        """Advance every patient to now; return whether counts changed."""
        before = (survived.copy(), events.copy())
        s, e, _, _ = _interval_counts(enrolled, now, endpoints, doses)
        survived[:] = s
        events[:] = e
        return not (np.array_equal(before[0], survived) and np.array_equal(before[1], events))

    while True:
        if status == "early_stop":
            break
        next_times: list[float] = []
        if arrival_index < n_candidates:
            next_times.append(float(arrivals[arrival_index]))
        for patient in enrolled:
            entry = float(patient["entry"])
            completed = int(patient["completed"])
            if bool(patient["event_seen"]):
                continue
            delay = float(patient["delay"])
            if delay < endpoints[-1] and entry + delay > last_time:
                next_times.append(entry + delay)
            if completed < endpoints.size:
                next_times.append(entry + float(endpoints[completed]))
        if not next_times:
            break
        now = min(next_times)
        if now < last_time:
            raise ArithmeticError("calendar event order moved backward")
        last_time = now
        data_changed = refresh(now)
        arrivals_now: list[int] = []
        while arrival_index < n_candidates and float(arrivals[arrival_index]) == now:
            arrivals_now.append(arrival_index)
            arrival_index += 1
        # Conduct decisions while suspended are reconsidered after the batch of
        # same-time updates and arrivals, never using future latent outcomes.
        if suspended:
            if waiting_policy == "queue":
                queue.extend(arrivals_now)
            elif arrivals_now:
                declined_count += len(arrivals_now)
            if data_changed or arrivals_now:
                decision = analyze(now, "suspension_update" if data_changed else "arrival")
                if decision.action == "stop":
                    break
                if not suspended:
                    release_queue(now)
        else:
            for position, candidate in enumerate(arrivals_now):
                if suspended:
                    if waiting_policy == "queue":
                        queue.append(candidate)
                    else:
                        declined_count += 1
                    continue
                if len(enrolled) >= max_patients:
                    enrollment_reason = "maximum_enrollment"
                    declined_count += len(arrivals_now) - position
                    break
                if not enroll(candidate, now, current_dose):
                    declined_count += len(arrivals_now) - position
                    break
                if cohort_count >= cohort_size or len(enrolled) >= max_patients:
                    close_cohort(now, "cohort_complete")
                    if status == "early_stop":
                        declined_count += len(arrivals_now) - position - 1
                        break
            if (
                arrival_index == n_candidates
                and not queue
                and cohort_count
                and status != "early_stop"
            ):
                close_cohort(now, "partial_final_cohort")
        if len(enrolled) >= max_patients and not cohort_count:
            if status != "early_stop":
                enrollment_reason = "maximum_enrollment"
                enrollment_end_time = now
                status = enrollment_reason
            break
        if arrival_index == n_candidates and not enrolled:
            status = "no_enrollment"
            enrollment_end_time = now
            break
        if (
            arrival_index == n_candidates
            and not queue
            and cohort_count == 0
            and status == "arrival_tape_exhausted"
            and enrollment_end_time is None
        ):
            # Enrollment ends when the last arrival has been handled, not when
            # the last enrolled patient’s follow-up window later completes.
            enrollment_end_time = now
            enrollment_reason = "arrival_tape_exhausted"
        if arrival_index == n_candidates and not next_times and status == "arrival_tape_exhausted":
            break

    if enrollment_end_time is None:
        enrollment_end_time = last_time
        if queue:
            enrollment_reason = "arrival_tape_exhausted_with_queued_candidates"
        else:
            enrollment_reason = "arrival_tape_exhausted"

    # If enrollment ended normally or the arrival tape was exhausted, observe
    # every enrolled patient through their event or full window before Rule 7.
    if status not in ("no_enrollment",) and enrolled:
        end_time = max(
            float(p["entry"])
            + (float(p["delay"]) if float(p["delay"]) < endpoints[-1] else float(endpoints[-1]))
            for p in enrolled
        )
        refresh(end_time)
        if status != "early_stop":
            final_analysis(end_time)
        last_time = max(last_time, end_time)

    candidates = np.asarray([row[0] for row in ledger], dtype=np.int64)
    entry_times = np.asarray([row[2] for row in ledger], dtype=np.float64)
    doses_actual = np.asarray([row[3] for row in ledger], dtype=np.int64)
    delays_actual = np.asarray([row[4] for row in ledger], dtype=np.float64)
    event_times = np.full(len(enrolled), np.nan)
    event_intervals = np.full(len(enrolled), -1, dtype=np.int64)
    completed_out = np.zeros(len(enrolled), dtype=np.int64)
    event_observed = np.zeros(len(enrolled), dtype=np.bool_)
    for i, patient in enumerate(enrolled):
        if bool(patient["event_seen"]):
            event_observed[i] = True
            event_times[i] = float(patient["entry"]) + float(patient["delay"])
            event_intervals[i] = int(patient["event_interval"])
            completed_out[i] = int(patient["completed"])
        elif last_time >= float(patient["entry"]) + endpoints[-1]:
            completed_out[i] = endpoints.size
        else:
            completed_out[i] = int(patient["completed"])
    if arrival_index < n_candidates:
        declined_count += n_candidates - arrival_index
    followup_offsets = [
        (float(patient["entry"]) - time_origin)
        + (
            float(patient["delay"])
            if float(patient["delay"]) < endpoints[-1]
            else float(endpoints[-1])
        )
        for patient in enrolled
    ]
    duration_candidates = [0.0, *followup_offsets]
    if enrollment_end_time is not None:
        duration_candidates.append(float(enrollment_end_time - time_origin))
    if stop_time is not None:
        duration_candidates.append(float(stop_time - time_origin))
    duration = max(duration_candidates)
    return PRTCalendarResult(
        _freeze(endpoints),
        _freeze(arrivals),
        _freeze(delays),
        _freeze_int(candidates),
        _freeze(np.asarray([arrivals[i] for i in candidates], dtype=float)),
        _freeze(entry_times),
        _freeze_int(doses_actual),
        _freeze(delays_actual),
        _freeze(event_times),
        _freeze_int(event_intervals),
        _freeze_int(completed_out),
        _freeze_bool(event_observed),
        _freeze_bool(np.ones(len(enrolled), dtype=bool)),
        tuple(analyses),
        status,
        selected,
        waiting_policy,
        n_candidates,
        len(enrolled),
        declined_count,
        len(queue),
        float(duration),
        enrollment_reason,
        None if enrollment_end_time is None else float(enrollment_end_time - time_origin),
        None if stop_time is None else float(stop_time - time_origin),
        total_evaluations,
        total_work,
        fit_count,
        target,
        lo,
        hi,
        eps,
        _freeze(pri_mean),
        _freeze(pri_var),
        draws,
        warmup,
        chains,
        max_likelihood_evaluations,
        max_work,
    )


def _fit_loglik(survived: np.ndarray, events: np.ndarray, beta: FloatArray) -> float:
    """Evaluate the sampled PRT likelihood at posterior mean beta for reporting."""
    from scipy.special import log_ndtr

    positive = np.multiply(log_ndtr(beta), events, out=np.zeros_like(beta), where=events > 0)
    negative = np.multiply(log_ndtr(-beta), survived, out=np.zeros_like(beta), where=survived > 0)
    value = float(np.sum(positive + negative))
    if not isfinite(value):
        raise ArithmeticError("PRT reported likelihood is not finite")
    return value
