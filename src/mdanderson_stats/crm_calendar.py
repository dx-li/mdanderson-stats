"""Retrospective patient snapshots and posterior routing for CRM designs."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._tite_calendar import _advance
from ._validation import FloatArray
from .bmacrm import BMACRMPosterior, _raw_numeric, _scalar_value, fit_bmacrm
from .bmacrm_decision import BMACRMDecision, bmacrm_decision
from .bmacrm_lookahead import BMACRMLookAhead, bmacrm_lookahead
from .dacrm import DACRMPosterior, DACRMPrior, fit_dacrm
from .dacrm_decision import DACRMDecision, dacrm_decision

_MAX_PATIENTS = 200
_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_MAX_BMA_EVALUATIONS = 200_000
_MAX_DA_EVALUATIONS = 2_000_000
_MAX_TOTAL_EVALUATIONS = 2_000_000


@dataclass(frozen=True)
class CRMCalendarSnapshot:
    """Immutable as-of patient rows and dose-level sufficient counts."""

    row_indices: NDArray[np.int64]
    doses: NDArray[np.int64]
    outcomes: NDArray[np.int64]
    times: FloatArray
    treated_counts: NDArray[np.int64]
    observed_counts: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    pending_counts: NDArray[np.int64]
    window: float
    at: float


@dataclass(frozen=True)
class CRMCalendarDecision:
    """A snapshot, its fitted posterior, and the selected CRM decision."""

    snapshot: CRMCalendarSnapshot
    posterior: BMACRMPosterior | DACRMPosterior
    decision: BMACRMDecision | BMACRMLookAhead | DACRMDecision
    routing: Literal["complete", "lookahead", "data_augmentation"]
    evaluations: int


def _integer_vector(value: ArrayLike, name: str, *, limit: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size > limit or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector with at most {limit} values")
    values = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain finite integer values")
    if np.any(np.abs(values) > 2**53):
        raise ValueError(f"{name} contains values too large to represent exactly")
    return values.astype(np.int64)


def _freeze_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(values.tobytes(), dtype=np.int64).reshape(values.shape)


def crm_calendar_snapshot(
    doses: ArrayLike,
    enrollment_times: ArrayLike,
    dlt_delays: ArrayLike,
    *,
    window: float,
    at: float,
    dose_count: int,
) -> CRMCalendarSnapshot:
    """Replay fully ascertained patient records into an as-of snapshot.

    ``dlt_delays`` are times from enrollment to observed DLT, or positive
    infinity for no DLT through the full toxicity window. Only patients
    enrolled by ``at`` are retained. A pending patient's reported time is
    elapsed follow-up; a completed DLT uses its event delay.
    """
    raw_dose_count = np.asarray(dose_count)
    if (
        raw_dose_count.ndim != 0
        or raw_dose_count.dtype.kind not in "iu"
        or isinstance(dose_count, (bool, np.bool_))
        or not 1 <= int(dose_count) <= _MAX_DOSES
    ):
        raise ValueError("dose_count must be an integer from 1 to 20")
    dose_count_value = int(dose_count)
    dose_values = _integer_vector(doses, "doses", limit=_MAX_PATIENTS)
    enrollment = _raw_numeric(enrollment_times, "enrollment_times", _MAX_PATIENTS)
    delays = np.asarray(dlt_delays)
    if delays.size > _MAX_PATIENTS or delays.ndim != 1 or delays.dtype.kind not in "iuf":
        raise ValueError("dlt_delays must be a numeric vector with at most 200 values")
    delays = np.asarray(delays, dtype=float)
    if dose_values.shape != enrollment.shape or delays.shape != enrollment.shape:
        raise ValueError("doses, enrollment_times, and dlt_delays must have equal lengths")
    if np.any(dose_values < 0) or np.any(dose_values >= dose_count_value):
        raise ValueError("doses must be zero-based indices within dose_count")
    if np.any(enrollment < 0) or np.any(np.diff(enrollment) < 0):
        raise ValueError("enrollment_times must be nonnegative and nondecreasing")
    if np.any(np.isnan(delays) | np.isneginf(delays) | (delays < 0)):
        raise ValueError("dlt_delays must be in [0,window] or positive infinity")
    window_value = _scalar_value(window, "window")
    at_value = _scalar_value(at, "at")
    if not np.isfinite(window_value) or window_value <= 0:
        raise ValueError("window must be positive and finite")
    if not np.isfinite(at_value) or at_value < 0:
        raise ValueError("at must be nonnegative and finite")
    if np.any(delays[np.isfinite(delays)] > window_value):
        raise ValueError("finite dlt_delays must not exceed the toxicity window")

    retained = np.flatnonzero(enrollment <= at_value)
    result_doses = dose_values[retained]
    outcomes = np.empty(retained.size, dtype=np.int64)
    times = np.empty(retained.size, dtype=float)
    for output_index, row_index in enumerate(retained):
        entry = float(enrollment[row_index])
        age = at_value - entry
        if age < 0 or not np.isfinite(age):
            raise ArithmeticError("calendar follow-up is not representable; rescale time units")
        # Compare calendar endpoints so a subtraction rounded just below the
        # delay/window cannot hide an event or completed no-DLT assessment.
        window_end = _advance(entry, window_value)
        delay = float(delays[row_index])
        event_end = _advance(entry, delay) if np.isfinite(delay) else np.inf
        if event_end <= at_value:
            outcomes[output_index] = 1
            times[output_index] = delay
        elif window_end <= at_value:
            outcomes[output_index] = 0
            times[output_index] = window_value
        else:
            if not 0 <= age < window_value:
                raise ArithmeticError("pending follow-up is not representable; rescale time units")
            _advance(entry, age)
            outcomes[output_index] = -1
            times[output_index] = age

    treated = np.bincount(result_doses, minlength=int(dose_count)).astype(np.int64)
    observed = np.bincount(result_doses[outcomes != -1], minlength=int(dose_count)).astype(np.int64)
    toxicities = np.bincount(result_doses[outcomes == 1], minlength=int(dose_count)).astype(
        np.int64
    )
    pending = np.bincount(result_doses[outcomes == -1], minlength=int(dose_count)).astype(np.int64)
    return CRMCalendarSnapshot(
        _freeze_int(retained.astype(np.int64)),
        _freeze_int(result_doses),
        _freeze_int(outcomes),
        _freeze(times),
        _freeze_int(treated),
        _freeze_int(observed),
        _freeze_int(toxicities),
        _freeze_int(pending),
        window_value,
        at_value,
    )


def _validate_snapshot(snapshot: CRMCalendarSnapshot, dose_count: int) -> None:
    if not isinstance(snapshot, CRMCalendarSnapshot):
        raise TypeError("snapshot must be a CRMCalendarSnapshot")
    if (
        not 1 <= dose_count <= _MAX_DOSES
        or not np.isfinite(snapshot.window)
        or snapshot.window <= 0
    ):
        raise ValueError("snapshot window or dose_count is invalid")
    if not np.isfinite(snapshot.at) or snapshot.at < 0:
        raise ValueError("snapshot time is invalid")
    rows = np.asarray(snapshot.row_indices)
    doses = np.asarray(snapshot.doses)
    outcomes = np.asarray(snapshot.outcomes)
    times = np.asarray(snapshot.times, dtype=float)
    if (
        rows.ndim != 1
        or rows.size > _MAX_PATIENTS
        or doses.shape != rows.shape
        or outcomes.shape != rows.shape
        or times.shape != rows.shape
        or rows.dtype.kind not in "iu"
        or doses.dtype.kind not in "iu"
        or outcomes.dtype.kind not in "iu"
    ):
        raise ValueError("snapshot patient rows have inconsistent shapes or types")
    if (
        np.any(rows < 0)
        or np.any(rows >= _MAX_PATIENTS)
        or np.any(np.diff(rows) <= 0)
        or np.any((doses < 0) | (doses >= dose_count))
        or np.any(~np.isin(outcomes, [-1, 0, 1]))
        or np.any(~np.isfinite(times))
        or np.any(times < 0)
        or np.any((outcomes == -1) & (times >= snapshot.window))
        or np.any((outcomes == 0) & (times != snapshot.window))
        or np.any((outcomes == 1) & (times > snapshot.window))
    ):
        raise ValueError("snapshot contains invalid retained patient data")
    for name, mask in (
        ("treated_counts", np.ones(rows.size, dtype=bool)),
        ("observed_counts", outcomes != -1),
        ("toxicities", outcomes == 1),
        ("pending_counts", outcomes == -1),
    ):
        counts = np.asarray(getattr(snapshot, name))
        expected = np.bincount(doses[mask], minlength=dose_count)
        if (
            counts.shape != (dose_count,)
            or counts.dtype.kind not in "iu"
            or not np.array_equal(counts, expected)
        ):
            raise ValueError(f"snapshot {name} do not match retained patient rows")


def _integer_setting(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


def crm_calendar_decision(
    snapshot: CRMCalendarSnapshot,
    skeletons: ArrayLike,
    *,
    target: float,
    method: Literal["bmacrm", "dacrm"] = "bmacrm",
    da_prior: DACRMPrior | None = None,
    model_prior: ArrayLike | None = None,
    prior_sd: float | None = None,
    aggregation: Literal["bma", "bms", "occam"] = "bma",
    occam_threshold: float | None = None,
    current_dose: int | None = None,
    starting_dose: int = 0,
    safety_cutoff: float = 0.9,
    minimum_observed: int | None = None,
    final: bool = False,
    rng: np.random.Generator | None = None,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_completions: int = 128,
    max_evaluations: int = 200_000,
) -> CRMCalendarDecision:
    """Fit and route a retrospective snapshot to ordinary CRM or DA-CRM.

    This function performs one as-of analysis only. It does not advance a
    trial calendar, enforce cohort timing, or infer final-study timing.
    """
    if method not in {"bmacrm", "dacrm"}:
        raise ValueError("method must be 'bmacrm' or 'dacrm'")
    target_value = _scalar_value(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    final_value = bool(final)
    raw_skeletons = _raw_numeric(skeletons, "skeletons", 5 * _MAX_DOSES)
    dose_count = raw_skeletons.shape[-1] if raw_skeletons.ndim in (1, 2) else 0
    _validate_snapshot(snapshot, dose_count)
    budget_limit = _integer_setting(max_evaluations, "max_evaluations", 1, _MAX_DA_EVALUATIONS)
    completion_limit = _integer_setting(max_completions, "max_completions", 1, 1024)
    if current_dose is None and snapshot.doses.size:
        current_dose = int(snapshot.doses[-1])
    if current_dose is None and (snapshot.doses.size or np.any(snapshot.pending_counts)):
        raise ValueError("current_dose cannot be inferred from an empty snapshot")
    if current_dose is not None:
        current_dose = _integer_setting(current_dose, "current_dose", 0, dose_count - 1)
    start = _integer_setting(starting_dose, "starting_dose", 0, dose_count - 1)
    cutoff = _scalar_value(safety_cutoff, "safety_cutoff")
    if not 0 <= cutoff <= 1:
        raise ValueError("safety_cutoff must lie in [0,1]")
    if method == "bmacrm":
        if da_prior is not None or minimum_observed is not None:
            raise ValueError("da_prior and minimum_observed apply only to method='dacrm'")
        if prior_sd is not None:
            sd_value = _scalar_value(prior_sd, "prior_sd")
        else:
            sd_value = sqrt(2)
        if not 1e-3 <= sd_value <= 10:
            raise ValueError("prior_sd must lie in [1e-3,10]")
        if max_evaluations > _MAX_TOTAL_EVALUATIONS:
            raise ValueError("BMA look-ahead max_evaluations must not exceed 2000000")
        observed = snapshot.observed_counts
        events = snapshot.toxicities
        remaining = budget_limit
        posterior = fit_bmacrm(
            raw_skeletons,
            events,
            observed,
            target=target_value,
            model_prior=model_prior,
            prior_sd=sd_value,
            max_evaluations=min(_MAX_BMA_EVALUATIONS, remaining),
            aggregation=aggregation,
            occam_threshold=occam_threshold,
        )
        evaluations = posterior.evaluations
        pending = snapshot.pending_counts
        if np.any(pending):
            remaining_budget = budget_limit - evaluations
            if remaining_budget < 1:
                raise RuntimeError(
                    "CRM calendar exhausted its BMA evaluation budget before look-ahead"
                )
            lookahead = bmacrm_lookahead(
                posterior,
                pending,
                current_dose=current_dose,
                starting_dose=start,
                safety_cutoff=cutoff,
                final=final_value,
                max_completions=completion_limit,
                max_evaluations=min(_MAX_TOTAL_EVALUATIONS, remaining_budget),
            )
            evaluations += lookahead.evaluations
            routing: Literal["complete", "lookahead", "data_augmentation"] = "lookahead"
            decision: BMACRMDecision | BMACRMLookAhead = lookahead
        else:
            decision = bmacrm_decision(
                posterior,
                current_dose=current_dose,
                starting_dose=start,
                safety_cutoff=cutoff,
                final=final_value,
            )
            routing = "complete"
        return CRMCalendarDecision(snapshot, posterior, decision, routing, evaluations)

    if model_prior is not None or prior_sd is not None:
        raise ValueError("model_prior and prior_sd apply only to method='bmacrm'")
    if aggregation != "bma" or occam_threshold is not None:
        raise ValueError("non-default aggregation options apply only to method='bmacrm'")
    if raw_skeletons.ndim == 2 and raw_skeletons.shape[0] != 1:
        raise ValueError("method='dacrm' requires exactly one dose skeleton")
    if da_prior is None or not isinstance(da_prior, DACRMPrior):
        raise ValueError("method='dacrm' requires an explicit DACRMPrior")
    if da_prior.breaks[-1] != snapshot.window:
        raise ValueError("DA-CRM prior endpoint must equal snapshot.window")
    if minimum_observed is None:
        raise ValueError("method='dacrm' requires explicit minimum_observed")
    minimum = _integer_setting(minimum_observed, "minimum_observed", 0, _MAX_PATIENTS)
    events = snapshot.toxicities
    observed = snapshot.observed_counts
    pending = snapshot.pending_counts
    if not np.any(pending):
        posterior = fit_bmacrm(
            raw_skeletons,
            events,
            observed,
            target=target_value,
            prior_sd=da_prior.alpha_sd,
            max_evaluations=min(_MAX_BMA_EVALUATIONS, budget_limit),
        )
        decision = bmacrm_decision(
            posterior,
            current_dose=current_dose,
            starting_dose=start,
            safety_cutoff=cutoff,
            final=final_value,
        )
        return CRMCalendarDecision(snapshot, posterior, decision, "complete", posterior.evaluations)

    if rng is None or not isinstance(rng, np.random.Generator):
        raise ValueError("pending DA-CRM snapshots require a numpy.random.Generator")
    posterior_da = fit_dacrm(
        raw_skeletons[0] if raw_skeletons.ndim == 2 else raw_skeletons,
        snapshot.doses,
        snapshot.outcomes,
        snapshot.times,
        prior=da_prior,
        target=target_value,
        rng=rng,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_evaluations=min(_MAX_DA_EVALUATIONS, budget_limit),
    )
    decision_da = dacrm_decision(
        posterior_da,
        current_dose=current_dose,
        starting_dose=start,
        policy="crm_suite",
        safety_cutoff=cutoff,
        minimum_observed=minimum,
        final=final_value,
    )
    return CRMCalendarDecision(
        snapshot, posterior_da, decision_da, "data_augmentation", posterior_da.evaluations
    )
