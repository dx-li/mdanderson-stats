"""Serial fixed-cohort CRM conduct replay using as-of calendar decisions."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._tite_calendar import _advance
from ._validation import FloatArray, scalar
from .bmacrm import _raw_numeric
from .bmacrm_decision import BMACRMDecision
from .bmacrm_lookahead import BMACRMLookAhead
from .crm_calendar import (
    CRMCalendarDecision,
    CRMCalendarSnapshot,
    crm_calendar_decision,
    crm_calendar_snapshot,
)
from .dacrm import (
    _MAX_RETAINED_CELLS,
    _MAX_TRANSITION_WORK,
    DACRMPosterior,
    DACRMPrior,
)
from .dacrm import (
    _integer_setting as _da_integer_setting,
)
from .dacrm_decision import DACRMDecision

_MAX_PATIENTS = 200
_MAX_DOSES = 20
_MAX_COHORT_SIZE = 4
_DEFAULT_PER_CALL_EVALUATIONS = 200_000
_DEFAULT_TOTAL_EVALUATIONS = 2_000_000
_MAX_PER_CALL_EVALUATIONS = 2_000_000
_MAX_TOTAL_EVALUATIONS = 20_000_000


@dataclass(frozen=True)
class CRMTrialStep:
    """Compact record of one interim, retry, or final decision."""

    time: float
    current_dose: int | None
    action: str
    dose: int | None
    routing: str
    reason: str
    treated_counts: NDArray[np.int64]
    observed_counts: NDArray[np.int64]
    pending_counts: NDArray[np.int64]
    safety_probability: float
    high_uncertainty: bool
    evaluations: int
    max_dose_mcse: float | None
    max_split_rhat: float | None


@dataclass(frozen=True)
class CRMTrial:
    """Immutable result for one fixed-cohort CRM trial replay."""

    assigned_doses: NDArray[np.int64]
    enrollment_times: FloatArray
    dlt_times: FloatArray
    treated_counts: NDArray[np.int64]
    observed_counts: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: int | None
    stop_reason: str
    decision_time: float
    final_time: float
    suspension_time: float
    evaluations: int
    max_dose_mcse: float | None
    max_split_rhat: float | None
    steps: tuple[CRMTrialStep, ...]


def _freeze_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(values.tobytes(), dtype=np.int64).reshape(values.shape)


def _count_setting(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


def _diagnostic_max(values: FloatArray) -> float:
    if values.size == 0 or np.any(~np.isfinite(values)):
        return float("nan")
    return float(np.max(values))


def _step(
    result: CRMCalendarDecision,
    snapshot: CRMCalendarSnapshot,
    current_dose: int | None,
) -> CRMTrialStep:
    decision = result.decision
    reason: str
    if isinstance(decision, BMACRMLookAhead):
        action, dose, reason = decision.action, decision.dose, decision.reason
        safety_probability = (
            float(result.posterior.overdose_probability[0])
            if hasattr(result.posterior, "overdose_probability")
            else float("nan")
        )
        high_uncertainty = False
    else:
        action, dose = decision.action, decision.dose
        reason = decision.explanation
        safety_probability = float(decision.safety_probability)
        high_uncertainty = (
            bool(decision.high_uncertainty)
            if isinstance(decision, (BMACRMDecision, DACRMDecision))
            else False
        )
    max_mcse: float | None = None
    max_rhat: float | None = None
    if isinstance(result.posterior, DACRMPosterior):
        max_mcse = _diagnostic_max(np.asarray(result.posterior.dose_summary.batch_mean_mcse))
        max_rhat = _diagnostic_max(np.asarray(result.posterior.dose_summary.split_rhat))
    return CRMTrialStep(
        result.snapshot.at,
        current_dose,
        action,
        dose,
        result.routing,
        reason,
        _freeze_int(snapshot.treated_counts.copy()),
        _freeze_int(snapshot.observed_counts.copy()),
        _freeze_int(snapshot.pending_counts.copy()),
        safety_probability,
        high_uncertainty,
        result.evaluations,
        max_mcse,
        max_rhat,
    )


def run_crm_trial(
    skeletons: ArrayLike,
    interarrival: ArrayLike,
    dlt_delays: ArrayLike,
    window: float,
    *,
    target: float,
    cohort_size: int = 3,
    starting_dose: int = 0,
    method: Literal["bmacrm", "dacrm"] = "bmacrm",
    da_prior: DACRMPrior | None = None,
    model_prior: ArrayLike | None = None,
    prior_sd: float | None = None,
    aggregation: Literal["bma", "bms", "occam"] = "bma",
    occam_threshold: float | None = None,
    safety_cutoff: float = 0.9,
    minimum_observed: int | None = None,
    rng: np.random.Generator | None = None,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_completions: int = 128,
    max_evaluations: int = _DEFAULT_PER_CALL_EVALUATIONS,
    max_total_evaluations: int = _DEFAULT_TOTAL_EVALUATIONS,
) -> CRMTrial:
    """Replay one staggered, fixed-dose-cohort CRM trial.

    ``interarrival[i]`` is the gap before patient ``i``: the first gap is
    measured from calendar time zero, and later cohort gaps are applied before
    the next cohort decision. Within a cohort, patients keep the current dose.
    A pending-outcome wait advances to the earliest ascertainment of an
    already-enrolled patient and does not add a second arrival gap afterward.
    """
    if method not in {"bmacrm", "dacrm"}:
        raise ValueError("method must be 'bmacrm' or 'dacrm'")
    arrivals = _raw_numeric(interarrival, "interarrival", _MAX_PATIENTS)
    delays = np.asarray(dlt_delays)
    if arrivals.ndim != 1 or not 1 <= arrivals.size <= _MAX_PATIENTS:
        raise ValueError("interarrival must contain 1..200 nonnegative patient gaps")
    if np.any(arrivals < 0):
        raise ValueError("interarrival gaps must be nonnegative")
    raw_skeletons = _raw_numeric(skeletons, "skeletons", 5 * _MAX_DOSES)
    if raw_skeletons.ndim == 1:
        skeleton_matrix = raw_skeletons[None, :]
    elif raw_skeletons.ndim == 2:
        skeleton_matrix = raw_skeletons
    else:
        raise ValueError("skeletons must be one- or two-dimensional")
    dose_count = skeleton_matrix.shape[1]
    if not 2 <= dose_count <= _MAX_DOSES:
        raise ValueError("CRM trial requires 2..20 doses")
    if method == "bmacrm" and not 1 <= skeleton_matrix.shape[0] <= 5:
        raise ValueError("BMA-CRM trials require 1..5 dose skeleton models")
    if method == "dacrm" and skeleton_matrix.shape[0] != 1:
        raise ValueError("DA-CRM trials require exactly one dose skeleton")
    if (
        delays.ndim != 2
        or delays.shape != (arrivals.size, dose_count)
        or delays.dtype.kind not in "iuf"
    ):
        raise ValueError("dlt_delays must have shape (patients,doses)")
    delays = np.asarray(delays, dtype=float)
    if np.any(~(np.isposinf(delays) | (np.isfinite(delays) & (delays >= 0)))):
        raise ValueError("potential DLT delays must be nonnegative or positive infinity")
    duration = scalar(window, "window")
    target_value = scalar(target, "target")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("window must be positive and finite")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    if np.any(delays[np.isfinite(delays)] > duration):
        raise ValueError("finite DLT delays must not exceed window")
    size = _count_setting(cohort_size, "cohort_size", 1, _MAX_COHORT_SIZE)
    if arrivals.size % size:
        raise ValueError("planned patients must be a multiple of cohort_size")
    start = _count_setting(starting_dose, "starting_dose", 0, dose_count - 1)
    per_call_limit = _count_setting(
        max_evaluations, "max_evaluations", 1, _MAX_PER_CALL_EVALUATIONS
    )
    total_limit = _count_setting(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    completion_limit = _count_setting(max_completions, "max_completions", 1, 1024)
    cutoff = scalar(safety_cutoff, "safety_cutoff")
    if not 0 <= cutoff <= 1:
        raise ValueError("safety_cutoff must lie in [0,1]")
    if method == "bmacrm":
        if da_prior is not None or minimum_observed is not None:
            raise ValueError("da_prior and minimum_observed apply only to method='dacrm'")
        sd_value = sqrt(2) if prior_sd is None else scalar(prior_sd, "prior_sd")
        if not 1e-3 <= sd_value <= 10:
            raise ValueError("prior_sd must lie in [1e-3,10]")
    elif aggregation != "bma" or occam_threshold is not None:
        raise ValueError("non-default aggregation options apply only to method='bmacrm'")
    else:
        if model_prior is not None or prior_sd is not None:
            raise ValueError("model_prior and prior_sd apply only to method='bmacrm'")
        if da_prior is None or not isinstance(da_prior, DACRMPrior):
            raise ValueError("DA-CRM trials require an explicit DACRMPrior")
        if da_prior.breaks[-1] != duration:
            raise ValueError("DA-CRM prior endpoint must equal window")
        if minimum_observed is None:
            raise ValueError("DA-CRM trials require explicit minimum_observed")
        _count_setting(minimum_observed, "minimum_observed", 0, _MAX_PATIENTS)
        draws = _da_integer_setting(draws, "draws", 8, 10_000)
        warmup = _da_integer_setting(warmup, "warmup", 0, 10_000)
        chains = _da_integer_setting(chains, "chains", 2, 4)
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        transition_count = chains * (draws + warmup)
        if (
            transition_count * (arrivals.size + dose_count + da_prior.shape.size)
            > _MAX_TRANSITION_WORK
        ):
            raise ValueError("worst-case DA-CRM transition work exceeds 20000000 units")
        retained = chains * draws * (1 + da_prior.shape.size + dose_count + arrivals.size)
        if retained > _MAX_RETAINED_CELLS:
            raise ValueError("worst-case retained DA-CRM draws exceed 2000000 cells")

    n_planned = arrivals.size
    assigned = np.full(n_planned, -1, dtype=np.int64)
    enrollment = np.empty(n_planned, dtype=float)
    dlt_calendar = np.full(n_planned, np.inf, dtype=float)
    known_at = np.empty(n_planned, dtype=float)
    treated_counts = np.zeros(dose_count, dtype=np.int64)
    current: int | None = None
    clock = 0.0
    suspension = 0.0
    evaluations = 0
    steps: list[CRMTrialStep] = []
    enrolled = 0
    stop_reason = "max_patients"
    decision_time = 0.0
    early_stop = False

    def assess(
        at_time: float, *, final: bool = False
    ) -> tuple[CRMCalendarSnapshot, CRMCalendarDecision]:
        nonlocal evaluations
        if evaluations >= total_limit:
            raise RuntimeError("CRM trial exhausted max_total_evaluations")
        snapshot = crm_calendar_snapshot(
            assigned[:enrolled],
            enrollment[:enrolled],
            delays[np.arange(enrolled), assigned[:enrolled]],
            window=duration,
            at=at_time,
            dose_count=dose_count,
        )
        result = crm_calendar_decision(
            snapshot,
            raw_skeletons,
            target=target_value,
            method=method,
            da_prior=da_prior,
            model_prior=model_prior,
            prior_sd=prior_sd,
            aggregation=aggregation,
            occam_threshold=occam_threshold,
            current_dose=current,
            starting_dose=start,
            safety_cutoff=cutoff,
            minimum_observed=minimum_observed,
            rng=rng,
            draws=draws,
            warmup=warmup,
            chains=chains,
            final=final,
            max_completions=completion_limit,
            max_evaluations=min(per_call_limit, total_limit - evaluations),
        )
        evaluations += result.evaluations
        steps.append(_step(result, snapshot, current))
        return snapshot, result

    while enrolled < n_planned:
        clock = _advance(clock, float(arrivals[enrolled]))
        while True:
            snapshot, result = assess(clock)
            decision = result.decision
            if decision.action == "stop":
                stop_reason = "safety_stop"
                decision_time = clock
                early_stop = enrolled < n_planned
                break
            if decision.action == "wait":
                future = known_at[:enrolled][known_at[:enrolled] > clock]
                if not future.size:
                    raise RuntimeError(
                        "CRM decision requested a wait with no pending ascertainment"
                    )
                next_time = float(np.min(future))
                if next_time <= clock:
                    raise RuntimeError("CRM pending-outcome wait did not advance the calendar")
                suspension += next_time - clock
                clock = next_time
                continue
            if decision.action not in {"start", "treat"} or decision.dose is None:
                raise RuntimeError("unexpected nonfinal CRM decision action")
            current = int(decision.dose)
            break
        if early_stop:
            break
        if current is None:
            raise RuntimeError("CRM start decision did not select a dose")
        for offset in range(size):
            patient = enrolled + offset
            if offset:
                clock = _advance(clock, float(arrivals[patient]))
            enrollment[patient] = clock
            assigned[patient] = current
            delay = float(delays[patient, current])
            window_end = _advance(clock, duration)
            event_time = _advance(clock, delay) if np.isfinite(delay) else np.inf
            dlt_calendar[patient] = event_time
            known_at[patient] = min(window_end, event_time)
            treated_counts[current] += 1
        enrolled += size

    if enrolled == 0:
        raise RuntimeError("CRM trial ended before enrolling any patients")
    final_time = max(clock, float(np.max(known_at[:enrolled])))
    dlt_by_patient = np.isfinite(dlt_calendar[:enrolled])
    toxicities = np.bincount(assigned[:enrolled][dlt_by_patient], minlength=dose_count).astype(
        np.int64
    )
    observed_counts = treated_counts.copy()
    selected_dose: int | None = None
    if not early_stop:
        decision_time = final_time
        _, final_result = assess(final_time, final=True)
        final_decision = final_result.decision
        if final_decision.action == "select_mtd":
            selected_dose = final_decision.dose
        elif final_decision.action == "stop":
            stop_reason = "safety_stop_final"
        else:
            raise RuntimeError("final CRM decision did not select an MTD or stop")

    max_mcse_values = np.asarray(
        [step.max_dose_mcse for step in steps if step.max_dose_mcse is not None]
    )
    max_rhat_values = np.asarray(
        [step.max_split_rhat for step in steps if step.max_split_rhat is not None]
    )
    max_mcse = _diagnostic_max(max_mcse_values) if max_mcse_values.size else None
    max_rhat = _diagnostic_max(max_rhat_values) if max_rhat_values.size else None
    return CRMTrial(
        _freeze_int(assigned[:enrolled].copy()),
        _freeze(enrollment[:enrolled]),
        _freeze(dlt_calendar[:enrolled]),
        _freeze_int(treated_counts),
        _freeze_int(observed_counts),
        _freeze_int(toxicities),
        selected_dose,
        stop_reason,
        decision_time,
        final_time,
        suspension,
        evaluations,
        max_mcse,
        max_rhat,
        tuple(steps),
    )
