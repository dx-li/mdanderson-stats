"""Explicit-outcome calendar replay for the BARD BF-BLRM stage-one design.

This replay uses the fitted BF-BLRM model and decision rules. Outcomes and
assessment delays are supplied by the caller; the module does not invent an
outcome or timing distribution. Waiting until every DLT assessment in a full
escalation cohort is known before moving to the next dose is an explicit
Python scheduling convention because the paper does not fully specify that
calendar detail.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite, scalar
from .bard_blrm import BARDLogisticPrior, fit_bard_blrm
from .bard_blrm_decision import (
    BARDBLRMDecision,
    BARDBLRMSelection,
    bard_blrm_backfill,
    bard_blrm_next_dose,
    bard_blrm_select_mtd,
)

_MAX_ARRIVALS = 2_000
_MAX_DOSES = 100
_MAX_INPUT_CELLS = 200_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_TOTAL_EVALUATIONS = 2_000_000
_MAX_TOTAL_WORK = 50_000_000


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _real_matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> NDArray:
    if np.shape(value) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    return finite(value, name)


def _integer_settings(**values: int) -> dict[str, int]:
    names = tuple(values)
    raw = [values[name] for name in names]
    if any(np.ndim(value) != 0 or np.iscomplexobj(value) for value in raw):
        raise ValueError(f"{', '.join(names)} must be integer scalars")
    parsed = finite(raw, "trial settings")
    if parsed.shape != (len(names),) or np.any(parsed != np.floor(parsed)):
        raise ValueError(f"{', '.join(names)} must be integer scalars")
    return {name: int(value) for name, value in zip(names, parsed, strict=True)}


@dataclass(frozen=True)
class BARDBLRMPatient:
    """One assigned patient; dose indices are one-based."""

    arrival_index: int
    arrival_time: float
    dose: int
    role: str
    dlt: bool
    response: bool
    dlt_assessment_time: float
    response_assessment_time: float


@dataclass(frozen=True)
class BARDBLRMSnapshot:
    """Compact posterior state retained at a fit or decision."""

    time: float
    evaluated: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    assigned: NDArray[np.int64]
    responses_observed: NDArray[np.int64]
    responses: NDArray[np.int64]
    ptt: NDArray[np.float64]
    pod: NDArray[np.float64]
    coefficient_mean: NDArray[np.float64]
    ptt_mcse: NDArray[np.float64]
    pod_mcse: NDArray[np.float64]
    ptt_split_rhat: NDArray[np.float64]
    pod_split_rhat: NDArray[np.float64]
    likelihood_evaluations: int
    work_units: int


@dataclass(frozen=True)
class BARDBLRMStep:
    """A replay event, assignment, decline, or terminal state."""

    time: float
    kind: str
    arrival_index: int | None
    dose: int | None
    role: str | None
    reason: str
    snapshot: BARDBLRMSnapshot
    decision: BARDBLRMDecision | None = None


@dataclass(frozen=True)
class BARDBLRMTrial:
    """Immutable BF-BLRM calendar replay and final posterior decision."""

    patients: tuple[BARDBLRMPatient, ...]
    steps: tuple[BARDBLRMStep, ...]
    arrival_times: NDArray[np.float64]
    final_evaluated: NDArray[np.int64]
    final_toxicities: NDArray[np.int64]
    final_assigned: NDArray[np.int64]
    final_responses_observed: NDArray[np.int64]
    final_responses: NDArray[np.int64]
    final_ptt: NDArray[np.float64]
    final_pod: NDArray[np.float64]
    final_selection: BARDBLRMSelection
    selected_mtd: int | None
    stop_reason: str
    boundary_policy: str
    boundary_event: str | None
    accepted_arrival_indices: tuple[int, ...]
    declined_arrival_indices: tuple[int, ...]
    escalation_patients: int
    backfill_patients: int
    response_observed_patients: int
    dlt_evaluable_patients: int
    start_time: float
    enrollment_stop_time: float
    final_time: float
    duration: float
    fit_count: int
    likelihood_evaluations: int
    work_units: int
    all_overdose_detected: bool
    all_overdose_time: float | None


@dataclass
class _Patient:
    arrival_index: int
    arrival: float
    dose: int
    role: str
    dlt: bool
    response: bool
    dlt_time: float
    response_time: float
    dlt_seen: bool = False
    response_seen: bool = False


def run_bard_blrm_trial(
    doses: ArrayLike,
    reference_dose: float,
    prior: BARDLogisticPrior,
    *,
    target_interval: ArrayLike,
    eta: float,
    arrival_times: ArrayLike,
    potential_toxicities: ArrayLike,
    potential_responses: ArrayLike,
    dlt_assessment_delays: ArrayLike,
    response_assessment_delays: ArrayLike,
    dlt_window: float,
    cohort_size: int,
    max_escalation_patients: int,
    backfill_evaluable_cap: int,
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
    boundary_policy: str,
    max_total_evaluations: int = _MAX_TOTAL_EVALUATIONS,
    max_total_work: int = _MAX_TOTAL_WORK,
) -> BARDBLRMTrial:
    """Replay BF-BLRM from explicit potential outcomes and assessment times.

    Outcome and delay matrices have shape ``(n_arrivals, n_doses)``; each
    assigned patient uses the column for their actual dose. DLT-positive
    assessments must occur within ``dlt_window``; DLT-negative assessments
    cannot occur before that window. Responses, including negative responses,
    become known at their supplied assessment times. Events tied in calendar
    time are processed together before the next allocation.

    Escalation patients are assigned in complete cohorts. Once a cohort is
    full, the next dose decision waits for all its DLT assessments; lower-dose
    backfill may occur during that wait. The final escalation cohort may be
    partial when the patient cap is reached. A one-step move to an unsafe
    dose, or a no-safe-dose equality case, requires the caller to choose
    ``boundary_policy='raise'`` or ``'stop'``. A boundary stop is permanent.
    This replay does not model native timing distributions or claim RNG parity.
    """
    if not isinstance(prior, BARDLogisticPrior):
        raise TypeError("prior must be a BARDLogisticPrior")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    if boundary_policy not in ("raise", "stop"):
        raise ValueError("boundary_policy must be 'raise' or 'stop'")

    # Validate shapes before converting/copying any potentially large input.
    dose_shape = np.shape(doses)
    if len(dose_shape) != 1 or not 1 <= dose_shape[0] <= _MAX_DOSES:
        raise ValueError(f"doses must contain 1..{_MAX_DOSES} values")
    if np.iscomplexobj(doses) or np.iscomplexobj(arrival_times):
        raise ValueError("doses and arrival_times must be real-valued")
    arrival_shape = np.shape(arrival_times)
    if len(arrival_shape) != 1 or not 1 <= arrival_shape[0] <= _MAX_ARRIVALS:
        raise ValueError(f"arrival_times must contain 1..{_MAX_ARRIVALS} values")
    n_arrivals, n_doses = arrival_shape[0], dose_shape[0]
    input_cells = n_arrivals * n_doses
    if input_cells > _MAX_INPUT_CELLS:
        raise ValueError("potential outcome and delay inputs exceed 200000 cells")
    matrix_shape = (n_arrivals, n_doses)
    raw_toxicities_shape = np.shape(potential_toxicities)
    raw_responses_shape = np.shape(potential_responses)
    if raw_toxicities_shape != matrix_shape or raw_responses_shape != matrix_shape:
        raise ValueError(f"potential outcomes must have shape {matrix_shape}")
    for value, name in (
        (potential_toxicities, "potential_toxicities"),
        (potential_responses, "potential_responses"),
    ):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be boolean or real binary values")
    raw_dlt_delay_shape = np.shape(dlt_assessment_delays)
    raw_response_delay_shape = np.shape(response_assessment_delays)
    if raw_dlt_delay_shape != matrix_shape or raw_response_delay_shape != matrix_shape:
        raise ValueError(f"assessment delay matrices must have shape {matrix_shape}")
    arrival = finite(arrival_times, "arrival_times")
    if np.any(arrival[1:] <= arrival[:-1]):
        raise ValueError("arrival_times must be strictly increasing")
    dose_grid = finite(doses, "doses")
    if np.any(dose_grid <= 0) or np.any(dose_grid[1:] <= dose_grid[:-1]):
        raise ValueError("doses must be strictly increasing and positive")
    toxic_raw = _real_matrix(potential_toxicities, matrix_shape, "potential_toxicities")
    response_raw = _real_matrix(potential_responses, matrix_shape, "potential_responses")
    if np.any((toxic_raw != 0) & (toxic_raw != 1)) or np.any(
        (response_raw != 0) & (response_raw != 1)
    ):
        raise ValueError("potential_toxicities and potential_responses must be binary")
    dlt_delay = _real_matrix(dlt_assessment_delays, matrix_shape, "dlt_assessment_delays")
    response_delay = _real_matrix(
        response_assessment_delays, matrix_shape, "response_assessment_delays"
    )
    if np.iscomplexobj(dlt_window):
        raise ValueError("dlt_window must be real-valued")
    window = scalar(dlt_window, "dlt_window")
    if window <= 0:
        raise ValueError("dlt_window must be finite and positive")
    if np.any(dlt_delay < 0) or np.any(response_delay < 0):
        raise ValueError("assessment delays must be nonnegative")
    positive_dlt = toxic_raw == 1
    if np.any(dlt_delay[positive_dlt] > window) or np.any(dlt_delay[~positive_dlt] < window):
        raise ValueError(
            "DLT-positive assessments must be within the DLT window and "
            "DLT-negative assessments cannot precede it"
        )
    dlt_absolute = arrival[:, None] + dlt_delay
    response_absolute = arrival[:, None] + response_delay
    if not np.all(np.isfinite(dlt_absolute)) or not np.all(np.isfinite(response_absolute)):
        raise ValueError("arrival plus assessment delay exceeds finite calendar range")
    if np.any((dlt_delay > 0) & (dlt_absolute == arrival[:, None])) or np.any(
        (response_delay > 0) & (response_absolute == arrival[:, None])
    ):
        raise ValueError("a positive assessment delay is below calendar-time resolution")

    settings = _integer_settings(
        cohort_size=cohort_size,
        max_escalation_patients=max_escalation_patients,
        backfill_evaluable_cap=backfill_evaluable_cap,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_total_evaluations=max_total_evaluations,
        max_total_work=max_total_work,
    )
    cohort = settings["cohort_size"]
    cap = settings["max_escalation_patients"]
    backfill_cap = settings["backfill_evaluable_cap"]
    draw_count, warmup_count, chain_count = (
        settings["draws"],
        settings["warmup"],
        settings["chains"],
    )
    eval_limit, work_limit = settings["max_total_evaluations"], settings["max_total_work"]
    if cohort < 1 or cap < 1 or backfill_cap < 1:
        raise ValueError("cohort_size, escalation cap, and backfill cap must be positive")
    if cap > 1_000 or backfill_cap > 1_000_000:
        raise ValueError("escalation and backfill caps exceed supported bounds")
    if (
        not 8 <= draw_count <= 100_000
        or not 0 <= warmup_count <= 100_000
        or not 2 <= chain_count <= 16
    ):
        raise ValueError("draws and warmup must be in 8..100000 and 0..100000; chains in 2..16")
    if not 1 <= eval_limit <= _MAX_TOTAL_EVALUATIONS:
        raise ValueError("max_total_evaluations must be in 1..2000000")
    if not 1 <= work_limit <= _MAX_TOTAL_WORK:
        raise ValueError("max_total_work must be in 1..50000000")
    if chain_count * draw_count * (2 + 3 * n_doses) > 2_000_000:
        raise ValueError("each retained BF-BLRM fit exceeds two million cells")
    retained_cells = (4 * n_arrivals + 2) * (14 * n_doses + 2)
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("replay snapshots exceed two million retained cells")

    if np.iscomplexobj(reference_dose) or np.iscomplexobj(target_interval) or np.iscomplexobj(eta):
        raise ValueError("reference dose, target interval and eta must be real-valued")
    if np.shape(target_interval) != (2,):
        raise ValueError("target_interval must have shape (2,)")
    # Validate remaining model inputs before any sampler randomness is used.
    reference = scalar(reference_dose, "reference_dose")
    if reference <= 0:
        raise ValueError("reference_dose must be finite and positive")
    target = finite(target_interval, "target_interval")
    if target.shape != (2,) or not 0 <= target[0] < target[1] <= 1:
        raise ValueError("target_interval must satisfy 0 <= gamma1 < gamma2 <= 1")
    cutoff = scalar(eta, "eta")
    if not 0 <= cutoff <= 1:
        raise ValueError("eta must lie in [0,1]")

    # The prior-only fit retains posterior draws but needs no likelihood calls.
    initial_min_work = chain_count * draw_count * n_doses
    if initial_min_work > work_limit:
        raise ValueError("total work budget cannot cover the minimum initial BF-BLRM fit")

    patients: list[_Patient] = []
    records: list[BARDBLRMPatient] = []
    steps: list[BARDBLRMStep] = []
    event_queue: list[tuple[float, int, str, int]] = []
    serial = 0
    evaluated = np.zeros(n_doses, dtype=np.int64)
    toxicities = np.zeros(n_doses, dtype=np.int64)
    assigned = np.zeros(n_doses, dtype=np.int64)
    response_observed = np.zeros(n_doses, dtype=np.int64)
    responses = np.zeros(n_doses, dtype=np.int64)
    escalation_count = 0
    backfill_count = 0
    response_patient_count = 0
    dlt_patient_count = 0
    fit_count = total_evals = total_work = 0
    ptt = pod = np.empty(0)
    coefficient_mean = ptt_mcse = pod_mcse = ptt_rhat = pod_rhat = np.empty(0)
    current_dose = 1
    cohort_records: list[int] = []
    terminal_reason: str | None = None
    terminal_boundary: str | None = None
    all_overdose_detected = False
    all_overdose_time: float | None = None
    enrollment_stop_time = float(arrival[0])
    accepted: list[int] = []
    declined: list[int] = []

    def fit_current() -> None:
        nonlocal ptt, pod, coefficient_mean, ptt_mcse, pod_mcse, ptt_rhat, pod_rhat
        nonlocal fit_count, total_evals, total_work
        remaining_evals = eval_limit - total_evals
        remaining_work = work_limit - total_work
        if remaining_evals < 1 or remaining_work < 1:
            raise RuntimeError("cumulative BF-BLRM replay budget exhausted before refit")
        fitted = fit_bard_blrm(
            dose_grid,
            evaluated,
            toxicities,
            reference,
            prior,
            target_interval=target,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            rng=rng,
            max_evaluations=min(2_000_000, remaining_evals),
            max_work=min(50_000_000, remaining_work),
        )
        fit_count += 1
        total_evals += fitted.likelihood_evaluations
        total_work += fitted.work_units
        ptt = np.array(fitted.posterior_target_probability, copy=True)
        pod = np.array(fitted.posterior_overdose_probability, copy=True)
        coefficient_mean = np.array(fitted.coefficient_summary.mean, copy=True)
        ptt_mcse = np.array(fitted.target_probability_summary.batch_mean_mcse, copy=True)
        pod_mcse = np.array(fitted.overdose_probability_summary.batch_mean_mcse, copy=True)
        ptt_rhat = np.array(fitted.target_probability_summary.split_rhat, copy=True)
        pod_rhat = np.array(fitted.overdose_probability_summary.split_rhat, copy=True)

    def snapshot(time: float) -> BARDBLRMSnapshot:
        if fit_count == 0:
            raise RuntimeError("internal replay state has no posterior fit")
        return BARDBLRMSnapshot(
            float(time),
            _readonly(evaluated, np.int64),
            _readonly(toxicities, np.int64),
            _readonly(assigned, np.int64),
            _readonly(response_observed, np.int64),
            _readonly(responses, np.int64),
            _readonly(ptt),
            _readonly(pod),
            _readonly(coefficient_mean),
            _readonly(ptt_mcse),
            _readonly(pod_mcse),
            _readonly(ptt_rhat),
            _readonly(pod_rhat),
            total_evals,
            total_work,
        )

    def boundary(event: str, time: float, *, decision: BARDBLRMDecision | None = None) -> None:
        nonlocal terminal_reason, terminal_boundary, enrollment_stop_time
        terminal_boundary = event
        enrollment_stop_time = float(time)
        if boundary_policy == "raise":
            raise RuntimeError(f"BF-BLRM replay reached boundary policy event: {event}")
        terminal_reason = "boundary_policy_stop"
        steps.append(
            BARDBLRMStep(float(time), "terminal", None, None, None, event, snapshot(time), decision)
        )

    def decision_after_cohort(time: float) -> None:
        nonlocal current_dose, cohort_records, terminal_reason, enrollment_stop_time
        if len(cohort_records) != cohort or not all(patients[i].dlt_seen for i in cohort_records):
            return
        decision = bard_blrm_next_dose(ptt, pod, current_dose, eta=cutoff)
        snap = snapshot(time)
        if decision.action == "stop_all_overdose":
            terminal_reason = "stop_all_overdose"
            enrollment_stop_time = float(time)
            steps.append(
                BARDBLRMStep(time, "terminal", None, None, None, decision.action, snap, decision)
            )
        elif decision.action == "no_eligible_safe_dose":
            boundary("no_eligible_safe_dose", time, decision=decision)
        elif decision.next_dose_safe is False:
            boundary("unsafe_one_step_dose", time, decision=decision)
        else:
            if decision.next_dose is None:
                raise RuntimeError("internal BF-BLRM decision omitted its next dose")
            old = current_dose
            current_dose = int(decision.next_dose)
            cohort_records = []
            steps.append(
                BARDBLRMStep(
                    time,
                    "cohort_decision",
                    None,
                    current_dose,
                    None,
                    f"{decision.action}:{old}->{current_dose}",
                    snap,
                    decision,
                )
            )

    def process_until(until: float, *, conduct: bool = True) -> None:
        nonlocal response_patient_count, dlt_patient_count, terminal_reason
        nonlocal all_overdose_detected, all_overdose_time
        while event_queue and event_queue[0][0] <= until:
            event_time = event_queue[0][0]
            group: list[tuple[float, int, str, int]] = []
            while event_queue and event_queue[0][0] == event_time:
                group.append(heapq.heappop(event_queue))
            changed = False
            for _, _, kind, index in group:
                patient = patients[index]
                j = patient.dose - 1
                if kind == "response":
                    if not patient.response_seen:
                        patient.response_seen = True
                        response_observed[j] += 1
                        response_patient_count += 1
                        responses[j] += int(patient.response)
                elif not patient.dlt_seen:
                    patient.dlt_seen = True
                    evaluated[j] += 1
                    toxicities[j] += int(patient.dlt)
                    dlt_patient_count += 1
                    changed = True
            if changed:
                fit_current()
                snap = snapshot(event_time)
                steps.append(
                    BARDBLRMStep(event_time, "assessment", None, None, None, "dlt_assessment", snap)
                )
                newly_all_overdose = bool(np.all(pod > cutoff) and not all_overdose_detected)
                if np.all(pod > cutoff):
                    all_overdose_detected = True
                    if all_overdose_time is None:
                        all_overdose_time = float(event_time)
                    if not conduct and newly_all_overdose:
                        steps.append(
                            BARDBLRMStep(
                                event_time,
                                "safety_observation",
                                None,
                                None,
                                None,
                                "all_overdose_detected_after_enrollment",
                                snap,
                            )
                        )
                if conduct and terminal_reason is None and np.all(pod > cutoff):
                    terminal_reason = "stop_all_overdose"
                    nonlocal_enrollment_stop(event_time)
                    steps.append(
                        BARDBLRMStep(
                            event_time,
                            "terminal",
                            None,
                            None,
                            None,
                            "stop_all_overdose",
                            snapshot(event_time),
                        )
                    )
                    return
                if conduct and terminal_reason is None:
                    decision_after_cohort(event_time)
                    if terminal_reason is not None:
                        return
            elif group:
                steps.append(
                    BARDBLRMStep(
                        event_time,
                        "assessment",
                        None,
                        None,
                        None,
                        "response_assessment",
                        snapshot(event_time),
                    )
                )

    # Local helper avoids rebinding enrollment_stop_time in the nested closure.
    def nonlocal_enrollment_stop(time: float) -> None:
        nonlocal enrollment_stop_time
        enrollment_stop_time = float(time)

    fit_current()
    initial_snapshot = snapshot(arrival[0])
    steps.append(
        BARDBLRMStep(float(arrival[0]), "initial_fit", None, 1, None, "prior", initial_snapshot)
    )
    if np.all(pod > cutoff):
        terminal_reason = "stop_all_overdose"
        all_overdose_detected = True
        all_overdose_time = float(arrival[0])
        steps.append(
            BARDBLRMStep(
                float(arrival[0]),
                "terminal",
                None,
                None,
                None,
                "stop_all_overdose",
                snapshot(arrival[0]),
            )
        )
    elif pod[0] >= cutoff:
        boundary("unsafe_initial_dose", float(arrival[0]))

    for arrival_index, when in enumerate(arrival):
        if terminal_reason is not None:
            break
        process_until(float(when))
        if terminal_reason is not None:
            break
        if escalation_count >= cap:
            terminal_reason = "escalation_patient_cap"
            enrollment_stop_time = float(when)
            steps.append(
                BARDBLRMStep(
                    float(when),
                    "terminal",
                    arrival_index,
                    None,
                    None,
                    "escalation_patient_cap",
                    snapshot(when),
                )
            )
            break

        role: str | None = None
        selected_dose: int | None = None
        if len(cohort_records) < cohort:
            role, selected_dose = "escalation", current_dose
        else:
            fill = bard_blrm_backfill(
                pod,
                responses,
                evaluated,
                current_dose,
                eta=cutoff,
                cap=backfill_cap,
                assigned=assigned,
            )
            if fill.selected_dose is not None:
                role, selected_dose = "backfill", fill.selected_dose

        if selected_dose is None:
            declined.append(arrival_index)
            steps.append(
                BARDBLRMStep(
                    float(when),
                    "declined",
                    arrival_index,
                    None,
                    None,
                    "no_backfill_eligible",
                    snapshot(when),
                )
            )
            continue
        if role is None:
            raise RuntimeError("internal BF-BLRM assignment omitted its patient role")

        if role == "escalation" and pod[selected_dose - 1] >= cutoff:
            boundary("unsafe_current_dose", float(when))
            break

        j = selected_dose - 1
        dlt_at = float(when + dlt_delay[arrival_index, j])
        response_at = float(when + response_delay[arrival_index, j])
        if not np.isfinite(dlt_at) or not np.isfinite(response_at):
            raise ValueError("arrival plus assigned assessment delay exceeds finite calendar range")
        patient = _Patient(
            arrival_index,
            float(when),
            selected_dose,
            role,
            bool(toxic_raw[arrival_index, j]),
            bool(response_raw[arrival_index, j]),
            dlt_at,
            response_at,
        )
        patients.append(patient)
        patient_index = len(patients) - 1
        records.append(
            BARDBLRMPatient(
                arrival_index,
                float(when),
                selected_dose,
                role,
                patient.dlt,
                patient.response,
                dlt_at,
                response_at,
            )
        )
        assigned[j] += 1
        accepted.append(arrival_index)
        if role == "escalation":
            escalation_count += 1
            cohort_records.append(patient_index)
        else:
            backfill_count += 1
        serial += 1
        heapq.heappush(event_queue, (dlt_at, serial, "dlt", patient_index))
        serial += 1
        heapq.heappush(event_queue, (response_at, serial, "response", patient_index))
        steps.append(
            BARDBLRMStep(
                float(when),
                "assignment",
                arrival_index,
                selected_dose,
                role,
                "assigned",
                snapshot(when),
            )
        )
        if escalation_count >= cap:
            terminal_reason = "escalation_patient_cap"
            enrollment_stop_time = float(when)
            break

    if terminal_reason is None:
        terminal_reason = "arrival_schedule_exhausted"
        enrollment_stop_time = float(arrival[-1])
        steps.append(
            BARDBLRMStep(
                float(enrollment_stop_time),
                "terminal",
                None,
                None,
                None,
                terminal_reason,
                snapshot(enrollment_stop_time),
            )
        )

    # Complete follow-up for all assigned patients after enrollment stops.
    process_until(float("inf"), conduct=False)
    final_time = max(
        enrollment_stop_time,
        max((p.dlt_time for p in patients), default=enrollment_stop_time),
        max((p.response_time for p in patients), default=enrollment_stop_time),
    )
    if fit_count == 0:
        raise RuntimeError("replay ended without a posterior fit")
    # All assigned DLT outcomes are now known; the last event refit already
    # used the final grouped counts. Response-only changes do not alter it.
    selection = bard_blrm_select_mtd(
        ptt,
        pod,
        assigned,
        eta=cutoff,
        minimum_treated=6,
    )
    selected_mtd = (
        None
        if terminal_reason in ("stop_all_overdose", "boundary_policy_stop") or all_overdose_detected
        else selection.selected_dose
    )
    relative_completion = max(
        [float(enrollment_stop_time - arrival[0]), 0.0]
        + [
            float(arrival[p.arrival_index] - arrival[0])
            + max(
                float(dlt_delay[p.arrival_index, p.dose - 1]),
                float(response_delay[p.arrival_index, p.dose - 1]),
            )
            for p in patients
        ]
    )
    duration = float(relative_completion)
    return BARDBLRMTrial(
        tuple(records),
        tuple(steps),
        _readonly(arrival),
        _readonly(evaluated, np.int64),
        _readonly(toxicities, np.int64),
        _readonly(assigned, np.int64),
        _readonly(response_observed, np.int64),
        _readonly(responses, np.int64),
        _readonly(ptt),
        _readonly(pod),
        selection,
        selected_mtd,
        terminal_reason,
        boundary_policy,
        terminal_boundary,
        tuple(accepted),
        tuple(declined),
        escalation_count,
        backfill_count,
        response_patient_count,
        dlt_patient_count,
        float(arrival[0]),
        float(enrollment_stop_time),
        float(final_time),
        duration,
        fit_count,
        total_evals,
        total_work,
        all_overdose_detected,
        all_overdose_time,
    )
