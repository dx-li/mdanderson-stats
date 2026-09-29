"""Replay TITE-BOIN12 conduct on explicit cohort and endpoint-time tapes.

The calendar protocol is a Python policy around the TITE-BOIN12 AL and BDA
decisions. A first cohort is enrolled at the first supplied calendar time
without an interim look. Before every later cohort, a conduct look is made;
strict pending-endpoint suspensions advance the look to the earliest endpoint
ascertainment and retry. A decision lag separates a successful look from the
cohort's enrollment. Interarrival gaps are measured from the previous actual
patient arrival, so suspended cohorts do not queue. This is not a claim of native
calendar parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boin12 import BOIN12Design, BOIN12Selection
from .tite_boin12 import (
    TITEBOIN12Decision,
    _readonly,
    tite_boin12_decision,
    tite_boin12_select_obd,
)
from .tite_boin12_bda import (
    TITEBOIN12BDADecision,
    tite_boin12_bda_decision,
)
from .tite_boin12_bda import (
    _inputs_valid as _bda_inputs_valid,
)
from .tite_boin12_bda import (
    _prior as _bda_prior,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

_MAX_PATIENTS = 1000
_MAX_DOSES = 100
_MAX_LOOKS = 10_000
_MAX_CALENDAR_WORK = 100_000_000
_MAX_BDA_FIT_WORK = 20_000_000
_MAX_HISTORY_VALUES = 10_000_000


@dataclass(frozen=True)
class TITEBOIN12CalendarStep:
    """Compact conduct snapshot; BDA posterior draw arrays are never retained."""

    time: float
    current_dose: int
    action: str
    next_dose: int | None
    eliminated: NDArray[np.bool_]
    pending_counts: IntArray
    admissible: NDArray[np.bool_] | None
    toxicity_overdose_probability: FloatArray | None
    efficacy_futility_probability: FloatArray | None
    utility_probability: FloatArray | None
    expected_utility: FloatArray | None
    toxicity_rate: FloatArray | None
    toxicity_ess: FloatArray | None
    imputed_toxicity_rate: FloatArray | None


@dataclass(frozen=True)
class TITEBOIN12CalendarTrial:
    """One calendar replay, with accrual stop and final ascertainment separated."""

    enrollment_times: FloatArray
    assigned_doses: IntArray
    toxicity_delays: FloatArray
    efficacy_delays: FloatArray
    toxicity_outcomes: IntArray
    efficacy_outcomes: IntArray
    toxicity_followup: FloatArray
    efficacy_followup: FloatArray
    patients: IntArray
    observed_toxicities: IntArray
    observed_efficacies: IntArray
    selected_obd: int | None
    selection: BOIN12Selection
    eliminated: NDArray[np.bool_]
    stop_reason: str
    accrual_stop_time: float
    final_time: float
    steps: tuple[TITEBOIN12CalendarStep, ...]


def _advance(time: float, amount: float, name: str) -> float:
    result = time + amount
    if not np.isfinite(result) or (amount > 0 and result <= time):
        raise ArithmeticError(f"{name} is not representable; rescale the calendar time units")
    return float(result)


def _snapshots(
    decision: TITEBOIN12Decision | TITEBOIN12BDADecision,
) -> tuple[
    NDArray[np.bool_] | None,
    FloatArray | None,
    FloatArray | None,
    FloatArray | None,
    FloatArray | None,
    FloatArray | None,
    FloatArray | None,
    FloatArray | None,
]:
    if isinstance(decision, TITEBOIN12Decision):
        result_al = decision.posterior
        if result_al is None:
            return None, None, None, None, None, None, None, None
        return (
            result_al.admissible,
            result_al.posterior.toxicity_overdose_probability,
            result_al.posterior.efficacy_futility_probability,
            result_al.posterior.utility_probability,
            result_al.posterior.utility_mean,
            result_al.MLE[:, 0],
            result_al.ESS[:, 0],
            None,
        )
    if isinstance(decision, TITEBOIN12BDADecision):
        result_bda = decision.posterior
        if result_bda is None:
            return None, None, None, None, None, None, None, None
        return (
            result_bda.admissible,
            result_bda.posterior.toxicity_overdose_probability,
            result_bda.posterior.efficacy_futility_probability,
            result_bda.posterior.utility_probability,
            result_bda.posterior.utility_mean,
            None,
            None,
            decision.imputed_toxicity_rate,
        )
    raise TypeError("unexpected TITE-BOIN12 decision posterior")


def _step(
    time: float,
    current: int,
    decision: TITEBOIN12Decision | TITEBOIN12BDADecision,
) -> TITEBOIN12CalendarStep:
    admissible, tox, eff, utility, mean, rate, ess, imputed = _snapshots(decision)
    return TITEBOIN12CalendarStep(
        time,
        current,
        decision.action,
        decision.next_dose,
        _readonly(decision.eliminated, np.bool_),
        _readonly(decision.pending_counts, np.int64),
        None if admissible is None else _readonly(admissible, np.bool_),
        None if tox is None else _readonly(tox),
        None if eff is None else _readonly(eff),
        None if utility is None else _readonly(utility),
        None if mean is None else _readonly(mean),
        None if rate is None else _readonly(rate),
        None if ess is None else _readonly(ess),
        None if imputed is None else _readonly(imputed),
    )


def _histories(
    now: float,
    enrolled: list[float],
    tox_delays: list[float],
    eff_delays: list[float],
    tox_window: float,
    eff_window: float,
) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    n = len(enrolled)
    tox = np.full(n, -1, dtype=np.int64)
    eff = np.full(n, -1, dtype=np.int64)
    tox_follow = np.empty(n, dtype=float)
    eff_follow = np.empty(n, dtype=float)
    for j, start in enumerate(enrolled):
        tox_end = _advance(start, tox_window, "toxicity assessment deadline")
        eff_end = _advance(start, eff_window, "efficacy assessment deadline")
        td, ed = tox_delays[j], eff_delays[j]
        tox_event = np.isfinite(td) and now >= _advance(start, td, "toxicity event time")
        eff_event = np.isfinite(ed) and now >= _advance(start, ed, "efficacy event time")
        tox_complete = now >= tox_end
        eff_complete = now >= eff_end
        if tox_event:
            tox[j] = 1
            tox_follow[j] = td
        elif tox_complete:
            tox[j] = 0
            tox_follow[j] = tox_window
        else:
            tox_follow[j] = min(max(now - start, 0.0), float(np.nextafter(tox_window, 0.0)))
        if eff_event:
            eff[j] = 1
            eff_follow[j] = ed
        elif eff_complete:
            eff[j] = 0
            eff_follow[j] = eff_window
        else:
            eff_follow[j] = min(max(now - start, 0.0), float(np.nextafter(eff_window, 0.0)))
    return tox, eff, tox_follow, eff_follow


def _pending_deadlines(
    enrolled: list[float],
    tox_delays: list[float],
    eff_delays: list[float],
    tox: IntArray,
    eff: IntArray,
    tox_window: float,
    eff_window: float,
) -> FloatArray:
    deadlines: list[float] = []
    for j, start in enumerate(enrolled):
        if tox[j] < 0:
            tox_end = _advance(start, tox_window, "toxicity assessment deadline")
            if np.isfinite(tox_delays[j]):
                tox_end = min(tox_end, _advance(start, tox_delays[j], "toxicity event time"))
            deadlines.append(tox_end)
        if eff[j] < 0:
            eff_end = _advance(start, eff_window, "efficacy assessment deadline")
            if np.isfinite(eff_delays[j]):
                eff_end = min(eff_end, _advance(start, eff_delays[j], "efficacy event time"))
            deadlines.append(eff_end)
    return np.asarray(deadlines, dtype=float)


def run_tite_boin12_calendar_trial(
    design: BOIN12Design,
    interarrival_gaps: ArrayLike,
    toxicity_delay_tape: ArrayLike,
    efficacy_delay_tape: ArrayLike,
    *,
    toxicity_window: float,
    efficacy_window: float,
    cohort_size: int = 3,
    start_dose: int = 1,
    method: Literal["al", "bda"] = "al",
    decision_lag: float = 0.0,
    prior_concentrations: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    draws: int | None = None,
    warmup: int | None = None,
    chains: int | None = None,
    bda_max_work: int = _MAX_BDA_FIT_WORK,
    max_calendar_work: int = _MAX_CALENDAR_WORK,
    max_pending_toxicity: float = 0.5,
    max_pending_efficacy: float = 0.5,
    run_in_3plus3: bool = False,
) -> TITEBOIN12CalendarTrial:
    """Replay one AL- or BDA-guided TITE-BOIN12 trial.

    Each interarrival entry corresponds to one possible patient: the first is
    time from zero to the first patient, and later gaps are measured from the
    previous patient's actual enrollment. Dose decisions occur only before a
    cohort; patients within each cohort arrive at their supplied staggered
    times and retain that cohort's dose. Delay tapes have shape
    ``(possible patients, doses)``; finite entries are endpoint event times
    since enrollment within that endpoint's assessment window, while ``+inf``
    means no event during the window. Only the row and dose actually assigned
    are revealed. Event and assessment ties are visible at the look.

    The first cohort is assigned at ``start_dose`` without an interim analysis.
    Each boundary gap sets a candidate look from the previous patient's actual
    arrival. Pending gates cause calendar suspension until the earliest
    pending event or assessment completion,
    followed by a new look. A successful decision is followed by
    ``decision_lag`` before enrollment. All these calendar details are
    explicit Python policies, not native scheduling defaults. Once accrual
    stops, each endpoint is followed only to its event or assessment deadline,
    then the existing complete-data OBD selector is applied. A safety stop
    reports no selected OBD; ``selection`` retains the complete-data selector
    diagnostics separately.

    BDA requires an explicit dose-shared or dose-specific four-cell Dirichlet
    prior, ``numpy.random.Generator``, retained-draw count, warmup count, and
    chain count. Posterior draw arrays are discarded after each look; only
    compact dose-level summaries are retained.
    """
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    if method not in ("al", "bda"):
        raise ValueError("method must be 'al' or 'bda'")
    gaps_raw = np.asarray(interarrival_gaps)
    tox_raw = np.asarray(toxicity_delay_tape)
    eff_raw = np.asarray(efficacy_delay_tape)
    if gaps_raw.ndim != 1 or gaps_raw.size == 0 or gaps_raw.size > _MAX_PATIENTS:
        raise ValueError("interarrival_gaps must be a vector of 1..1000 patient gaps")
    if (
        tox_raw.ndim != 2
        or eff_raw.ndim != 2
        or tox_raw.shape != eff_raw.shape
        or tox_raw.shape[1] < 1
        or tox_raw.shape[1] > _MAX_DOSES
        or tox_raw.shape[0] < 1
        or tox_raw.shape[0] > _MAX_PATIENTS
    ):
        raise ValueError("endpoint delay tapes must share shape (1..1000 patients,1..100 doses)")
    gaps = np.asarray(gaps_raw, dtype=float)
    tox_tape = np.asarray(tox_raw, dtype=float)
    eff_tape = np.asarray(eff_raw, dtype=float)
    tw = float(toxicity_window)
    ew = float(efficacy_window)
    lag = float(decision_lag)
    if not np.isfinite(tw) or tw <= 0 or not np.isfinite(ew) or ew <= 0:
        raise ValueError("assessment windows must be finite and positive")
    if not np.isfinite(lag) or lag < 0:
        raise ValueError("decision_lag must be finite and nonnegative")
    if np.any(~np.isfinite(gaps)) or np.any(gaps < 0):
        raise ValueError("interarrival_gaps must contain finite nonnegative values")
    if not np.all(np.isfinite(tox_tape) | np.isposinf(tox_tape)) or not np.all(
        np.isfinite(eff_tape) | np.isposinf(eff_tape)
    ):
        raise ValueError("event delays must be finite or +inf")
    size = int(cohort_size)
    start = int(start_dose)
    if isinstance(cohort_size, (bool, np.bool_)) or size != cohort_size or size < 1:
        raise ValueError("cohort_size must be a positive integer")
    if (
        isinstance(start_dose, (bool, np.bool_))
        or start != start_dose
        or not 1 <= start <= tox_tape.shape[1]
    ):
        raise ValueError("start_dose must be an integer in [1,n_doses]")
    n_possible = tox_tape.shape[0]
    if n_possible == 0 or n_possible % size or gaps.size != n_possible:
        raise ValueError(
            "delay tapes must contain complete cohorts and one gap per possible patient"
        )
    if np.any(np.isfinite(tox_tape) & ((tox_tape < 0) | (tox_tape > tw))) or np.any(
        np.isfinite(eff_tape) & ((eff_tape < 0) | (eff_tape > ew))
    ):
        raise ValueError("finite event delays must lie in their endpoint assessment windows")
    if (
        isinstance(max_calendar_work, (bool, np.bool_))
        or not isinstance(max_calendar_work, (int, np.integer))
        or not 1 <= max_calendar_work <= _MAX_CALENDAR_WORK
    ):
        raise ValueError(f"max_calendar_work must lie in [1,{_MAX_CALENDAR_WORK}]")
    candidate_patients = n_possible
    cohort_count = candidate_patients // size
    look_bound = 2 * candidate_patients + cohort_count
    if look_bound > _MAX_LOOKS:
        raise ValueError(f"calendar look bound {look_bound} exceeds {_MAX_LOOKS}")
    levels = tox_tape.shape[1]
    if look_bound * levels * 12 > _MAX_HISTORY_VALUES:
        raise ValueError("compact decision history exceeds the configured calendar storage bound")
    calendar_work = 2 * look_bound * candidate_patients
    if calendar_work > max_calendar_work:
        raise ValueError(
            f"calendar work estimate {calendar_work} exceeds max_calendar_work={max_calendar_work}"
        )
    if method == "bda":
        if prior_concentrations is None or not isinstance(rng, np.random.Generator):
            raise ValueError("BDA requires prior_concentrations and a numpy.random.Generator")
        if draws is None or warmup is None or chains is None:
            raise ValueError("BDA requires explicit draws, warmup, and chains settings")
        if (
            isinstance(bda_max_work, (bool, np.bool_))
            or not isinstance(bda_max_work, (int, np.integer))
            or not 1 <= bda_max_work <= _MAX_BDA_FIT_WORK
        ):
            raise ValueError(f"bda_max_work must lie in [1,{_MAX_BDA_FIT_WORK}]")
        _bda_inputs_valid(design, rng, draws, warmup, chains, int(bda_max_work))
        _bda_prior(prior_concentrations, levels)
        fit_work = int(chains) * (int(warmup) + int(draws) + 1) * (candidate_patients + levels)
        if fit_work > bda_max_work:
            raise ValueError(
                f"per-look BDA work estimate {fit_work} exceeds bda_max_work={bda_max_work}"
            )
        aggregate_work = max(0, cohort_count - 1) * fit_work
        if calendar_work + aggregate_work > max_calendar_work:
            raise ValueError(
                "combined calendar and BDA work estimate "
                f"{calendar_work + aggregate_work} exceeds max_calendar_work={max_calendar_work}"
            )
    elif (
        prior_concentrations is not None
        or rng is not None
        or draws is not None
        or warmup is not None
        or chains is not None
    ):
        raise ValueError("BDA prior, RNG, and sampler settings are only used with method='bda'")
    max_pending_toxicity = float(max_pending_toxicity)
    max_pending_efficacy = float(max_pending_efficacy)
    if (
        not np.isfinite(max_pending_toxicity)
        or not 0 <= max_pending_toxicity <= 1
        or not np.isfinite(max_pending_efficacy)
        or not 0 <= max_pending_efficacy <= 1
    ):
        raise ValueError("pending thresholds must be finite values in [0,1]")
    if not isinstance(run_in_3plus3, (bool, np.bool_)):
        raise ValueError("run_in_3plus3 must be a boolean")
    if run_in_3plus3 and design.toxicity_limit != 0.25:
        raise ValueError("the 3+3 run-in is available only when toxicity_limit is 0.25")

    # Validate representability of the no-suspension patient calendar before
    # any sampler can consume RNG. Delays are checked after a dose is assigned.
    calendar = 0.0
    for patient_index, gap in enumerate(gaps):
        calendar = _advance(calendar, float(gap), "candidate patient time")
        if patient_index and patient_index % size == 0:
            calendar = _advance(calendar, lag, "decision-lag patient time")

    enrolled: list[float] = []
    assigned: list[int] = []
    tox_delays: list[float] = []
    eff_delays: list[float] = []
    steps: list[TITEBOIN12CalendarStep] = []
    excluded = np.zeros(levels, dtype=bool)
    current = start
    stop_reason = "maximum_cohorts"
    accrual_stop = float(gaps[0])

    for cohort_index in range(cohort_count):
        first_patient = cohort_index * size
        if cohort_index:
            look = _advance(enrolled[-1], float(gaps[first_patient]), "candidate cohort look")
            if look < enrolled[-1] or not np.isfinite(look):
                raise ArithmeticError("cohort look time is not representable")
            while True:
                tox, eff, tox_follow, eff_follow = _histories(
                    look, enrolled, tox_delays, eff_delays, tw, ew
                )
                doses = np.asarray(assigned, dtype=np.int64)
                decision: TITEBOIN12Decision | TITEBOIN12BDADecision
                if method == "al":
                    decision = tite_boin12_decision(
                        design,
                        doses,
                        tox,
                        eff,
                        tox_follow,
                        eff_follow,
                        toxicity_window=tw,
                        efficacy_window=ew,
                        n_doses=levels,
                        current_dose=current,
                        eliminated=excluded,
                        max_pending_toxicity=max_pending_toxicity,
                        max_pending_efficacy=max_pending_efficacy,
                        run_in_3plus3=run_in_3plus3,
                    )
                else:
                    assert (
                        prior_concentrations is not None
                        and rng is not None
                        and draws is not None
                        and warmup is not None
                        and chains is not None
                    )
                    decision = tite_boin12_bda_decision(
                        design,
                        doses,
                        tox,
                        eff,
                        tox_follow,
                        eff_follow,
                        toxicity_window=tw,
                        efficacy_window=ew,
                        n_doses=levels,
                        current_dose=current,
                        prior_concentrations=prior_concentrations,
                        rng=rng,
                        draws=int(draws),
                        warmup=int(warmup),
                        chains=int(chains),
                        max_work=int(bda_max_work),
                        eliminated=excluded,
                        max_pending_toxicity=max_pending_toxicity,
                        max_pending_efficacy=max_pending_efficacy,
                        run_in_3plus3=run_in_3plus3,
                    )
                steps.append(_step(look, current, decision))
                excluded = np.asarray(decision.eliminated, dtype=bool).copy()
                action = decision.action
                next_dose = decision.next_dose
                del decision
                if action not in ("suspend_pending", "suspend_no_information"):
                    break
                deadlines = _pending_deadlines(enrolled, tox_delays, eff_delays, tox, eff, tw, ew)
                future = deadlines[deadlines > look]
                if future.size == 0:
                    raise RuntimeError("suspension has no future endpoint ascertainment")
                look = float(np.min(future))
            if next_dose is None:
                stop_reason = action
                accrual_stop = look
                break
            assigned_dose = int(next_dose)
            enrollment_time = _advance(look, lag, "decision-lag cohort time")
        else:
            assigned_dose = start
            enrollment_time = float(gaps[0])

        for offset in range(size):
            patient_index = first_patient + offset
            if offset:
                enrollment_time = _advance(
                    enrolled[-1], float(gaps[patient_index]), "within-cohort patient arrival"
                )
            enrolled.append(enrollment_time)
            assigned.append(assigned_dose)
            td = float(tox_tape[patient_index, assigned_dose - 1])
            ed = float(eff_tape[patient_index, assigned_dose - 1])
            tox_delays.append(td)
            eff_delays.append(ed)
            _advance(enrollment_time, tw, "toxicity assessment deadline")
            _advance(enrollment_time, ew, "efficacy assessment deadline")
            if np.isfinite(td):
                _advance(enrollment_time, td, "toxicity event time")
            if np.isfinite(ed):
                _advance(enrollment_time, ed, "efficacy event time")
        current = assigned_dose
        accrual_stop = enrollment_time

    final_deadlines = [
        min(
            _advance(start, tw, "toxicity assessment deadline"),
            _advance(start, td, "toxicity event time") if np.isfinite(td) else np.inf,
        )
        for start, td in zip(enrolled, tox_delays, strict=True)
    ] + [
        min(
            _advance(start, ew, "efficacy assessment deadline"),
            _advance(start, ed, "efficacy event time") if np.isfinite(ed) else np.inf,
        )
        for start, ed in zip(enrolled, eff_delays, strict=True)
    ]
    final_time = max(max(final_deadlines), accrual_stop)
    tox, eff, tox_follow, eff_follow = _histories(
        final_time, enrolled, tox_delays, eff_delays, tw, ew
    )
    dose_array = np.asarray(assigned, dtype=np.int64)
    selection = tite_boin12_select_obd(
        design,
        dose_array,
        tox,
        eff,
        tox_follow,
        eff_follow,
        toxicity_window=tw,
        efficacy_window=ew,
        n_doses=levels,
        eliminated=excluded,
    )
    patients = np.bincount(dose_array, minlength=levels + 1)[1:].astype(np.int64)
    observed_tox = np.bincount(dose_array[tox == 1], minlength=levels + 1)[1:].astype(np.int64)
    observed_eff = np.bincount(dose_array[eff == 1], minlength=levels + 1)[1:].astype(np.int64)
    return TITEBOIN12CalendarTrial(
        _readonly(enrolled),
        _readonly(dose_array, np.int64),
        _readonly(tox_delays),
        _readonly(eff_delays),
        _readonly(tox, np.int64),
        _readonly(eff, np.int64),
        _readonly(tox_follow),
        _readonly(eff_follow),
        _readonly(patients, np.int64),
        _readonly(observed_tox, np.int64),
        _readonly(observed_eff, np.int64),
        None if stop_reason == "stop_safety" else selection.obd,
        selection,
        _readonly(excluded, np.bool_),
        stop_reason,
        float(accrual_stop),
        float(final_time),
        tuple(steps),
    )
