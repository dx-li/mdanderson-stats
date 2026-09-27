"""Operating-characteristic comparison for TITE-BOIN and Rolling Six."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .boin import BOINDesign, _owned
from .rolling_six import RollingSixDesign
from .rolling_six_simulation import RollingSixSimulation, simulate_rolling_six
from .tite_boin_simulation import TITEBOINSimulation, simulate_tite_boin
from .tite_keyboard import toxicity_followup_weights
from .toxicity_timing import toxicity_time_quantile

_MAX_TRIALS = 100_000
_MAX_PATIENTS = 200
_MAX_OUTPUT_CELLS = 10_000_000
_MAX_SCENARIO_DRAWS = 100_000_000


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return result


def _mass_vector(value: ArrayLike | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (3,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with three entries")
    return finite(raw, name)


def _late_vector(value: ArrayLike | None, doses: int) -> FloatArray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim > 1 or raw.size not in (1, doses):
        raise ValueError("late_probability must be a scalar or a dose-length real vector")
    return finite(raw, "late_probability")


@dataclass(frozen=True)
class OperatingCharacteristicsSummary:
    """Mean allocation/outcome/time summaries for one simulator."""

    mean_patients_by_dose: NDArray[np.float64]
    mean_toxicities_by_dose: FloatArray
    mean_total_patients: float
    mean_total_toxicities: float
    mean_duration: float
    mean_suspension_time: float
    status_probabilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TITEBOINRollingSixComparison:
    """Both original simulation outputs, summaries and their shared scenario."""

    tite_boin: TITEBOINSimulation
    rolling_six: RollingSixSimulation
    tite_boin_summary: OperatingCharacteristicsSummary
    rolling_six_summary: OperatingCharacteristicsSummary
    true_toxicity: FloatArray
    target: float
    window: float
    accrual_rate: float
    cohorts: int
    cohort_size: int
    start_dose: int
    tite_boin_max_patients: int
    rolling_six_max_patients: int
    arrival: str
    event_distribution: str
    late_probability: FloatArray | None
    event_trimester_probabilities: FloatArray | None
    trimester_probabilities: FloatArray | None
    minimum_complete_fraction: float
    minimum_pending_followup: float
    trials: int
    tite_boin_design: BOINDesign
    rolling_six_design: RollingSixDesign


def _summary(
    patients: ArrayLike,
    toxicities: ArrayLike,
    duration: FloatArray,
    suspension: FloatArray,
    statuses: tuple[str, ...],
) -> OperatingCharacteristicsSummary:
    patient_values = np.asarray(patients, dtype=float)
    toxicity_values = np.asarray(toxicities, dtype=float)
    if (
        not np.all(np.isfinite(patient_values))
        or not np.all(np.isfinite(toxicity_values))
        or not np.all(np.isfinite(duration))
        or not np.all(np.isfinite(suspension))
    ):
        raise ArithmeticError("simulation summaries contain non-finite values")

    def nonnegative_mean(values: FloatArray) -> float:
        scale = float(np.max(values, initial=0.0))
        return 0.0 if scale == 0 else float(np.mean(values / scale) * scale)

    unique = tuple(sorted(set(statuses)))
    probabilities = tuple((status, statuses.count(status) / len(statuses)) for status in unique)
    return OperatingCharacteristicsSummary(
        _owned(patient_values.mean(axis=0)),
        _owned(toxicity_values.mean(axis=0)),
        float(patient_values.sum(axis=1).mean()),
        float(toxicity_values.sum(axis=1).mean()),
        nonnegative_mean(duration),
        nonnegative_mean(suspension),
        probabilities,
    )


def compare_tite_boin_rolling_six(
    design: BOINDesign,
    true_toxicity: ArrayLike,
    window: float,
    accrual_rate: float,
    *,
    rolling_six_design: RollingSixDesign | None = None,
    cohorts: int = 10,
    cohort_size: int = 3,
    rolling_six_max_patients: int | None = None,
    trials: int = 1000,
    start_dose: int = 1,
    arrival: Literal["fixed", "exponential"] = "fixed",
    event_distribution: Literal["uniform", "weibull", "log-logistic"] = "uniform",
    late_probability: ArrayLike | None = None,
    event_trimester_probabilities: ArrayLike | None = None,
    trimester_probabilities: ArrayLike | None = None,
    minimum_complete_fraction: float = 0.51,
    minimum_pending_followup: float = 0.25,
    rng: int | np.integer | np.random.Generator | None = None,
) -> TITEBOINRollingSixComparison:
    """Compare both designs under one toxicity/timing scenario and independent draws.

    Each simulator retains its own enrollment cap and conduct rules. The same
    arrival and event-time model is applied to both, with one shared generator
    advanced serially. This Python comparison does not claim unpublished native
    sample-matching, scheduler, seed-stream, or report-layout equivalence.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    comparator = RollingSixDesign() if rolling_six_design is None else rolling_six_design
    if not isinstance(comparator, RollingSixDesign):
        raise ValueError("rolling_six_design must be a RollingSixDesign")
    raw_probability = np.asarray(true_toxicity)
    if (
        raw_probability.ndim != 1
        or not 2 <= raw_probability.size <= 100
        or raw_probability.dtype.kind not in "iuf"
    ):
        raise ValueError("true_toxicity must be a real vector for 2..100 doses")
    probability = finite(raw_probability, "true_toxicity")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("true_toxicity probabilities must be in [0,1]")
    if isinstance(start_dose, bool) or not isinstance(start_dose, (int, np.integer)):
        raise ValueError("start_dose must be a one-based integer")
    start = int(start_dose)
    if not 1 <= start <= probability.size:
        raise ValueError("start_dose must identify a planned dose")
    cohort_count = _integer(cohorts, "cohorts", 1, _MAX_PATIENTS)
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    repetitions = _integer(trials, "trials", 1, _MAX_TRIALS)
    tite_cap = cohort_count * size
    if tite_cap > _MAX_PATIENTS:
        raise ValueError("TITE-BOIN planned enrollment exceeds 200 patients")
    rolling_cap = (
        min(6 * probability.size, _MAX_PATIENTS)
        if rolling_six_max_patients is None
        else _integer(rolling_six_max_patients, "rolling_six_max_patients", 1, _MAX_PATIENTS)
    )
    if rolling_six_max_patients is not None and (
        isinstance(rolling_six_max_patients, (bool, np.bool_))
        or not isinstance(rolling_six_max_patients, (int, np.integer))
    ):
        raise ValueError("rolling_six_max_patients must be an integer")
    if arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    if event_distribution not in ("uniform", "weibull", "log-logistic"):
        raise ValueError("unsupported event_distribution")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raw_rng = np.asarray(rng)
        if raw_rng.ndim != 0 or raw_rng.dtype.kind not in "iu" or isinstance(rng, (bool, np.bool_)):
            raise ValueError("rng must be an integer seed, Generator or None")
        if int(raw_rng) < 0:
            raise ValueError("rng seed must be nonnegative")
    duration = scalar(window, "window")
    rate = scalar(accrual_rate, "accrual_rate")
    if duration <= 0 or rate <= 0 or not isfinite(1 / rate):
        raise ValueError("window and accrual_rate must be positive and representable")
    complete = scalar(minimum_complete_fraction, "minimum_complete_fraction")
    minimum_followup = scalar(minimum_pending_followup, "minimum_pending_followup")
    if not 0.25 <= complete <= 1 or not 0 <= minimum_followup <= 1:
        raise ValueError("require completion fraction in [.25,1] and follow-up fraction in [0,1]")
    # Validate all timing and analysis priors before acquiring/advancing RNG state.
    timing_prior = _mass_vector(event_trimester_probabilities, "event_trimester_probabilities")
    analysis_prior = _mass_vector(trimester_probabilities, "trimester_probabilities")
    toxicity_followup_weights([], duration, trimester_probabilities=timing_prior)
    toxicity_followup_weights([], duration, trimester_probabilities=analysis_prior)
    if event_distribution != "uniform" and timing_prior is not None:
        raise ValueError("event trimester masses require uniform event timing")
    late = _late_vector(late_probability, probability.size)
    # The quantile primitive provides canonical timing-parameter validation.
    toxicity_time_quantile(
        0.5,
        probability,
        duration,
        distribution=event_distribution,
        late_probability=late,
    )
    output_cells = repetitions * (4 * probability.size + 10) + 4 * probability.size + 4
    if output_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("combined comparison results exceed the bounded output-cell budget")
    scenario_draws = repetitions * (tite_cap + rolling_cap) * probability.size
    if scenario_draws > _MAX_SCENARIO_DRAWS:
        raise ValueError("comparison exceeds the bounded potential-outcome work limit")

    generator = np.random.default_rng(rng)
    tite = simulate_tite_boin(
        design,
        probability,
        duration,
        rate,
        cohorts=cohort_count,
        cohort_size=size,
        trials=repetitions,
        start_dose=start,
        arrival=arrival,
        event_distribution=event_distribution,
        late_probability=late,
        event_trimester_probabilities=timing_prior,
        trimester_probabilities=analysis_prior,
        minimum_complete_fraction=complete,
        minimum_pending_followup=minimum_followup,
        rng=generator,
    )
    rolling = simulate_rolling_six(
        comparator,
        probability,
        duration,
        rate,
        max_patients=rolling_cap,
        trials=repetitions,
        start_dose=start,
        arrival=arrival,
        event_distribution=event_distribution,
        late_probability=late,
        event_trimester_probabilities=timing_prior,
        rng=generator,
    )
    tite_statuses = tuple(
        "no_recommendation" if dose == 0 else "recommendation" for dose in tite.selected_dose
    )
    tite_summary = _summary(
        tite.patients,
        tite.toxicities,
        tite.duration,
        tite.suspension_time,
        tite_statuses,
    )
    rolling_summary = _summary(
        rolling.patients,
        rolling.toxicities,
        rolling.duration,
        rolling.suspension_time,
        rolling.selection_status,
    )
    return TITEBOINRollingSixComparison(
        tite,
        rolling,
        tite_summary,
        rolling_summary,
        _owned(probability),
        design.target,
        duration,
        rate,
        cohort_count,
        size,
        start,
        tite_cap,
        rolling_cap,
        arrival,
        event_distribution,
        None if late is None else _owned(late),
        None if timing_prior is None else _owned(timing_prior),
        None if analysis_prior is None else _owned(analysis_prior),
        complete,
        minimum_followup,
        repetitions,
        design,
        comparator,
    )


def tite_boin_rolling_six_report(
    comparison: TITEBOINRollingSixComparison, *, digits: int = 4
) -> str:
    """Format selection, dose allocation, enrollment and calendar summaries as Markdown."""
    if not isinstance(comparison, TITEBOINRollingSixComparison):
        raise ValueError("comparison must be a TITEBOINRollingSixComparison")
    places = _integer(digits, "digits", 0, 10)
    analysis_masses = (
        None
        if comparison.trimester_probabilities is None
        else comparison.trimester_probabilities.tolist()
    )
    event_masses = (
        None
        if comparison.event_trimester_probabilities is None
        else comparison.event_trimester_probabilities.tolist()
    )
    summary_a, summary_b = comparison.tite_boin_summary, comparison.rolling_six_summary
    lines = [
        "# TITE-BOIN vs Rolling Six operating characteristics",
        "",
        f"Trials: {comparison.trials}; target: {comparison.target!r}; "
        f"window: {comparison.window!r}; accrual rate: {comparison.accrual_rate!r} "
        "patients/time unit.",
        f"Scenario: {comparison.event_distribution} event timing, {comparison.arrival} arrivals; "
        f"TITE-BOIN cap {comparison.tite_boin_max_patients}, Rolling Six cap "
        f"{comparison.rolling_six_max_patients}.",
        f"Cohorts: {comparison.cohorts} × {comparison.cohort_size}; "
        f"start dose: {comparison.start_dose}.",
        f"TITE-BOIN gates: completion fraction {comparison.minimum_complete_fraction!r}, "
        f"minimum pending follow-up {comparison.minimum_pending_followup!r}; "
        "trimester analysis masses "
        f"{analysis_masses}.",
        f"Event timing settings: late probabilities "
        f"{None if comparison.late_probability is None else comparison.late_probability.tolist()}, "
        "event trimester masses "
        f"{event_masses}.",
        f"BOIN rules: extra_safe={comparison.tite_boin_design.extra_safe}, "
        f"stay_at_one_of_three={comparison.tite_boin_design.stay_at_one_of_three}, "
        f"deescalate_at_two_of_six={comparison.tite_boin_design.deescalate_at_two_of_six}, "
        f"bound_mtd={comparison.tite_boin_design.bound_mtd}; Rolling Six "
        f"require_complete_before_escalation={comparison.rolling_six_design.require_complete_before_escalation}.",
        "",
        "The designs use independent outcomes and retain distinct conduct rules; no sample-size "
        "matching is applied. TITE-BOIN and a Rolling Six highest-dose recommendation are not "
        "established MTD claims.",
        "",
        "## Selection probabilities",
        "",
        "| Recommendation | TITE-BOIN probability | MCSE | Rolling Six probability | MCSE |",
        "|:--|--:|--:|--:|--:|",
    ]
    tite_p, tite_se = (
        comparison.tite_boin.selection_probability,
        comparison.tite_boin.selection_mcse,
    )
    roll_p, roll_se = (
        comparison.rolling_six.selection_probability,
        comparison.rolling_six.selection_mcse,
    )
    for code in range(len(comparison.true_toxicity) + 1):
        label = "No recommendation" if code == 0 else f"Dose {code}"
        lines.append(
            f"| {label} | {tite_p[code]:.{places}f} | {tite_se[code]:.{places}f} | "
            f"{roll_p[code]:.{places}f} | {roll_se[code]:.{places}f} |"
        )
    lines.extend(
        [
            "",
            "Rolling Six status distinguishes an identified MTD from a recommendation at the "
            "highest planned dose:",
            "",
            "| Rolling Six terminal status | Probability |",
            "|:--|--:|",
        ]
    )
    for status, probability in comparison.rolling_six_summary.status_probabilities:
        lines.append(f"| {status} | {probability:.{places}f} |")
    lines.extend(
        [
            "",
            "## Mean allocation and outcomes",
            "",
            "| Dose | True toxicity | TITE-BOIN patients | TITE-BOIN DLTs | "
            "Rolling Six patients | Rolling Six DLTs |",
            "|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for dose, truth in enumerate(comparison.true_toxicity, start=1):
        i = dose - 1
        lines.append(
            f"| {dose} | {truth:.{places}f} | "
            f"{comparison.tite_boin_summary.mean_patients_by_dose[i]:.{places}f} | "
            f"{comparison.tite_boin_summary.mean_toxicities_by_dose[i]:.{places}f} | "
            f"{comparison.rolling_six_summary.mean_patients_by_dose[i]:.{places}f} | "
            f"{comparison.rolling_six_summary.mean_toxicities_by_dose[i]:.{places}f} |"
        )
    lines.extend(
        [
            "",
            "## Enrollment and follow-up",
            "",
            "| Measure | TITE-BOIN | Rolling Six |",
            "|:--|--:|--:|",
            f"| Mean total patients | {summary_a.mean_total_patients:.{places}f} | "
            f"{summary_b.mean_total_patients:.{places}f} |",
            f"| Mean total DLTs | {summary_a.mean_total_toxicities:.{places}f} | "
            f"{summary_b.mean_total_toxicities:.{places}f} |",
            f"| Mean duration | {summary_a.mean_duration:.{places}f} | "
            f"{summary_b.mean_duration:.{places}f} |",
            f"| Mean suspension time | {summary_a.mean_suspension_time:.{places}f} | "
            f"{summary_b.mean_suspension_time:.{places}f} |",
        ]
    )
    return "\n".join(lines)
