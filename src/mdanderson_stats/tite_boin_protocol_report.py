"""Reproducible TITE-BOIN operating-characteristic protocol reports."""

from __future__ import annotations

import html
import math
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .boin import BOINDesign, _owned
from .tite_boin_simulation import TITEBOINSimulation, simulate_tite_boin
from .tite_keyboard import toxicity_followup_weights
from .toxicity_timing import toxicity_time_quantile

_MAX_SCENARIOS = 20
_MAX_TRIAL_DOSE_CELLS = 1_000_000
_MAX_PATIENT_REPLICATES = 2_000_000
_MAX_REPORT_CHARS = 2_000_000
_MAX_LABEL_CHARS = 256


@dataclass(frozen=True, slots=True)
class TITEBOINScenario:
    """One named vector of true dose-level DLT probabilities."""

    name: str
    true_toxicity: ArrayLike


@dataclass(frozen=True, slots=True)
class TITEBOINProtocolRequest:
    """Design, plan, scenarios and reproducibility settings for one report."""

    trial_name: str
    design: BOINDesign
    scenarios: Sequence[TITEBOINScenario]
    window: float
    accrual_rate: float
    cohorts: int = 10
    cohort_size: int = 3
    start_dose: int = 1
    trials: int = 1000
    arrival: Literal["fixed", "exponential"] = "fixed"
    event_distribution: Literal["uniform", "weibull", "log-logistic"] = "uniform"
    late_probability: ArrayLike | None = None
    event_trimester_probabilities: ArrayLike | None = None
    trimester_probabilities: ArrayLike | None = None
    minimum_complete_fraction: float = 0.51
    minimum_pending_followup: float = 0.25
    time_unit: str = "days"
    seed: int = 1


@dataclass(frozen=True, slots=True)
class TITEBOINScenarioSummary:
    """Compact immutable summaries from a newly computed scenario simulation."""

    name: str
    seed: int
    true_toxicity: FloatArray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients_by_dose: FloatArray
    mean_toxicities_by_dose: FloatArray
    mean_duration: float
    mean_suspension_time: float
    stop_reason_probability: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class TITEBOINProtocolReport:
    """Effective protocol settings and scenario summaries, with HTML rendering."""

    trial_name: str
    target: float
    safe_probability: float
    toxic_probability: float
    escalation_boundary: float
    deescalation_boundary: float
    elimination_probability: float
    safety_offset: float
    extra_safe: bool
    stay_at_one_of_three: bool
    deescalate_at_two_of_six: bool
    bound_mtd: bool
    early_stop_patients: int | None
    window: float
    accrual_rate: float
    time_unit: str
    cohorts: int
    cohort_size: int
    start_dose: int
    maximum_patients: int
    arrival: str
    event_distribution: str
    late_probability: FloatArray | None
    late_probability_defaulted: bool
    event_trimester_probabilities: FloatArray | None
    event_trimester_probabilities_defaulted: bool
    trimester_probabilities: FloatArray | None
    trimester_probabilities_defaulted: bool
    minimum_complete_fraction: float
    minimum_pending_followup: float
    trials: int
    seed: int
    rng_convention: str
    scenarios: tuple[TITEBOINScenarioSummary, ...]

    def to_html(self, *, digits: int = 6) -> str:
        """Render a bounded, escaped UTF-8-ready protocol and OC summary."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")

        def number(value: float) -> str:
            return format(float(value), f".{digits}g")

        def vector(values: FloatArray) -> str:
            return ", ".join(number(float(value)) for value in values)

        def optional_vector(values: FloatArray | None) -> str:
            return "not captured" if values is None else vector(values)

        def detail(label: str, value: str) -> str:
            return f"<dt>{html.escape(label)}</dt><dd>{value}</dd>"

        def table_row(label: str, value: str, error: str = "—") -> str:
            return f"<tr><td>{html.escape(label)}</td><td>{value}</td><td>{error}</td></tr>"

        if self.event_trimester_probabilities_defaulted:
            if self.event_distribution == "uniform":
                event_mass_text = (
                    "uniform over the full window; effective masses by third "
                    f"{optional_vector(self.event_trimester_probabilities)} "
                    "(default)"
                )
            else:
                event_mass_text = "not used with parametric event-time generation"
        elif self.event_distribution == "uniform":
            event_mass_text = (
                "piecewise-uniform within thirds, masses "
                f"{optional_vector(self.event_trimester_probabilities)}"
            )
        else:
            event_mass_text = "not used with parametric event-time generation"
        analysis_mass_text = (
            "uniform conditional time-to-DLT weights; effective masses by third "
            f"{optional_vector(self.trimester_probabilities)} "
            "(default)"
            if self.trimester_probabilities_defaulted
            else "conditional time-to-DLT weights by third, masses "
            f"{optional_vector(self.trimester_probabilities)}"
        )
        late_text = (
            "not used for uniform event timing"
            if self.late_probability is None
            else vector(self.late_probability)
            + (" (default 0.5 at each dose)" if self.late_probability_defaulted else "")
        )
        stop_threshold = (
            "disabled" if self.early_stop_patients is None else str(self.early_stop_patients)
        )

        lines = [
            "<!doctype html>",
            '<html lang="en"><meta charset="utf-8"><title>TITE-BOIN protocol report</title>',
            "<body>",
            f"<h1>TITE-BOIN protocol summary: {html.escape(self.trial_name)}</h1>",
            "<h2>Effective design and conduct settings</h2>",
            "<dl>",
            detail("Target DLT probability", number(self.target)),
            detail(
                "Safe and toxic alternatives",
                f"{number(self.safe_probability)}, {number(self.toxic_probability)}",
            ),
            detail(
                "Exact escalation and de-escalation cutoffs",
                f"{self.escalation_boundary!r}; {self.deescalation_boundary!r}",
            ),
            detail("Safety elimination probability", number(self.elimination_probability)),
            detail(
                "Extra-safe rule and offset",
                f"{self.extra_safe}; {number(self.safety_offset)}",
            ),
            detail("Stay-at-1/3 modification", str(self.stay_at_one_of_three)),
            detail("De-escalate-at-2/6 modification", str(self.deescalate_at_two_of_six)),
            detail("Bound selected MTD", str(self.bound_mtd)),
            detail("Precision stopping threshold", stop_threshold),
            detail(
                "Assessment window",
                f"{number(self.window)} {html.escape(self.time_unit)}",
            ),
            detail(
                "Accrual rate",
                f"{number(self.accrual_rate)} patients per {html.escape(self.time_unit)}",
            ),
            detail(
                "Cohort plan and starting dose",
                f"{self.cohorts} cohorts × {self.cohort_size}; start dose {self.start_dose}; "
                f"planned maximum {self.maximum_patients}",
            ),
            detail(
                "Arrival and event-time generation",
                f"{html.escape(self.arrival)}; {html.escape(self.event_distribution)}",
            ),
            detail("Event-time late probability", late_text),
            detail("Event-time generation", event_mass_text),
            detail("Analysis follow-up weights", analysis_mass_text),
            detail(
                "Interim gates",
                f"minimum complete fraction {number(self.minimum_complete_fraction)}; "
                f"minimum pending follow-up {number(self.minimum_pending_followup)}",
            ),
            detail("Trials per scenario; root seed", f"{self.trials}; {self.seed}"),
            detail("RNG convention", html.escape(self.rng_convention)),
            "</dl>",
            "<h2>Decision sequence</h2>",
            "<ol>",
            (
                "<li>Apply enrolled-count overdose safety and retained exclusions first; "
                "elimination excludes the affected dose and higher doses, and lowest-dose "
                "elimination stops enrollment.</li>"
            ),
            (
                "<li>Compute current-dose imputation and the ordinary BOIN transition, then "
                "apply enabled modifications. The 1/3 rule (target 0.25–0.279) forces stay "
                "only with no pending patients. The 2/6 rule (target 0.28–0.33) forces "
                "de-escalation and overrides completion suspension.</li>"
            ),
            (
                "<li>Otherwise suspend when completion is below its threshold and observed "
                "toxicity has not met the de-escalation cutoff; an actual escalation also "
                "waits for minimum pending follow-up.</li>"
            ),
            (
                "<li>Apply precision stopping only when the actual next assignment stays "
                "at the current dose. Enrollment also ends at the planned cap.</li>"
            ),
            (
                "<li>After enrollment stops, wait until all enrolled DLT outcomes are ascertained "
                "before final complete-outcome selection. This summarizes simulation, not "
                "a patient-specific MTD decision.</li>"
            ),
            "</ol>",
            "<h2>Operating characteristics</h2>",
        ]
        for scenario in self.scenarios:
            lines.extend(
                [
                    f"<h3>{html.escape(scenario.name)}</h3>",
                    f"<p>Scenario seed: {scenario.seed}</p>",
                    f"<p>True dose-level DLT probabilities: {vector(scenario.true_toxicity)}</p>",
                    (
                        "<table><thead><tr><th>Outcome</th><th>Probability / mean</th>"
                        "<th>Monte Carlo SE</th></tr></thead><tbody>"
                    ),
                ]
            )
            for index, probability in enumerate(scenario.selection_probability):
                label = "No MTD" if index == 0 else f"Selected dose {index}"
                lines.append(
                    table_row(
                        label,
                        number(float(probability)),
                        number(float(scenario.selection_mcse[index])),
                    )
                )
            for index, (patients, toxicities) in enumerate(
                zip(scenario.mean_patients_by_dose, scenario.mean_toxicities_by_dose, strict=True),
                start=1,
            ):
                lines.append(
                    table_row(
                        f"Dose {index}: mean treated / DLT",
                        f"{number(float(patients))} / {number(float(toxicities))}",
                    )
                )
            lines.extend(
                [
                    table_row(
                        "Mean trial duration",
                        f"{number(scenario.mean_duration)} {html.escape(self.time_unit)}",
                    ),
                    table_row(
                        "Mean suspension time",
                        f"{number(scenario.mean_suspension_time)} {html.escape(self.time_unit)}",
                    ),
                ]
            )
            for reason, probability in scenario.stop_reason_probability:
                lines.append(table_row(f"Stop reason: {reason}", number(probability)))
            lines.append("</tbody></table>")
        lines.extend(
            [
                (
                    "<p>Simulation summaries are newly computed from the recorded settings "
                    "and seed. This Python summary is not the application's HTML/Word/PDF "
                    "template or an assertion of native scheduler or random-number parity.</p>"
                ),
                "</body></html>",
            ]
        )
        content = "\n".join(lines) + "\n"
        if len(content) > _MAX_REPORT_CHARS:
            raise ValueError("HTML report exceeds the two-million-character limit")
        return content

    def write_html(self, path: str | Path, *, digits: int = 6) -> Path:
        """Atomically replace ``path`` with the UTF-8 HTML report."""
        destination = Path(path)
        content = self.to_html(digits=digits)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            raise
        return destination


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return result


def _vector(
    value: object, name: str, *, size: int | None = None, allow_scalar: bool = False
) -> FloatArray:
    maximum = size if size is not None else 100
    if isinstance(value, np.ndarray):
        if value.ndim == 0 and allow_scalar:
            raw = value.reshape(1)
        else:
            raw = value
    elif (
        allow_scalar
        and isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    ):
        raw = np.asarray([value])
    elif isinstance(value, (list, tuple)):
        if len(value) > maximum or any(
            isinstance(item, (list, tuple, np.ndarray)) for item in value
        ):
            raise ValueError(f"{name} must be a bounded flat vector")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded vector")
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or (size is not None and raw.size != size):
        expected = "a real vector" if size is None else f"a real vector of length {size}"
        raise ValueError(f"{name} must be {expected}")
    if raw.size > maximum:
        raise ValueError(f"{name} exceeds the bounded vector length {maximum}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = finite(raw, name)
    return _owned(result)


def _optional_mass(value: ArrayLike | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    result = _vector(value, name, size=3)
    if np.any((result < 0) | (result > 1)) or not math.isclose(
        float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} must contain three nonnegative probabilities summing to one")
    return result


def _mean(values: FloatArray) -> float:
    scale = float(np.max(values, initial=0.0))
    return 0.0 if scale == 0 else float(np.mean(values / scale) * scale)


def _capture_design(requested: BOINDesign) -> BOINDesign:
    """Revalidate an independent design while preserving its exact cutoffs."""
    if requested.safe_probability is None or requested.toxic_probability is None:
        raise ValueError("design must have effective safe and toxic probabilities")
    captured = BOINDesign(
        requested.target,
        requested.safe_probability,
        requested.toxic_probability,
        elimination_probability=requested.elimination_probability,
        extra_safe=requested.extra_safe,
        safety_offset=requested.safety_offset,
        stay_at_one_of_three=requested.stay_at_one_of_three,
        deescalate_at_two_of_six=requested.deescalate_at_two_of_six,
        bound_mtd=requested.bound_mtd,
        early_stop_patients=requested.early_stop_patients,
    )
    tolerance = 64 * np.finfo(float).eps
    for name in ("escalation_boundary", "deescalation_boundary"):
        original = float(getattr(requested, name))
        reconstructed = float(getattr(captured, name))
        if not math.isfinite(original) or abs(reconstructed - original) > tolerance * original:
            raise ValueError(f"design {name} is inconsistent with its probability alternatives")
        object.__setattr__(captured, name, original)
    return captured


def run_tite_boin_protocol(request: TITEBOINProtocolRequest) -> TITEBOINProtocolReport:
    """Run serial scenario simulations and capture an immutable HTML report."""
    if not isinstance(request, TITEBOINProtocolRequest):
        raise TypeError("request must be a TITEBOINProtocolRequest")
    if not isinstance(request.design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    if (
        not isinstance(request.trial_name, str)
        or not request.trial_name.strip()
        or len(request.trial_name) > _MAX_LABEL_CHARS
    ):
        raise ValueError(f"trial_name must be nonempty and at most {_MAX_LABEL_CHARS} characters")
    if (
        not isinstance(request.time_unit, str)
        or not request.time_unit.strip()
        or len(request.time_unit) > _MAX_LABEL_CHARS
    ):
        raise ValueError(f"time_unit must be nonempty and at most {_MAX_LABEL_CHARS} characters")
    if (
        not isinstance(request.scenarios, (list, tuple))
        or not 1 <= len(request.scenarios) <= _MAX_SCENARIOS
    ):
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} entries")
    window = scalar(request.window, "window")
    accrual_rate = scalar(request.accrual_rate, "accrual_rate")
    if window <= 0 or accrual_rate <= 0 or not math.isfinite(1.0 / accrual_rate):
        raise ValueError("window and accrual_rate must be positive and representable")
    cohorts = _integer(request.cohorts, "cohorts", 1, 200)
    cohort_size = _integer(request.cohort_size, "cohort_size", 1, 200)
    trials = _integer(request.trials, "trials", 1, 100_000)
    seed = _integer(request.seed, "seed", 0, 2**32 - 1)
    maximum_patients = cohorts * cohort_size
    if maximum_patients > 200:
        raise ValueError("planned enrollment must not exceed 200 patients")
    # First inspect each scenario's bounded vector shape, then materialize and validate it.
    scenario_inputs: list[tuple[str, object]] = []
    names: set[str] = set()
    total_patient_replicates = 0
    dose_count: int | None = None
    for scenario in request.scenarios:
        if not isinstance(scenario, TITEBOINScenario):
            raise ValueError("each scenario must be a TITEBOINScenario")
        if (
            not isinstance(scenario.name, str)
            or not scenario.name.strip()
            or len(scenario.name) > _MAX_LABEL_CHARS
        ):
            raise ValueError(
                f"scenario names must be nonempty and at most {_MAX_LABEL_CHARS} characters"
            )
        if scenario.name in names:
            raise ValueError("scenario names must be unique")
        names.add(scenario.name)
        value = scenario.true_toxicity
        if isinstance(value, np.ndarray):
            if (
                value.ndim != 1
                or value.size < 2
                or value.size > 100
                or value.dtype.kind not in "iuf"
            ):
                raise ValueError("true_toxicity must be a real vector for 2..100 doses")
            current_doses = int(value.size)
        elif isinstance(value, (list, tuple)):
            if not 2 <= len(value) <= 100 or any(
                isinstance(item, (list, tuple, np.ndarray)) for item in value
            ):
                raise ValueError("true_toxicity must be a flat real vector for 2..100 doses")
            current_doses = len(value)
        else:
            raise ValueError("true_toxicity must be a bounded real vector")
        if dose_count is None:
            dose_count = current_doses
        elif current_doses != dose_count:
            raise ValueError("all scenario vectors must have the same dose count")
        total_patient_replicates += trials * maximum_patients
        scenario_inputs.append((scenario.name, value))
    assert dose_count is not None
    start = _integer(request.start_dose, "start_dose", 1, dose_count)
    if trials * dose_count > _MAX_TRIAL_DOSE_CELLS:
        raise ValueError("trial-by-dose output exceeds the per-scenario cell limit")
    if total_patient_replicates > _MAX_PATIENT_REPLICATES:
        raise ValueError("aggregate simulated patient replications exceed the configured limit")
    if request.arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    if request.event_distribution not in ("uniform", "weibull", "log-logistic"):
        raise ValueError("unsupported event_distribution")
    complete = scalar(request.minimum_complete_fraction, "minimum_complete_fraction")
    minimum = scalar(request.minimum_pending_followup, "minimum_pending_followup")
    if not 0.25 <= complete <= 1 or not 0 <= minimum <= 1:
        raise ValueError("require completion fraction in [.25,1] and follow-up fraction in [0,1]")
    event_masses = _optional_mass(
        request.event_trimester_probabilities, "event_trimester_probabilities"
    )
    analysis_masses = _optional_mass(request.trimester_probabilities, "trimester_probabilities")
    if request.event_distribution != "uniform" and event_masses is not None:
        raise ValueError("event trimester masses require uniform event timing")
    toxicity_followup_weights([], window, trimester_probabilities=event_masses)
    toxicity_followup_weights([], window, trimester_probabilities=analysis_masses)
    late: FloatArray | None
    late_defaulted = request.late_probability is None
    if request.late_probability is None:
        late = None if request.event_distribution == "uniform" else _owned(np.full(dose_count, 0.5))
    else:
        late = _vector(request.late_probability, "late_probability", allow_scalar=True)
        if late.size not in (1, dose_count) or np.any((late < 0) | (late > 1)):
            raise ValueError("late_probability must be in [0,1] and scalar or dose-length")
        if late.size == 1:
            late = _owned(np.full(dose_count, late[0]))
    event_masses_defaulted = request.event_trimester_probabilities is None
    effective_event_masses = (
        (_owned(np.full(3, 1 / 3)) if event_masses_defaulted else event_masses)
        if request.event_distribution == "uniform"
        else None
    )
    analysis_masses_defaulted = request.trimester_probabilities is None
    effective_analysis_masses = (
        _owned(np.full(3, 1 / 3)) if analysis_masses_defaulted else analysis_masses
    )
    probability_vectors = [
        (name, _vector(value, f"scenario {name} true_toxicity", size=dose_count))
        for name, value in scenario_inputs
    ]
    if any(np.any((values < 0) | (values > 1)) for _, values in probability_vectors):
        raise ValueError("true_toxicity probabilities must be in [0,1]")
    # Let the shared quantile validator check the complete timing calibration before RNG use.
    for _, values in probability_vectors:
        toxicity_time_quantile(
            0.5, values, window, distribution=request.event_distribution, late_probability=late
        )
    design = _capture_design(request.design)
    assert design.safe_probability is not None and design.toxic_probability is not None
    child_sequences = np.random.SeedSequence(seed).spawn(len(probability_vectors))
    scenario_seeds = [
        sum(
            int(word) << (32 * index)
            for index, word in enumerate(child.generate_state(4, dtype=np.uint32))
        )
        for child in child_sequences
    ]
    summaries: list[TITEBOINScenarioSummary] = []
    for (name, values), scenario_seed in zip(probability_vectors, scenario_seeds, strict=True):
        simulation: TITEBOINSimulation = simulate_tite_boin(
            design,
            values,
            window,
            accrual_rate,
            cohorts=cohorts,
            cohort_size=cohort_size,
            trials=trials,
            start_dose=start,
            arrival=request.arrival,
            event_distribution=request.event_distribution,
            late_probability=late,
            event_trimester_probabilities=event_masses,
            trimester_probabilities=analysis_masses,
            minimum_complete_fraction=complete,
            minimum_pending_followup=minimum,
            rng=scenario_seed,
        )
        statuses = Counter(simulation.stop_reason)
        summaries.append(
            TITEBOINScenarioSummary(
                name,
                scenario_seed,
                _owned(values),
                _owned(simulation.selection_probability),
                _owned(simulation.selection_mcse),
                _owned(np.mean(simulation.patients, axis=0)),
                _owned(np.mean(simulation.toxicities, axis=0)),
                _mean(simulation.duration),
                _mean(simulation.suspension_time),
                tuple((reason, statuses[reason] / trials) for reason in sorted(statuses)),
            )
        )
        del simulation
    return TITEBOINProtocolReport(
        request.trial_name,
        design.target,
        float(design.safe_probability),
        float(design.toxic_probability),
        design.escalation_boundary,
        design.deescalation_boundary,
        design.elimination_probability,
        design.safety_offset,
        design.extra_safe,
        design.stay_at_one_of_three,
        design.deescalate_at_two_of_six,
        design.bound_mtd,
        design.early_stop_patients,
        window,
        accrual_rate,
        request.time_unit,
        cohorts,
        cohort_size,
        start,
        maximum_patients,
        request.arrival,
        request.event_distribution,
        None if late is None else _owned(late),
        late_defaulted,
        effective_event_masses,
        event_masses_defaulted,
        effective_analysis_masses,
        analysis_masses_defaulted,
        complete,
        minimum,
        trials,
        seed,
        "One 128-bit integer seed derived from SeedSequence.spawn per scenario in request order; "
        "each simulation runs serially with its recorded seed.",
        tuple(summaries),
    )
