"""Reproducible TITE-Keyboard protocol and scenario OC reports."""

from __future__ import annotations

import html
import math
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .keyboard import KeyboardDesign
from .tite_keyboard_adaptive_calendar import TITEKeyboardAdaptiveSettings
from .tite_keyboard_boundaries import TITEKeyboardBoundaries, tite_keyboard_boundaries
from .tite_keyboard_simulation import TITEKeyboardSimulation, simulate_tite_keyboard
from .toxicity_timing import toxicity_time_quantile

_MAX_SCENARIOS = 20
_MAX_DOSES = 100
_MAX_PATIENTS = 200
_MAX_TRIAL_DOSE_CELLS = 1_000_000
_MAX_PATIENT_REPLICATES = 2_000_000
_MAX_AGGREGATE_ADAPTIVE_WORK = 1_000_000_000
_MAX_REPORT_CHARS = 2_000_000
_MAX_LABEL_CHARS = 256


@dataclass(frozen=True, slots=True)
class TITEKeyboardScenario:
    """One named vector of true dose-level DLT probabilities."""

    name: str
    true_toxicity: ArrayLike


@dataclass(frozen=True, slots=True)
class TITEKeyboardProtocolRequest:
    """Captured design, calendar plan, scenarios and reproducibility settings."""

    trial_name: str
    design: KeyboardDesign
    scenarios: Sequence[TITEKeyboardScenario]
    window: float
    accrual_rate: float
    time_unit: str = "days"
    cohorts: int = 10
    cohort_size: int = 3
    start_dose: int = 1
    trials: int = 1000
    arrival: Literal["fixed", "exponential"] = "fixed"
    event_distribution: Literal["uniform", "weibull", "log-logistic"] = "uniform"
    late_probability: ArrayLike | None = None
    event_trimester_probabilities: ArrayLike | None = None
    trimester_probabilities: ArrayLike | None = None
    pending_fraction_limit: float | None = 0.5
    adaptive_timing: TITEKeyboardAdaptiveSettings | None = None
    seed: int = 6
    max_aggregate_adaptive_work: int = 100_000_000


@dataclass(frozen=True, slots=True)
class TITEKeyboardScenarioSummary:
    """Compact immutable operating characteristics for one scenario."""

    name: str
    scenario_seed: int
    outcome_seed: int | None
    sampler_seed: int | None
    true_toxicity: FloatArray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients_by_dose: FloatArray
    mean_patients_mcse: FloatArray
    mean_toxicities_by_dose: FloatArray
    mean_toxicities_mcse: FloatArray
    mean_duration: float
    mean_duration_mcse: float
    mean_suspension_time: float
    mean_suspension_mcse: float
    stop_reason_probability: tuple[tuple[str, float, float], ...]
    adaptive_fit_count: int
    adaptive_work_units: int
    adaptive_diagnostics_passed: int | None
    max_adaptive_split_rhat: float | None
    max_adaptive_weight_mcse: float | None
    adaptive_work_budget: int


@dataclass(frozen=True, slots=True)
class TITEKeyboardProtocolReport:
    """Effective TITE-Keyboard settings, transition boundaries, and scenario OC."""

    trial_name: str
    target: float
    lower: float
    upper: float
    target_key: int
    edge_rule: str
    elimination_probability: float
    extra_safe: bool
    safety_offset: float
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
    pending_fraction_limit: float | None
    trials: int
    seed: int
    rng_convention: str
    adaptive_settings: TITEKeyboardAdaptiveSettings | None
    max_aggregate_adaptive_work: int
    boundary_table: TITEKeyboardBoundaries
    scenarios: tuple[TITEKeyboardScenarioSummary, ...]

    def to_html(self, *, digits: int = 6) -> str:
        """Render a bounded, escaped, standalone HTML protocol report."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")

        def number(value: float) -> str:
            return format(float(value), f".{digits}g")

        def vector(values: FloatArray) -> str:
            return ", ".join(number(float(value)) for value in values)

        def item(label: str, value: str) -> str:
            return f"<dt>{html.escape(label)}</dt><dd>{value}</dd>"

        late_text = (
            "not used for uniform event timing"
            if self.late_probability is None
            else vector(self.late_probability)
            + (" (default 0.5 per dose)" if self.late_probability_defaulted else "")
        )
        if self.event_distribution == "uniform":
            event_mass_text = vector(self.event_trimester_probabilities)  # type: ignore[arg-type]
            if self.event_trimester_probabilities_defaulted:
                event_mass_text += " (uniform default)"
        else:
            event_mass_text = "not used for parametric timing"
        if self.adaptive_settings is not None:
            analysis_mass_text = "not used with adaptive timing"
        else:
            assert self.trimester_probabilities is not None
            analysis_mass_text = vector(self.trimester_probabilities)
            if self.trimester_probabilities_defaulted:
                analysis_mass_text += " (uniform default)"
        pending_gate = (
            "disabled"
            if self.pending_fraction_limit is None
            else f"strictly greater than {number(self.pending_fraction_limit)}"
        )
        precision = (
            "disabled" if self.early_stop_patients is None else str(self.early_stop_patients)
        )
        adaptive_text = "disabled"
        if self.adaptive_settings is not None:
            settings = self.adaptive_settings
            adaptive_text = (
                f"Gamma(lambda)={settings.lambda_prior!r}; Gamma(gamma)={settings.gamma_prior!r}; "
                f"chains={settings.chains}, warmup={settings.warmup}, draws={settings.draws}; "
                f"fit work cap={settings.max_fit_work}; per-simulation cap="
                f"{settings.max_total_work}; report-wide cap={self.max_aggregate_adaptive_work}; "
                f"R-hat threshold={number(settings.max_split_rhat)}; weight MCSE threshold="
                f"{number(settings.max_weight_mcse)}"
            )

        lines = [
            "<!doctype html>",
            '<html lang="en"><meta charset="utf-8"><title>TITE-Keyboard protocol report</title>',
            "<body>",
            f"<h1>TITE-Keyboard protocol: {html.escape(self.trial_name)}</h1>",
            "<h2>Effective design and conduct settings</h2>",
            "<dl>",
            item("Target toxicity probability", number(self.target)),
            item(
                "Target interval and key",
                f"[{number(self.lower)}, {number(self.upper)}]; key {self.target_key}",
            ),
            item("Endpoint-key rule", html.escape(self.edge_rule)),
            item(
                "Overdose cutoff and extra-safe rule",
                f"{number(self.elimination_probability)}; {self.extra_safe}",
            ),
            item("Extra-safe offset", number(self.safety_offset)),
            item("Precision stopping threshold", precision),
            item("Assessment window", f"{number(self.window)} {html.escape(self.time_unit)}"),
            item(
                "Accrual rate",
                f"{number(self.accrual_rate)} patients per {html.escape(self.time_unit)}",
            ),
            item(
                "Plan",
                f"{self.cohorts} cohorts × {self.cohort_size}; start dose "
                f"{self.start_dose}; maximum {self.maximum_patients}",
            ),
            item(
                "Arrival and DLT-time generation",
                f"{html.escape(self.arrival)}; {html.escape(self.event_distribution)}",
            ),
            item("Late-half DLT probability", late_text),
            item("Conditional DLT-time masses by third for generation", event_mass_text),
            item("Conditional DLT-time masses by third for follow-up weights", analysis_mass_text),
            item("Pending-fraction suspension gate", pending_gate),
            item("Replicates and root seed", f"{self.trials} trials per scenario; {self.seed}"),
            item("Adaptive timing sampler", html.escape(adaptive_text)),
            item("RNG convention", html.escape(self.rng_convention)),
            "</dl>",
            "<h2>Decision flow</h2>",
            "<ol>",
            (
                "<li>At an interim, count observed DLTs and apply the enrolled-count "
                "overdose rule; an unsafe lowest dose stops, and excluded doses and "
                "higher doses remain unavailable.</li>"
            ),
            (
                "<li>If the current dose is excluded, move to the highest nonexcluded "
                "dose. Otherwise, suspend if the current-dose pending fraction is "
                "strictly above its configured gate.</li>"
            ),
            (
                "<li>Compute effective sample size from complete outcomes and pending "
                "follow-up weights, then select the dose move from posterior key masses.</li>"
            ),
            (
                "<li>If the proposed move escalates before two outcomes are ascertained "
                "at the current dose, suspend until an assessment. Otherwise apply the "
                "precision stop threshold, then assign the adjacent dose or stay at a "
                "boundary.</li>"
            ),
            (
                "<li>After enrollment stops, ascertain all outcomes and select an MTD "
                "from complete dose-level counts.</li>"
            ),
            "</ol>",
            (
                "<p>The table below gives posterior transition brackets in effective "
                "non-DLT count. Safety cutoffs are separate and take precedence.</p>"
            ),
            "<h2>Posterior transitions and safety cutoffs</h2>",
            (
                "<table><thead><tr><th>Observed DLTs</th>"
                "<th>Stay/escalate bracket</th><th>Escalation bracket</th>"
                "</tr></thead><tbody>"
            ),
        ]
        for row, dlt in enumerate(self.boundary_table.toxicities):
            stay = self.boundary_table.stay_or_escalate[row]
            escalation = self.boundary_table.escalate[row]
            lines.append(
                "<tr><td>"
                + str(int(dlt))
                + "</td><td>"
                + html.escape(", ".join(format(float(value), ".17g") for value in stay))
                + "</td><td>"
                + html.escape(", ".join(format(float(value), ".17g") for value in escalation))
                + "</td></tr>"
            )
        lines.append("</tbody></table>")
        lines.append(
            "<h3>Enrolled-count safety cutoffs</h3>"
            "<table><thead><tr><th>Enrolled patients</th>"
            "<th>First DLT count eliminating a dose</th>"
            "<th>First DLT count stopping at lowest dose</th></tr></thead><tbody>"
        )
        for patients, eliminate, lowest in zip(
            self.boundary_table.enrolled_patients,
            self.boundary_table.eliminate_min,
            self.boundary_table.lowest_stop_min,
            strict=True,
        ):
            lines.append(
                f"<tr><td>{int(patients)}</td><td>{int(eliminate)}</td><td>{int(lowest)}</td></tr>"
            )
        lines.append("</tbody></table>")
        lines.append("<h2>Scenario operating characteristics</h2>")
        for scenario in self.scenarios:
            outcome_seed_label = (
                "single stream" if scenario.outcome_seed is None else str(scenario.outcome_seed)
            )
            sampler_seed_label = (
                "not used" if scenario.sampler_seed is None else str(scenario.sampler_seed)
            )
            lines.extend(
                [
                    f"<h3>{html.escape(scenario.name)}</h3>",
                    f"<p>Scenario seed: {scenario.scenario_seed}; "
                    f"outcome stream seed: {outcome_seed_label}; "
                    "timing-sampler seed: "
                    f"{sampler_seed_label}.</p>",
                    f"<p>True dose-level DLT probabilities: {vector(scenario.true_toxicity)}</p>",
                    "<table><thead><tr><th>Outcome</th><th>Probability/mean</th><th>MCSE</th></tr></thead><tbody>",
                ]
            )
            for index, probability in enumerate(scenario.selection_probability):
                label = "No MTD" if index == 0 else f"Selected dose {index}"
                lines.append(
                    f"<tr><td>{label}</td><td>{number(float(probability))}</td>"
                    f"<td>{number(float(scenario.selection_mcse[index]))}</td></tr>"
                )
            for dose, (patients, patients_se, toxicities, toxicities_se) in enumerate(
                zip(
                    scenario.mean_patients_by_dose,
                    scenario.mean_patients_mcse,
                    scenario.mean_toxicities_by_dose,
                    scenario.mean_toxicities_mcse,
                    strict=True,
                ),
                start=1,
            ):
                lines.append(
                    f"<tr><td>Dose {dose}: mean patients treated</td>"
                    f"<td>{number(float(patients))}</td><td>{number(float(patients_se))}</td></tr>"
                )
                lines.append(
                    f"<tr><td>Dose {dose}: mean DLTs</td>"
                    f"<td>{number(float(toxicities))}</td><td>{number(float(toxicities_se))}</td></tr>"
                )
            lines.extend(
                [
                    f"<tr><td>Mean trial duration ({html.escape(self.time_unit)})</td>"
                    f"<td>{number(scenario.mean_duration)}</td><td>{number(scenario.mean_duration_mcse)}</td></tr>",
                    f"<tr><td>Mean suspension time ({html.escape(self.time_unit)})</td>"
                    f"<td>{number(scenario.mean_suspension_time)}</td><td>{number(scenario.mean_suspension_mcse)}</td></tr>",
                ]
            )
            for reason, probability, mcse in scenario.stop_reason_probability:
                lines.append(
                    f"<tr><td>Stop reason: {html.escape(reason)}</td>"
                    f"<td>{number(probability)}</td><td>{number(mcse)}</td></tr>"
                )
            if self.adaptive_settings is not None:
                passed_count = (
                    "not reported"
                    if scenario.adaptive_diagnostics_passed is None
                    else str(scenario.adaptive_diagnostics_passed)
                )
                rhat = (
                    math.nan
                    if scenario.max_adaptive_split_rhat is None
                    else scenario.max_adaptive_split_rhat
                )
                weight_mcse = (
                    math.nan
                    if scenario.max_adaptive_weight_mcse is None
                    else scenario.max_adaptive_weight_mcse
                )
                lines.append(
                    "<tr><td>Adaptive fits passing diagnostics / total</td>"
                    f"<td>{passed_count} / {scenario.adaptive_fit_count}</td>"
                    "<td>—</td></tr>"
                )
                lines.append(
                    "<tr><td>Adaptive work / assigned scenario budget</td>"
                    f"<td>{scenario.adaptive_work_units} / "
                    f"{scenario.adaptive_work_budget}</td><td>—</td></tr>"
                )
                lines.append(
                    "<tr><td>Maximum adaptive split R-hat / pending-weight MCSE</td>"
                    f"<td>{number(rhat)} / {number(weight_mcse)}"
                    "</td><td>—</td></tr>"
                )
            lines.append("</tbody></table>")
        lines.extend(
            [
                (
                    "<p>Scenario results are newly simulated from these inputs. Means have "
                    "Monte Carlo standard errors; patient histories are not retained. "
                    "Adaptive diagnostics are empirical checks, not convergence guarantees.</p>"
                ),
                (
                    "<p>Boundary pairs show effective non-DLT counts immediately below "
                    "and at the posterior transition; NaN marks an unreachable transition. "
                    "Safety cutoffs use enrolled patients and are separate.</p>"
                ),
                (
                    "<p>This self-contained Python HTML and decision-flow summary do not "
                    "reproduce native HTML/Word files, exact Figure 1/Table 1 formatting, "
                    "random streams, or the hidden operating-characteristic table. Its "
                    "column definitions are not exposed by the cached app page; no native "
                    "correct-selection/regret estimand is asserted.</p>"
                ),
                "</body></html>",
            ]
        )
        content = "\n".join(lines) + "\n"
        if len(content) > _MAX_REPORT_CHARS:
            raise ValueError("HTML report exceeds the two-million-character limit")
        return content

    def write_html(self, path: str | Path, *, digits: int = 6) -> Path:
        """Atomically write this report as UTF-8 HTML."""
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


def _vector(value: object, name: str, *, size: int | None = None, maximum: int = 100) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > maximum or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
        raw = value
    elif isinstance(value, (list, tuple)):
        if len(value) > maximum or any(
            isinstance(item, (list, tuple, np.ndarray)) for item in value
        ):
            raise ValueError(f"{name} must be a bounded flat vector")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded vector")
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or (size is not None and raw.size != size):
        raise ValueError(
            f"{name} must be a real vector" if size is None else f"{name} must have length {size}"
        )
    if raw.size > maximum:
        raise ValueError(f"{name} exceeds {maximum} values")
    with np.errstate(over="ignore", invalid="ignore"):
        return _owned(finite(raw, name))


def _probability_scalar_or_vector(value: object, name: str, *, size: int) -> FloatArray:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a probability scalar or vector")
    if np.isscalar(value):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real probability scalar")
        raw = np.asarray([scalar(float(value), name)], dtype=np.float64)
    else:
        raw = _vector(value, name, maximum=size)
    if raw.size not in (1, size) or np.any((raw < 0) | (raw > 1)):
        raise ValueError(f"{name} must be in [0,1] and scalar or dose-length")
    if raw.size == 1:
        return _owned(np.full(size, raw[0]))
    return _owned(raw)


def _optional_masses(value: ArrayLike | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    masses = _vector(value, name, size=3)
    if np.any((masses < 0) | (masses > 1)) or not math.isclose(
        float(masses.sum()), 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} must contain three probabilities summing to one")
    return masses


def _capture_design(requested: KeyboardDesign) -> KeyboardDesign:
    if not isinstance(requested, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
    captured = KeyboardDesign(
        target=requested.target,
        lower=requested.lower,
        upper=requested.upper,
        edge_rule=requested.edge_rule,
        elimination_probability=requested.elimination_probability,
        extra_safe=requested.extra_safe,
        safety_offset=requested.safety_offset,
        early_stop_patients=requested.early_stop_patients,
    )
    intervals = finite(requested.intervals, "design.intervals")
    if intervals.shape != captured.intervals.shape or not np.array_equal(
        intervals, captured.intervals
    ):
        raise ValueError("design intervals do not match its recorded key settings")
    if requested.target_key != captured.target_key:
        raise ValueError("design target key does not match its interval settings")
    if captured.lower is None or captured.upper is None:
        raise ArithmeticError("captured design does not have a target interval")
    object.__setattr__(captured, "intervals", _owned(intervals))
    return captured


def _mean_se(values: FloatArray) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0:
        return 0.0, 0.0
    scaled = values / scale
    mean = float(np.mean(scaled)) * scale
    if values.size < 2:
        return mean, math.nan
    se = float(np.std(scaled, ddof=1) / math.sqrt(values.size)) * scale
    return mean, se


def _scenario_seed_words(seed: int, count: int) -> list[int]:
    children = np.random.SeedSequence(seed).spawn(count)
    return [
        sum(
            int(word) << (32 * index)
            for index, word in enumerate(child.generate_state(4, dtype=np.uint32))
        )
        for child in children
    ]


def run_tite_keyboard_protocol(
    request: TITEKeyboardProtocolRequest,
) -> TITEKeyboardProtocolReport:
    """Revalidate inputs, preflight budgets, then simulate each scenario serially."""
    if not isinstance(request, TITEKeyboardProtocolRequest):
        raise TypeError("request must be a TITEKeyboardProtocolRequest")
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
        not isinstance(request.scenarios, (tuple, list))
        or not 1 <= len(request.scenarios) <= _MAX_SCENARIOS
    ):
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} entries")
    window, rate = scalar(request.window, "window"), scalar(request.accrual_rate, "accrual_rate")
    if window <= 0 or rate <= 0 or not math.isfinite(1 / rate):
        raise ValueError("window and accrual_rate must be positive and representable")
    cohorts = _integer(request.cohorts, "cohorts", 1, 100)
    cohort_size = _integer(request.cohort_size, "cohort_size", 1, 10)
    trials = _integer(request.trials, "trials", 1, 100_000)
    seed = _integer(request.seed, "seed", 0, 2**64 - 1)
    maximum_patients = cohorts * cohort_size
    if maximum_patients > _MAX_PATIENTS:
        raise ValueError(f"planned enrollment must not exceed {_MAX_PATIENTS} patients")
    if request.arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    if request.event_distribution not in ("uniform", "weibull", "log-logistic"):
        raise ValueError("unsupported event_distribution")
    dose_count: int | None = None
    names: set[str] = set()
    scenario_inputs: list[tuple[str, object]] = []
    total_patient_replicates = 0
    for scenario in request.scenarios:
        if not isinstance(scenario, TITEKeyboardScenario):
            raise ValueError("each scenario must be a TITEKeyboardScenario")
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
        raw = scenario.true_toxicity
        if isinstance(raw, np.ndarray):
            if raw.ndim != 1 or not 2 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
                raise ValueError("true_toxicity must be a real vector for 2..100 doses")
            current_doses = int(raw.size)
        elif isinstance(raw, (list, tuple)):
            if not 2 <= len(raw) <= _MAX_DOSES or any(
                isinstance(item, (list, tuple, np.ndarray)) for item in raw
            ):
                raise ValueError("true_toxicity must be a flat vector for 2..100 doses")
            current_doses = len(raw)
        else:
            raise ValueError("true_toxicity must be a bounded real vector")
        if dose_count is None:
            dose_count = current_doses
        elif current_doses != dose_count:
            raise ValueError("all scenarios must have the same number of doses")
        if trials * current_doses > _MAX_TRIAL_DOSE_CELLS:
            raise ValueError("trial-by-dose output exceeds the per-scenario cell limit")
        total_patient_replicates += trials * maximum_patients
        scenario_inputs.append((scenario.name, raw))
    assert dose_count is not None
    if total_patient_replicates > _MAX_PATIENT_REPLICATES:
        raise ValueError("aggregate simulated patient replications exceed the configured limit")
    start_dose = _integer(request.start_dose, "start_dose", 1, dose_count)
    if request.pending_fraction_limit is None:
        pending_limit = None
    else:
        pending_limit = scalar(request.pending_fraction_limit, "pending_fraction_limit")
        if not 0 < pending_limit <= 0.65:
            raise ValueError("pending_fraction_limit must be in (0,.65] or None")
    aggregate_work_budget = _integer(
        request.max_aggregate_adaptive_work,
        "max_aggregate_adaptive_work",
        1,
        _MAX_AGGREGATE_ADAPTIVE_WORK,
    )
    if request.adaptive_timing is not None and not isinstance(
        request.adaptive_timing, TITEKeyboardAdaptiveSettings
    ):
        raise ValueError("adaptive_timing must be TITEKeyboardAdaptiveSettings or None")
    if request.adaptive_timing is not None:
        if request.adaptive_timing.max_total_work <= 0 or request.adaptive_timing.max_fit_work <= 0:
            raise ValueError("adaptive work limits must be positive")
    masses_event = _optional_masses(
        request.event_trimester_probabilities, "event_trimester_probabilities"
    )
    masses_analysis = _optional_masses(request.trimester_probabilities, "trimester_probabilities")
    if request.event_distribution != "uniform" and masses_event is not None:
        raise ValueError("event_trimester_probabilities are only used with uniform event timing")
    if request.adaptive_timing is not None and masses_analysis is not None:
        raise ValueError("trimester_probabilities cannot be combined with adaptive_timing")
    if request.event_distribution == "uniform" and request.late_probability is not None:
        raise ValueError("late_probability is only used with parametric event timing")
    event_masses_defaulted = masses_event is None
    analysis_masses_defaulted = masses_analysis is None
    effective_event_masses = (
        (_owned(np.full(3, 1 / 3)) if masses_event is None else masses_event)
        if request.event_distribution == "uniform"
        else None
    )
    effective_analysis_masses = (
        _owned(np.full(3, 1 / 3)) if masses_analysis is None else masses_analysis
    )
    late_defaulted = request.late_probability is None
    if request.event_distribution == "uniform":
        late_probability = None
    else:
        late_probability = (
            _owned(np.full(dose_count, 0.5))
            if request.late_probability is None
            else _probability_scalar_or_vector(
                request.late_probability, "late_probability", size=dose_count
            )
        )
    scenarios = [
        (name, _vector(values, f"scenario {name} true_toxicity", size=dose_count))
        for name, values in scenario_inputs
    ]
    if any(np.any((values < 0) | (values > 1)) for _, values in scenarios):
        raise ValueError("true_toxicity probabilities must be in [0,1]")
    for _, values in scenarios:
        toxicity_time_quantile(
            0.5,
            values,
            window,
            distribution=request.event_distribution,
            late_probability=late_probability,
        )
    design = _capture_design(request.design)
    lower, upper = design.lower, design.upper
    if lower is None or upper is None:
        raise ArithmeticError("captured Keyboard design has no target interval")
    boundaries = tite_keyboard_boundaries(design, maximum_patients)
    adaptive_settings = (
        None if request.adaptive_timing is None else replace(request.adaptive_timing)
    )
    scenario_seeds = _scenario_seed_words(seed, len(scenarios))
    summaries: list[TITEKeyboardScenarioSummary] = []
    adaptive_work_remaining = aggregate_work_budget
    for index, ((name, true_toxicity), scenario_seed) in enumerate(
        zip(scenarios, scenario_seeds, strict=True)
    ):
        simulation_settings = adaptive_settings
        scenario_work_budget = 0
        if simulation_settings is not None:
            scenario_work_budget = min(
                simulation_settings.max_total_work,
                adaptive_work_remaining,
            )
            if scenario_work_budget < 1:
                raise ValueError("adaptive-work budget is exhausted before the next scenario")
            simulation_settings = replace(
                simulation_settings,
                max_fit_work=min(simulation_settings.max_fit_work, scenario_work_budget),
                max_total_work=scenario_work_budget,
            )
        simulation: TITEKeyboardSimulation = simulate_tite_keyboard(
            design,
            true_toxicity,
            window,
            rate,
            cohorts=cohorts,
            cohort_size=cohort_size,
            trials=trials,
            start_dose=start_dose,
            arrival=request.arrival,
            event_distribution=request.event_distribution,
            late_probability=late_probability,
            event_trimester_probabilities=(
                None if event_masses_defaulted else effective_event_masses
            ),
            trimester_probabilities=None
            if adaptive_settings is not None
            else effective_analysis_masses,
            pending_fraction_limit=pending_limit,
            adaptive_timing=simulation_settings,
            rng=scenario_seed,
        )
        selection = _owned(simulation.selection_probability)
        selection_mcse = _owned(simulation.selection_mcse)
        mean_patients = _owned(simulation.patients.mean(axis=0))
        mean_toxicities = _owned(simulation.toxicities.mean(axis=0))
        patient_mcse = (
            _owned(np.std(simulation.patients, axis=0, ddof=1) / math.sqrt(trials))
            if trials > 1
            else _owned(np.full(dose_count, math.nan))
        )
        toxicity_mcse = (
            _owned(np.std(simulation.toxicities, axis=0, ddof=1) / math.sqrt(trials))
            if trials > 1
            else _owned(np.full(dose_count, math.nan))
        )
        mean_duration, duration_mcse = _mean_se(simulation.duration)
        mean_suspension, suspension_mcse = _mean_se(simulation.suspension_time)
        counts = Counter(simulation.stop_reason)
        stop_summaries = tuple(
            (
                reason,
                counts[reason] / trials,
                math.sqrt((counts[reason] / trials) * (1 - counts[reason] / trials) / trials),
            )
            for reason in sorted(counts)
        )
        adaptive_fit_count = (
            int(simulation.adaptive_fit_count) if adaptive_settings is not None else 0
        )
        adaptive_work = int(simulation.adaptive_work_units) if adaptive_settings is not None else 0
        adaptive_diagnostic_count = (
            int(simulation.adaptive_diagnostics_passed) if adaptive_settings is not None else None
        )
        if adaptive_settings is not None and (
            adaptive_fit_count < 0
            or adaptive_diagnostic_count is None
            or not 0 <= adaptive_diagnostic_count <= adaptive_fit_count
            or adaptive_work < 0
            or adaptive_work > scenario_work_budget
        ):
            raise ArithmeticError("adaptive simulation returned inconsistent budget diagnostics")
        if adaptive_settings is not None:
            adaptive_work_remaining -= adaptive_work
        summaries.append(
            TITEKeyboardScenarioSummary(
                name,
                scenario_seed,
                getattr(simulation, "outcome_seed", None),
                getattr(simulation, "sampler_seed", None),
                _owned(true_toxicity),
                selection,
                selection_mcse,
                mean_patients,
                patient_mcse,
                mean_toxicities,
                toxicity_mcse,
                mean_duration,
                duration_mcse,
                mean_suspension,
                suspension_mcse,
                stop_summaries,
                adaptive_fit_count,
                adaptive_work,
                adaptive_diagnostic_count,
                simulation.max_adaptive_split_rhat if adaptive_settings is not None else None,
                simulation.max_adaptive_weight_mcse if adaptive_settings is not None else None,
                scenario_work_budget,
            )
        )
        del simulation
    if request.event_distribution == "uniform":
        late_result = None
    else:
        assert late_probability is not None
        late_result = _owned(late_probability)
    return TITEKeyboardProtocolReport(
        request.trial_name,
        design.target,
        lower,
        upper,
        design.target_key,
        design.edge_rule,
        design.elimination_probability,
        design.extra_safe,
        design.safety_offset,
        design.early_stop_patients,
        window,
        rate,
        request.time_unit,
        cohorts,
        cohort_size,
        start_dose,
        maximum_patients,
        request.arrival,
        request.event_distribution,
        late_result,
        late_defaulted,
        effective_event_masses,
        event_masses_defaulted,
        None if adaptive_settings is not None else effective_analysis_masses,
        False if adaptive_settings is not None else analysis_masses_defaulted,
        pending_limit,
        trials,
        seed,
        (
            "SeedSequence(root seed).spawn in scenario order; adaptive simulation "
            "derives separate outcome and sampler child seeds."
        ),
        adaptive_settings,
        aggregate_work_budget,
        boundaries,
        tuple(summaries),
    )
