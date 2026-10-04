"""Self-contained HTML protocol summaries for single-agent Keyboard designs."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .boin import BOINBoundaryTable
from .dose_allocation_risk import dose_allocation_risks
from .keyboard import KeyboardDesign
from .keyboard_simulation import KeyboardSimulation, simulate_keyboard

_MAX_SCENARIOS = 20
_MAX_SCENARIO_TRIAL_DOSE_CELLS = 1_000_000
_MAX_TOTAL_PATIENT_REPLICATIONS = 2_000_000
_STOP_REASONS = ("max_patients", "stop_safety", "stop_precision")


def _label(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _scenario_array(value: ArrayLike) -> np.ndarray:
    """Preflight a small scenario matrix before making its float64 snapshot."""
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError("true_toxicity_scenarios must be a 2D matrix")
        if value.dtype.kind not in "biuf":
            raise ValueError("scenario probabilities must be real numeric values")
        rows, doses = value.shape
        if not 1 <= rows <= _MAX_SCENARIOS or not 2 <= doses <= 20:
            raise ValueError("require 1..20 scenarios and 2..20 doses")
        if value.size > _MAX_SCENARIOS * 20:
            raise ValueError("scenario matrix exceeds the input limit")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_SCENARIOS:
            raise ValueError("require 1..20 scenarios")
        first = value[0]
        if isinstance(first, np.ndarray):
            if first.ndim != 1:
                raise ValueError("each scenario must be a one-dimensional vector")
            doses = first.size
        elif isinstance(first, (list, tuple)):
            doses = len(first)
        else:
            raise ValueError("each scenario must be a one-dimensional vector")
        if not 2 <= doses <= 20:
            raise ValueError("require 2..20 doses")
        for row in value:
            entries: Sequence[object] | np.ndarray
            if isinstance(row, np.ndarray):
                if row.ndim != 1 or row.size != doses:
                    raise ValueError("all scenarios must have the same dose count")
                entries = row
            elif isinstance(row, (list, tuple)):
                if len(row) != doses:
                    raise ValueError("all scenarios must have the same dose count")
                entries = row
            else:
                raise ValueError("each scenario must be a one-dimensional vector")
            if any(
                isinstance(item, (list, tuple, np.ndarray))
                or np.ndim(item) != 0
                or np.iscomplexobj(item)
                for item in entries
            ):
                raise ValueError("scenario probabilities must be scalar values")
    else:
        raise TypeError("true_toxicity_scenarios must be a 2D array or nested sequence")

    try:
        scenarios = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scenario probabilities must be real numbers") from exc
    if scenarios.ndim != 2 or not np.all(np.isfinite(scenarios)):
        raise ValueError("scenario probabilities must be finite")
    if np.any((scenarios < 0) | (scenarios > 1)):
        raise ValueError("scenario probabilities must lie in [0,1]")
    scenarios.setflags(write=False)
    return scenarios


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    result = int(value)
    if not 0 <= result <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    return result


def _copy_design(design: KeyboardDesign) -> KeyboardDesign:
    if not isinstance(design, KeyboardDesign):
        raise TypeError("design must be a KeyboardDesign")
    return KeyboardDesign(
        target=design.target,
        lower=design.lower,
        upper=design.upper,
        edge_rule=design.edge_rule,
        elimination_probability=design.elimination_probability,
        extra_safe=design.extra_safe,
        safety_offset=design.safety_offset,
        early_stop_patients=design.early_stop_patients,
    )


@dataclass(frozen=True)
class KeyboardScenarioSummary:
    """Compact operating-characteristic snapshot for one true-risk scenario."""

    true_toxicity: tuple[float, ...]
    selection_probability: tuple[float, ...]
    selection_mcse: tuple[float, ...]
    mean_patients: tuple[float, ...]
    mean_toxicities: tuple[float, ...]
    stop_frequency: tuple[tuple[str, float], ...]
    overdose_allocation_probability: tuple[float, float] | None
    overdose_allocation_mcse: tuple[float, float] | None
    overdose_allocation_unavailable_reason: str | None


@dataclass(frozen=True)
class KeyboardProtocolReport:
    """Immutable protocol and operating-characteristic snapshot."""

    design: KeyboardDesign
    cohort_size: int
    cohorts: int
    start_dose: int
    trials: int
    seed: int
    boundaries: BOINBoundaryTable
    scenarios: tuple[KeyboardScenarioSummary, ...]
    title: str

    def to_html(self) -> str:
        """Render a self-contained protocol summary with captured inputs/results."""
        title = escape(_label(self.title, "title", 120))
        design = self.design
        dose_count = len(self.scenarios[0].true_toxicity)
        settings = (
            ("Target toxicity probability", design.target),
            ("Lower key boundary", design.lower),
            ("Upper key boundary", design.upper),
            ("Key edge rule", design.edge_rule),
            ("Safety elimination cutoff", design.elimination_probability),
            ("Extra-safe lowest-dose stop", design.extra_safe),
            ("Extra-safe offset", design.safety_offset),
            ("Early stop at current dose", design.early_stop_patients),
            ("Dose levels", dose_count),
            ("Cohort size", self.cohort_size),
            ("Maximum cohorts", self.cohorts),
            ("Starting dose (1-based)", self.start_dose),
            ("Maximum enrollment", self.cohort_size * self.cohorts),
            ("Simulation trials per scenario", self.trials),
            ("NumPy integer seed", self.seed),
        )
        settings_rows = "\n".join(
            f'<tr><th scope="row">{escape(str(name))}</th><td>{escape(str(value))}</td></tr>'
            for name, value in settings
        )
        boundary_rows = "\n".join(
            "<tr>"
            f"<td>{int(self.boundaries.patients[index])}</td>"
            f"<td>{int(self.boundaries.escalate_max[index])}</td>"
            f"<td>{int(self.boundaries.deescalate_min[index])}</td>"
            f"<td>{int(self.boundaries.eliminate_min[index])}</td>"
            f"<td>{int(self.boundaries.lowest_stop_min[index])}</td>"
            "</tr>"
            for index in range(len(self.boundaries.patients))
        )
        scenario_sections: list[str] = []
        for index, scenario in enumerate(self.scenarios, start=1):
            risk_cells = "".join(f"<td>{value:.17g}</td>" for value in scenario.true_toxicity)
            oc_rows = []
            labels = ("No MTD", *(f"Dose {dose}" for dose in range(1, dose_count + 1)))
            for label, probability, mcse in zip(
                labels, scenario.selection_probability, scenario.selection_mcse, strict=True
            ):
                oc_rows.append(
                    f'<tr><th scope="row">{escape(label)}</th>'
                    f"<td>{probability:.17g}</td><td>{mcse:.17g}</td></tr>"
                )
            patient_cells = "".join(f"<td>{value:.17g}</td>" for value in scenario.mean_patients)
            toxicity_cells = "".join(f"<td>{value:.17g}</td>" for value in scenario.mean_toxicities)
            stop_rows = "".join(
                f'<tr><th scope="row">{escape(reason)}</th><td>{frequency:.17g}</td></tr>'
                for reason, frequency in scenario.stop_frequency
            )
            if (
                scenario.overdose_allocation_probability is None
                or scenario.overdose_allocation_mcse is None
            ):
                reason = scenario.overdose_allocation_unavailable_reason or "Not available"
                overdose_rows = (
                    f'<tr><td colspan="3">Allocation risks unavailable: {escape(reason)}</td></tr>'
                )
            else:
                overdose_rows = "".join(
                    f'<tr><th scope="row">More than {fraction:.0%} of planned enrollment '
                    "above target</th>"
                    f"<td>{probability:.17g}</td><td>{mcse:.17g}</td></tr>"
                    for fraction, probability, mcse in zip(
                        (0.6, 0.8),
                        scenario.overdose_allocation_probability,
                        scenario.overdose_allocation_mcse,
                        strict=True,
                    )
                )
            scenario_sections.append(
                f"<section><h3>Scenario {index}</h3>"
                "<table><caption>True toxicity probability by dose</caption><tbody>"
                f"<tr>{risk_cells}</tr></tbody></table>"
                "<table><caption>MTD selection probability</caption>"
                "<thead><tr><th>Selection</th><th>Probability</th>"
                "<th>Monte Carlo SE</th></tr></thead>"
                f"<tbody>{''.join(oc_rows)}</tbody></table>"
                "<table><caption>Mean patients and DLTs by dose</caption>"
                "<thead><tr><th>Quantity</th>"
                + "".join(f"<th>Dose {dose}</th>" for dose in range(1, dose_count + 1))
                + f"</tr></thead><tbody><tr><th>Patients</th>{patient_cells}</tr>"
                f"<tr><th>DLTs</th>{toxicity_cells}</tr></tbody></table>"
                "<table><caption>Trial stopping frequency</caption>"
                f"<tbody>{stop_rows}</tbody></table>"
                "<table><caption>Overdose allocation risk</caption>"
                "<thead><tr><th>Event</th><th>Probability</th><th>Monte Carlo SE</th></tr></thead>"
                f"<tbody>{overdose_rows}</tbody></table></section>"
            )
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.45em .65em;text-align:left}"
            "caption{text-align:left;font-weight:700;margin-bottom:.5em}"
            "section{border-top:2px solid #ddd;padding-top:.5em}</style></head><body>\n"
            f"<h1>{title}</h1>\n"
            "<p>Python Keyboard protocol summary. This static report records the design, "
            "decision cutoffs and simulated operating characteristics; it is not the "
            "MD Anderson app's HTML/Word template.</p>\n"
            "<table><caption>Design and simulation inputs</caption>"
            f"<tbody>{settings_rows}</tbody></table>\n"
            "<h2>Conduct rules</h2><ul>"
            "<li>Ordinary Keyboard key decisions move one dose level up, stay, or move down.</li>"
            "<li>After at least three patients at a dose, safety elimination uses the "
            "uniform Beta(1,1) prior and eliminates that dose and all higher doses when "
            "the posterior overdose probability is strictly above its configured cutoff; "
            "an eliminated dose cannot be revisited.</li>"
            "<li>Optional extra-safe stopping applies at the lowest dose after the minimum "
            "patient count, using the cutoff minus the configured offset.</li>"
            "<li>Optional precision stopping occurs after the configured number at the "
            "current dose, unless safety requires de-escalation or stopping.</li>"
            "<li>Each simulated trial's MTD is estimated with weak Beta(.05,.05) priors "
            "and inverse-variance weighted isotonic regression; untreated doses are not "
            "estimated. Estimating the completed real trial remains a separate "
            "KeyboardDesign.select_mtd call.</li></ul>\n"
            "<table><caption>Inclusive decision cutoffs by treated patient count</caption>"
            "<thead><tr><th>Patients</th><th>Escalate if DLTs ≤</th>"
            "<th>De-escalate if DLTs ≥</th><th>Eliminate if DLTs ≥</th>"
            "<th>Extra-safe lowest-dose stop if enabled, DLTs ≥</th></tr></thead>"
            f"<tbody>{boundary_rows}</tbody></table>\n"
            "<p>In the cutoff table, −1 means escalation is impossible at that sample "
            "size; n+1 means the corresponding de-escalation, elimination, or lowest-dose "
            "stop cutoff cannot be reached with n patients.</p>\n"
            f"{''.join(scenario_sections)}\n"
            "<p>Each scenario used the same captured design and seed stream, processed in "
            "listed order with NumPy's generator. This is a Python RNG policy, not native "
            "R seed parity. Selection probabilities include a no-MTD category; reported "
            "Monte Carlo standard errors are plug-in binomial errors, not confidence "
            "intervals.</p>\n"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically write the rendered UTF-8 report."""
        destination = Path(path)
        content = self.to_html()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination


def keyboard_protocol_report(
    design: KeyboardDesign,
    true_toxicity_scenarios: ArrayLike,
    *,
    cohort_size: int,
    cohorts: int,
    start_dose: int = 1,
    trials: int = 1000,
    seed: int,
    title: str = "Keyboard trial protocol summary",
) -> KeyboardProtocolReport:
    """Compute a bounded, reproducible protocol and OC snapshot.

    Scenarios are rows and dose levels are columns. Scenario simulations share
    one NumPy generator and are run serially in row order. The report retains
    compact summaries, not trial-level arrays. ``seed`` is mandatory so the
    generated stream is recorded explicitly.
    """
    validated_design = _copy_design(design)
    scenario_matrix = _scenario_array(true_toxicity_scenarios)
    size = _positive_int(cohort_size, "cohort_size")
    cohort_count = _positive_int(cohorts, "cohorts")
    repetition_count = _positive_int(trials, "trials")
    start = _positive_int(start_dose, "start_dose")
    seed_value = _seed(seed)
    dose_count = scenario_matrix.shape[1]
    if start > dose_count:
        raise ValueError("start_dose must identify a dose in every scenario")
    max_patients = cohort_count * size
    if max_patients > 200:
        raise ValueError("cohort_size * cohorts must not exceed 200 patients")
    if repetition_count * dose_count > _MAX_SCENARIO_TRIAL_DOSE_CELLS:
        raise ValueError("trials times dose count exceeds the per-scenario work limit")
    aggregate_work = len(scenario_matrix) * repetition_count * max_patients
    if aggregate_work > _MAX_TOTAL_PATIENT_REPLICATIONS:
        raise ValueError("aggregate simulated patient replications exceed the report limit")
    title_value = _label(title, "title", 120)
    boundaries = validated_design.boundary_table(max_patients)
    generator = np.random.default_rng(seed_value)
    summaries: list[KeyboardScenarioSummary] = []
    for true_toxicity in scenario_matrix:
        result: KeyboardSimulation = simulate_keyboard(
            validated_design,
            true_toxicity,
            cohorts=cohort_count,
            cohort_size=size,
            trials=repetition_count,
            start_dose=start,
            rng=generator,
        )
        stop_counts = Counter(result.stop_reason)
        risk = dose_allocation_risks(
            result.patients,
            true_toxicity,
            target=validated_design.target,
            planned_patients=max_patients,
        )
        if risk.has_exact_target:
            assert risk.overdose60_probability is not None
            assert risk.overdose80_probability is not None
            assert risk.overdose60_mcse is not None
            assert risk.overdose80_mcse is not None
            allocation_probability = (
                risk.overdose60_probability,
                risk.overdose80_probability,
            )
            allocation_mcse = (risk.overdose60_mcse, risk.overdose80_mcse)
        else:
            allocation_probability = None
            allocation_mcse = None
        summaries.append(
            KeyboardScenarioSummary(
                tuple(float(value) for value in true_toxicity),
                tuple(float(value) for value in result.selection_probability),
                tuple(float(value) for value in result.selection_mcse),
                tuple(float(value) for value in result.mean_patients),
                tuple(float(value) for value in result.mean_toxicities),
                tuple((reason, stop_counts[reason] / repetition_count) for reason in _STOP_REASONS),
                allocation_probability,
                allocation_mcse,
                risk.unavailable_reason,
            )
        )
        del result
    return KeyboardProtocolReport(
        validated_design,
        size,
        cohort_count,
        start,
        repetition_count,
        seed_value,
        boundaries,
        tuple(summaries),
        title_value,
    )
