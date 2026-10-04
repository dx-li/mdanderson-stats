"""Bounded, reproducible BOIN design reports with simulated operating characteristics."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .boin import BOINDesign, _owned
from .boin_protocol import boin_protocol
from .boin_simulation import simulate_boin
from .dose_allocation_risk import DoseAllocationRisks, dose_allocation_risks

_MAX_SCENARIOS = 20
_MAX_SIMULATION_CELLS = 1_000_000
_MAX_TOTAL_REPLICATIONS = 2_000_000


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 120:
        raise ValueError(f"{name} must be a nonempty string of at most 120 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _setting(value: float | int | bool | None) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    return _number(value)


def _positive_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not 1 <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(parsed)


def _snapshot_design(design: BOINDesign) -> BOINDesign:
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    snapshot = BOINDesign(
        target=design.target,
        safe_probability=design.safe_probability,
        toxic_probability=design.toxic_probability,
        elimination_probability=design.elimination_probability,
        extra_safe=design.extra_safe,
        safety_offset=design.safety_offset,
        stay_at_one_of_three=design.stay_at_one_of_three,
        deescalate_at_two_of_six=design.deescalate_at_two_of_six,
        bound_mtd=design.bound_mtd,
        early_stop_patients=design.early_stop_patients,
    )
    for name in ("escalation_boundary", "deescalation_boundary"):
        requested = float(getattr(design, name))
        reconstructed = float(getattr(snapshot, name))
        tolerance = 64 * np.finfo(float).eps * abs(requested)
        if not np.isfinite(requested) or abs(requested - reconstructed) > tolerance:
            raise ValueError(f"design.{name} is inconsistent with its captured probabilities")
        object.__setattr__(snapshot, name, requested)
    return snapshot


@dataclass(frozen=True, slots=True)
class BOINReportScenario:
    """One truth profile for a bounded BOIN operating-characteristic report."""

    label: str
    true_toxicity: ArrayLike
    _probabilities: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        label = _label(self.label, "scenario label")
        raw = self.true_toxicity
        if isinstance(raw, np.ndarray):
            if raw.ndim != 1 or np.iscomplexobj(raw) or raw.dtype.kind == "b":
                raise ValueError("true_toxicity must be a real one-dimensional vector")
            if not 2 <= raw.size <= 100:
                raise ValueError("true_toxicity must contain 2..100 probabilities")
        elif isinstance(raw, (list, tuple)):
            if not 2 <= len(raw) <= 100 or any(
                isinstance(
                    item,
                    (list, tuple, np.ndarray, complex, np.complexfloating, bool, np.bool_),
                )
                for item in raw
            ):
                raise ValueError("true_toxicity must be a real one-dimensional vector")
        else:
            raise ValueError("true_toxicity must be a bounded one-dimensional vector")
        probabilities = finite(raw, "true_toxicity")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("true_toxicity entries must lie in [0,1]")
        owned = _owned(probabilities)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "true_toxicity", owned)
        object.__setattr__(self, "_probabilities", tuple(float(p) for p in owned))


@dataclass(frozen=True, slots=True)
class BOINScenarioSummary:
    """Compact immutable operating-characteristic summary for one truth profile."""

    label: str
    true_toxicity: tuple[float, ...]
    selection_probability: tuple[float, ...]
    selection_mcse: tuple[float, ...]
    mean_patients: tuple[float, ...]
    mean_toxicities: tuple[float, ...]
    stop_frequency: tuple[tuple[str, float], ...]
    titration_end_frequency: tuple[tuple[str, float], ...]
    mean_titration_patients: float
    mean_titration_moderate_toxicities: tuple[float, ...]
    allocation_risk: DoseAllocationRisks


@dataclass(frozen=True, slots=True)
class BOINDesignReport:
    """A static snapshot of one BOIN design and its simulated scenario summaries."""

    design: BOINDesign
    scenarios: tuple[BOINScenarioSummary, ...]
    cohorts: int
    cohort_size: int
    trials: int
    start_dose: int
    titration: bool
    titration_cap: int | None
    moderate_toxicity: tuple[float, ...]
    seed: int
    language: Literal["en", "zh"]
    protocol: str

    def to_html(self) -> str:
        """Render the full report as escaped, self-contained UTF-8 HTML."""
        design = self.design
        parameters = (
            ("Target DLT probability", design.target),
            ("Safe indifference probability", design.safe_probability),
            ("Toxic indifference probability", design.toxic_probability),
            ("Escalation boundary", design.escalation_boundary),
            ("De-escalation boundary", design.deescalation_boundary),
            ("Safety elimination cutoff", design.elimination_probability),
            ("Extra safety enabled", design.extra_safe),
            ("Extra-safety offset", design.safety_offset),
            ("Stay at 1/3 modification", design.stay_at_one_of_three),
            ("De-escalate at 2/6 modification", design.deescalate_at_two_of_six),
            ("Bound MTD", design.bound_mtd),
            ("Early precision-stop patients", design.early_stop_patients),
            ("Cohorts", self.cohorts),
            ("Cohort size", self.cohort_size),
            ("Maximum planned enrollment", self.cohorts * self.cohort_size),
            ("Trials per scenario", self.trials),
            ("Starting dose", self.start_dose),
            ("Accelerated titration", self.titration),
            ("Titration dose cap", self.titration_cap),
            ("NumPy seed", self.seed),
        )
        parameter_rows = "".join(
            f'<tr><th scope="row">{escape(label)}</th><td>{escape(_setting(value))}</td></tr>'
            for label, value in parameters
        )
        titration_rows = "".join(
            f'<tr><th scope="row">Dose {dose}</th><td>{_number(value)}</td></tr>'
            for dose, value in enumerate(self.moderate_toxicity, start=1)
        )
        scenario_sections: list[str] = []
        for scenario in self.scenarios:
            truth_rows = "".join(
                f"<tr><td>{dose}</td><td>{_number(probability)}</td>"
                f"<td>{_number(scenario.mean_patients[dose - 1])}</td>"
                f"<td>{_number(scenario.mean_toxicities[dose - 1])}</td></tr>"
                for dose, probability in enumerate(scenario.true_toxicity, start=1)
            )
            selection_rows = "".join(
                f"<tr><td>{'No MTD' if dose == 0 else f'Dose {dose}'}</td>"
                f"<td>{_number(probability)}</td><td>{_number(scenario.selection_mcse[dose])}</td></tr>"
                for dose, probability in enumerate(scenario.selection_probability)
            )
            stop_rows = "".join(
                f"<tr><td>{escape(reason)}</td><td>{_number(frequency)}</td></tr>"
                for reason, frequency in scenario.stop_frequency
            )
            titration_stop_rows = "".join(
                f"<tr><td>{escape(reason)}</td><td>{_number(frequency)}</td></tr>"
                for reason, frequency in scenario.titration_end_frequency
            )
            risk = scenario.allocation_risk
            if risk.has_exact_target:
                risk_rows = (
                    f"<tr><td>More than 60% of planned enrollment above target</td>"
                    f"<td>{risk.overdose60_trials}/{risk.trials}</td>"
                    f"<td>{_number(risk.overdose60_probability)}</td>"
                    f"<td>{_number(risk.overdose60_mcse)}</td></tr>"
                    f"<tr><td>More than 80% of planned enrollment above target</td>"
                    f"<td>{risk.overdose80_trials}/{risk.trials}</td>"
                    f"<td>{_number(risk.overdose80_probability)}</td>"
                    f"<td>{_number(risk.overdose80_mcse)}</td></tr>"
                )
            else:
                reason = escape(risk.unavailable_reason or "Unavailable")
                risk_rows = f'<tr><td colspan="4">{reason}</td></tr>'
            moderate_rows = "".join(
                f"<tr><td>{dose}</td><td>{_number(value)}</td></tr>"
                for dose, value in enumerate(scenario.mean_titration_moderate_toxicities, start=1)
            )
            scenario_sections.append(
                f"<section><h2>{escape(scenario.label)}</h2>"
                f"<table><caption>Truth and mean enrollment</caption>"
                "<thead><tr><th>Dose</th><th>True DLT probability</th>"
                "<th>Mean patients</th><th>Mean DLTs</th></tr></thead>"
                f"<tbody>{truth_rows}</tbody></table>"
                f"<table><caption>Final MTD selection</caption>"
                "<thead><tr><th>Selection</th><th>Probability</th><th>MCSE</th></tr></thead>"
                f"<tbody>{selection_rows}</tbody></table>"
                f"<table><caption>Trial stop reasons</caption>"
                "<thead><tr><th>Reason</th><th>Frequency</th></tr></thead>"
                f"<tbody>{stop_rows}</tbody></table>"
                f"<table><caption>Over-target allocation risk</caption>"
                "<thead><tr><th>Event</th><th>Trials</th><th>Probability</th><th>MCSE</th></tr></thead>"
                f"<tbody>{risk_rows}</tbody></table>"
                f"<p>Mean titration patients: {_number(scenario.mean_titration_patients)}</p>"
                f"<table><caption>Mean grade-2 events during titration</caption>"
                "<thead><tr><th>Dose</th><th>Mean events</th></tr></thead>"
                f"<tbody>{moderate_rows}</tbody></table>"
                f"<table><caption>Titration end reasons</caption>"
                "<thead><tr><th>Reason</th><th>Frequency</th></tr></thead>"
                f"<tbody>{titration_stop_rows}</tbody></table></section>"
            )
        lang = self.language
        protocol_block = escape(self.protocol)
        return (
            "<!doctype html>\n"
            f'<html lang="{lang}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>BOIN design report</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em;"
            "width:100%}th,td{border:1px solid #b8bec5;padding:.45em .65em;text-align:left}"
            "caption{text-align:left;font-weight:700;margin-bottom:.5em}"
            "section{border-top:2px solid #ddd;margin-top:2em}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere;background:#f5f6f7;padding:1em}</style></head><body>\n"
            "<h1>BOIN single-agent design report</h1>"
            "<p>This static report records the design inputs and operating characteristics "
            "calculated by the independent Python implementation. It does not reproduce "
            "the native application's animation or its random-number stream.</p>"
            "<table><caption>Design and simulation settings</caption><tbody>"
            f"{parameter_rows}</tbody></table>"
            "<table><caption>Grade-2 probabilities used during accelerated titration</caption>"
            f"<tbody>{titration_rows}</tbody></table>"
            f"{''.join(scenario_sections)}"
            "<details><summary>Statistical methods and decision table</summary>"
            f"<pre>{protocol_block}</pre></details>"
            "<p>Selection probabilities include a no-MTD result followed by dose indices 1..J. "
            "Monte Carlo standard errors use the ordinary Bernoulli formula. Over-target "
            "allocation probabilities are fractions; the source application displays percentages. "
            "The HTML contains only summary values, not trial-level simulation arrays.</p>"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically write the rendered report as UTF-8 HTML."""
        destination = Path(path)
        content = self.to_html()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except Exception:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            raise
        return destination


def boin_design_report(
    design: BOINDesign,
    scenarios: tuple[BOINReportScenario, ...],
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1_000,
    start_dose: int = 1,
    titration: bool = False,
    titration_cap: int | None = None,
    moderate_toxicity: ArrayLike | None = None,
    seed: int,
    language: Literal["en", "zh"] = "en",
) -> BOINDesignReport:
    """Calculate a bounded serial scenario report from one captured BOIN design."""
    snapshot = _snapshot_design(design)
    if not isinstance(scenarios, tuple) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must be a tuple containing 1..{_MAX_SCENARIOS} values")
    if any(not isinstance(item, BOINReportScenario) for item in scenarios):
        raise ValueError("every scenario must be a BOINReportScenario")
    dose_count = len(scenarios[0]._probabilities)
    if any(len(item._probabilities) != dose_count for item in scenarios):
        raise ValueError("all scenarios must have the same number of dose levels")
    labels = [item.label for item in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError("scenario labels must be unique")

    n_cohorts = _positive_integer(cohorts, "cohorts", 200)
    size = _positive_integer(cohort_size, "cohort_size", 200)
    n_trials = _positive_integer(trials, "trials", 1_000_000)
    start = _positive_integer(start_dose, "start_dose", dose_count)
    max_patients = n_cohorts * size
    if max_patients > 200:
        raise ValueError("report design must plan at most 200 patients")
    if start > dose_count:
        raise ValueError("start_dose must not exceed the number of doses")
    if n_trials * dose_count > _MAX_SIMULATION_CELLS:
        raise ValueError("each scenario may use at most 1000000 trial-dose cells")
    if n_trials * max_patients * len(scenarios) > _MAX_TOTAL_REPLICATIONS:
        raise ValueError("report may contain at most 2000000 planned patient-replications")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer in [0,2**32-1]")
    seed_value = scalar(seed, "seed")
    if seed_value != int(seed_value) or not 0 <= seed_value <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0,2**32-1]")
    if not isinstance(titration, (bool, np.bool_)):
        raise ValueError("titration must be boolean")
    if language not in ("en", "zh"):
        raise ValueError("language must be 'en' or 'zh'")
    cap = dose_count if bool(titration) and titration_cap is None else None
    if titration_cap is not None:
        cap = _positive_integer(titration_cap, "titration_cap", dose_count)
        if cap < start:
            raise ValueError("titration_cap must be between start_dose and the highest dose")
    if not titration and (titration_cap is not None or moderate_toxicity is not None):
        raise ValueError("titration options require titration=True")

    moderate = np.zeros(dose_count, dtype=np.float64)
    if moderate_toxicity is not None:
        if isinstance(moderate_toxicity, np.ndarray):
            if (
                moderate_toxicity.ndim != 1
                or moderate_toxicity.shape != (dose_count,)
                or np.iscomplexobj(moderate_toxicity)
                or moderate_toxicity.dtype.kind == "b"
            ):
                raise ValueError("moderate_toxicity must be a one-dimensional probability vector")
        elif isinstance(moderate_toxicity, (list, tuple)):
            if len(moderate_toxicity) != dose_count or any(
                isinstance(
                    item,
                    (list, tuple, np.ndarray, complex, np.complexfloating, bool, np.bool_),
                )
                for item in moderate_toxicity
            ):
                raise ValueError("moderate_toxicity must match the dose count")
        else:
            raise ValueError("moderate_toxicity must be a bounded probability vector")
        moderate = _owned(finite(moderate_toxicity, "moderate_toxicity"))
        if moderate.shape != (dose_count,) or np.any((moderate < 0) | (moderate > 1)):
            raise ValueError("moderate_toxicity must match doses and lie in [0,1]")
    for scenario in scenarios:
        if np.any(moderate + np.asarray(scenario._probabilities) > 1):
            raise ValueError(
                "moderate_toxicity must be mutually exclusive with DLT in every scenario"
            )

    protocol = boin_protocol(
        snapshot,
        doses=dose_count,
        cohorts=n_cohorts,
        cohort_size=size,
        start_dose=start,
        titration=bool(titration),
        titration_cap=cap,
        language=language,
    )
    generator = np.random.default_rng(int(seed_value))
    summaries: list[BOINScenarioSummary] = []
    for scenario in scenarios:
        truth = np.asarray(scenario._probabilities, dtype=np.float64)
        simulation = simulate_boin(
            snapshot,
            truth,
            cohorts=n_cohorts,
            cohort_size=size,
            trials=n_trials,
            start_dose=start,
            titration=bool(titration),
            titration_cap=cap,
            moderate_toxicity=moderate if titration else None,
            rng=generator,
        )
        stop_reasons = Counter(simulation.stop_reason)
        titration_reasons = Counter(simulation.titration_end_reason)
        risk = dose_allocation_risks(
            simulation.patients,
            truth,
            target=snapshot.target,
            planned_patients=max_patients,
        )
        summaries.append(
            BOINScenarioSummary(
                scenario.label,
                tuple(float(value) for value in truth),
                tuple(float(value) for value in simulation.selection_probability),
                tuple(float(value) for value in simulation.selection_mcse),
                tuple(float(value) for value in simulation.mean_patients),
                tuple(float(value) for value in simulation.mean_toxicities),
                tuple(
                    (reason, stop_reasons[reason] / n_trials)
                    for reason in ("stop_safety", "stop_precision", "max_patients")
                ),
                tuple(
                    (reason, titration_reasons[reason] / n_trials)
                    for reason in (
                        "disabled",
                        "DLT",
                        "grade2",
                        "highest_dose",
                        "dose_cap",
                        "max_patients",
                    )
                ),
                float(simulation.titration_patients.mean()),
                tuple(
                    float(value) for value in simulation.titration_moderate_toxicities.mean(axis=0)
                ),
                risk,
            )
        )
        del simulation

    return BOINDesignReport(
        snapshot,
        tuple(summaries),
        n_cohorts,
        size,
        n_trials,
        start,
        bool(titration),
        cap,
        tuple(float(value) for value in moderate),
        int(seed_value),
        language,
        protocol,
    )
