"""Reproducible multi-scenario simulations and compact HTML reports for OneArmTTE."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

from ._validation import scalar
from .one_arm_tte import OneArmTTEDesign, one_arm_tte_design
from .one_arm_tte_simulation import OneArmTTESimulation, simulate_one_arm_tte

_MAX_SCENARIOS = 20
_MAX_TOTAL_PATIENT_DRAWS = 100_000
_MAX_TOTAL_MONITORING_CHECKS = 100_000
_MAX_LABEL_LENGTH = 160
_MAX_TIME_UNIT_LENGTH = 40


@dataclass(frozen=True)
class OneArmTTEScenario:
    """One assumed exponential TTE, Poisson accrual rate, and reproducible seed."""

    name: str
    true_tte: float
    accrual_rate: float
    seed: int


@dataclass(frozen=True)
class OneArmTTEScenarioSummary:
    """Compact immutable summary; per-replication simulation arrays are discarded."""

    name: str
    true_tte: float
    accrual_rate: float
    seed: int
    repetitions: int
    total_monitoring_checks: int
    early_inferior_count: int
    early_superior_count: int
    final_inferior_count: int
    final_superior_count: int
    early_inferior_probability: float
    early_superior_probability: float
    final_inferior_probability: float
    final_superior_probability: float
    early_inferior_mcse: float
    early_superior_mcse: float
    final_inferior_mcse: float
    final_superior_mcse: float
    mean_sample_size: float
    mean_events: float
    mean_exposure: float
    mean_accrual_stop_time: float
    mean_final_time: float
    sample_size_quantiles: tuple[float, float, float]
    duration_quantiles: tuple[float, float, float]


@dataclass(frozen=True)
class OneArmTTEScenarioReport:
    """Validated design snapshot and scenario summaries in the supplied order."""

    design: OneArmTTEDesign
    repetitions: int
    credible_level: float
    time_unit: str
    max_total_monitoring_checks: int
    scenarios: tuple[OneArmTTEScenarioSummary, ...]

    def to_html(self) -> str:
        """Render a standalone escaped report; no simulation is performed here."""
        design_rows = _design_rows(self.design)
        summary_rows = "".join(_summary_row(item, self.time_unit) for item in self.scenarios)
        sections = "".join(_scenario_section(item, self.time_unit) for item in self.scenarios)
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>One-arm time-to-event scenario report</title>"
            "<style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
            "padding:0 1rem;color:#202124}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}"
            "th,td{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}"
            "th{background:#f1f3f4}"
            "section{border-top:1px solid #bbb;padding-top:1rem;margin-top:2rem}"
            "code{white-space:pre-wrap}</style></head><body>"
            "<h1>One-arm time-to-event scenario report</h1>"
            f"<p>{self.repetitions} repetitions per scenario; central quantiles at level "
            f"{self.credible_level:.17g}. Scenario seeds reset independently. "
            f"All time quantities use {escape(self.time_unit)}. "
            f"Aggregate accrual-phase check ceiling: {self.max_total_monitoring_checks}.</p>"
            "<h2>Design inputs</h2><table><tbody>"
            f"{design_rows}</tbody></table>"
            "<h2>Scenario summary</h2><table><thead><tr>"
            "<th>Scenario</th><th>True TTE</th><th>Accrual rate</th>"
            "<th>Early inferior</th><th>Early superior</th>"
            "<th>Final inferior</th><th>Final superior</th>"
            "<th>Mean patients</th><th>Mean duration</th></tr></thead><tbody>"
            f"{summary_rows}</tbody></table>{sections}</body></html>"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically write this report as UTF-8 HTML."""
        destination = Path(path)
        if not destination.name:
            raise ValueError("path must name an output file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_html()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
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


def simulate_one_arm_tte_scenarios(
    design: OneArmTTEDesign,
    scenarios: Sequence[OneArmTTEScenario],
    repetitions: int,
    *,
    credible_level: float = 0.95,
    time_unit: str = "time units",
    max_total_monitoring_checks: int = _MAX_TOTAL_MONITORING_CHECKS,
) -> OneArmTTEScenarioReport:
    """Run bounded, serial scenario simulations and retain compact summaries.

    Every scenario resets NumPy's generator with its own required positive seed.
    All scenarios, the aggregate patient limit, and the requested aggregate
    monitoring-check ceiling are validated before the first simulation starts.
    The actual stochastic check count is enforced cumulatively across scenarios.
    """
    snapshot = _capture_design(design)
    if not isinstance(scenarios, (tuple, list)) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} entries")
    reps_value = scalar(repetitions, "repetitions")
    if isinstance(repetitions, (bool, np.bool_)) or int(reps_value) != reps_value or reps_value < 1:
        raise ValueError("repetitions must be a positive integer")
    reps = int(reps_value)
    if isinstance(max_total_monitoring_checks, (bool, np.bool_)):
        raise ValueError("max_total_monitoring_checks must be an integer in [0,100000]")
    check_budget_value = scalar(max_total_monitoring_checks, "max_total_monitoring_checks")
    if (
        int(check_budget_value) != check_budget_value
        or not 0 <= check_budget_value <= _MAX_TOTAL_MONITORING_CHECKS
    ):
        raise ValueError("max_total_monitoring_checks must be an integer in [0,100000]")
    check_budget = int(check_budget_value)
    level = scalar(credible_level, "credible_level")
    if not 0 < level < 1:
        raise ValueError("credible_level must be in (0,1)")
    unit = _text_label(time_unit, "time_unit", _MAX_TIME_UNIT_LENGTH)

    captured: list[OneArmTTEScenario] = []
    names: set[str] = set()
    for index, item in enumerate(scenarios):
        if not isinstance(item, OneArmTTEScenario):
            raise TypeError(f"scenarios[{index}] must be a OneArmTTEScenario")
        name = _text_label(item.name, f"scenarios[{index}].name", _MAX_LABEL_LENGTH)
        if name.casefold() in names:
            raise ValueError("scenario names must be unique ignoring case")
        names.add(name.casefold())
        tte = scalar(item.true_tte, f"scenarios[{index}].true_tte")
        rate = scalar(item.accrual_rate, f"scenarios[{index}].accrual_rate")
        if tte <= 0 or rate <= 0:
            raise ValueError("true_tte and accrual_rate must be positive")
        if isinstance(item.seed, (bool, np.bool_)) or not isinstance(item.seed, (int, np.integer)):
            raise ValueError("scenario seed must be a positive unsigned 64-bit integer")
        seed_value = int(item.seed)
        if seed_value <= 0 or seed_value > np.iinfo(np.uint64).max:
            raise ValueError("scenario seed must be a positive unsigned 64-bit integer")
        interarrival_mean = 1.0 / rate
        duration_mean = tte if snapshot.parameterization == "mean" else tte / np.log(2.0)
        if not np.isfinite(interarrival_mean) or interarrival_mean <= 0:
            raise ArithmeticError(f"scenarios[{index}] accrual scale is not representable")
        if not np.isfinite(duration_mean) or duration_mean <= 0:
            raise ArithmeticError(f"scenarios[{index}] event-duration scale is not representable")
        captured.append(OneArmTTEScenario(name, float(tte), float(rate), int(seed_value)))

    patient_work = reps * snapshot.max_patients * len(captured)
    if patient_work > _MAX_TOTAL_PATIENT_DRAWS:
        raise ValueError(f"aggregate potential patients cannot exceed {_MAX_TOTAL_PATIENT_DRAWS}")
    summaries: list[OneArmTTEScenarioSummary] = []
    remaining_checks = check_budget
    for item in captured:
        simulation = simulate_one_arm_tte(
            snapshot,
            item.true_tte,
            item.accrual_rate,
            reps,
            seed=item.seed,
            credible_level=float(level),
            max_monitoring_checks=remaining_checks,
        )
        summaries.append(_summarize(item, simulation))
        remaining_checks -= simulation.total_monitoring_checks
        del simulation
    return OneArmTTEScenarioReport(
        snapshot, reps, float(level), unit, check_budget, tuple(summaries)
    )


def _capture_design(design: OneArmTTEDesign) -> OneArmTTEDesign:
    if not isinstance(design, OneArmTTEDesign):
        raise TypeError("design must be a OneArmTTEDesign")
    return one_arm_tte_design(
        design.standard_prior,
        design.experimental_prior,
        parameterization=design.parameterization,
        maximize=design.maximize,
        delta_inferiority=design.delta_inferiority,
        cutoff_inferiority=design.cutoff_inferiority,
        delta_superiority=design.delta_superiority,
        cutoff_superiority=design.cutoff_superiority,
        max_patients=design.max_patients,
        minimum_patients=design.minimum_patients,
        periodic_interval=design.periodic_interval,
        monitor_at_accrual=design.monitor_at_accrual,
        followup_period=design.followup_period,
        absolute_tolerance=design.absolute_tolerance,
    )


def _text_label(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    text = value
    if not text.strip() or len(text) > limit or any(ord(char) < 32 for char in text):
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return text


def _stable_mean(values: np.ndarray) -> float:
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum == 0:
        return 0.0
    result = maximum * float(np.mean(values / maximum))
    if not np.isfinite(result):
        raise ArithmeticError("a scenario summary mean is not representable")
    return result


def _summarize(
    scenario: OneArmTTEScenario, result: OneArmTTESimulation
) -> OneArmTTEScenarioSummary:
    return OneArmTTEScenarioSummary(
        scenario.name,
        scenario.true_tte,
        scenario.accrual_rate,
        scenario.seed,
        result.repetitions,
        result.total_monitoring_checks,
        int(np.count_nonzero(result.early_inferior)),
        int(np.count_nonzero(result.early_superior)),
        int(np.count_nonzero(result.final_inferior)),
        int(np.count_nonzero(result.final_superior)),
        result.early_inferior_probability,
        result.early_superior_probability,
        result.final_inferior_probability,
        result.final_superior_probability,
        result.early_inferior_mcse,
        result.early_superior_mcse,
        result.final_inferior_mcse,
        result.final_superior_mcse,
        _stable_mean(result.sample_sizes),
        _stable_mean(result.events),
        _stable_mean(result.exposures),
        _stable_mean(result.accrual_stop_times),
        _stable_mean(result.final_times),
        _three_values(result.sample_size_quantiles),
        _three_values(result.duration_quantiles),
    )


def _three_values(values: np.ndarray) -> tuple[float, float, float]:
    if values.shape != (3,):
        raise ValueError("simulation quantiles must contain lower, median, and upper values")
    return float(values[0]), float(values[1]), float(values[2])


def _design_rows(design: OneArmTTEDesign) -> str:
    settings = (
        ("Parameterization", design.parameterization),
        ("Goal", "maximize" if design.maximize else "minimize"),
        ("Standard prior (shape, scale)", tuple(float(x) for x in design.standard_prior)),
        ("Experimental prior (shape, scale)", tuple(float(x) for x in design.experimental_prior)),
        ("Inferiority margin", design.delta_inferiority),
        ("Inferiority cutoff", design.cutoff_inferiority),
        ("Superiority margin", design.delta_superiority),
        ("Superiority cutoff", design.cutoff_superiority),
        ("Minimum patients", design.minimum_patients),
        ("Maximum patients", design.max_patients),
        ("Periodic monitoring interval", design.periodic_interval),
        ("Monitor before accrual", design.monitor_at_accrual),
        ("Post-accrual follow-up", design.followup_period),
        ("Absolute probability tolerance", design.absolute_tolerance),
    )
    return "".join(
        f'<tr><th scope="row">{escape(label)}</th><td>{escape(_display(value))}</td></tr>'
        for label, value in settings
    )


def _display(value: object) -> str:
    if value is None:
        return "Disabled / none"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, tuple):
        return "(" + ", ".join(f"{item:.17g}" for item in value) + ")"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    return str(value)


def _percent(probability: float, mcse: float) -> str:
    return f"{100 * probability:.6g}% (MCSE {100 * mcse:.4g} percentage points)"


def _summary_row(item: OneArmTTEScenarioSummary, time_unit: str) -> str:
    fields = (
        f'<th scope="row">{escape(item.name)}</th>',
        f"<td>{item.true_tte:.8g} {escape(time_unit)}</td>",
        f"<td>{item.accrual_rate:.8g} per {escape(time_unit)}</td>",
        f"<td>{_percent(item.early_inferior_probability, item.early_inferior_mcse)}</td>",
        f"<td>{_percent(item.early_superior_probability, item.early_superior_mcse)}</td>",
        f"<td>{_percent(item.final_inferior_probability, item.final_inferior_mcse)}</td>",
        f"<td>{_percent(item.final_superior_probability, item.final_superior_mcse)}</td>",
        f"<td>{item.mean_sample_size:.6g}</td>",
        f"<td>{item.mean_final_time:.6g} {escape(time_unit)}</td>",
    )
    return "<tr>" + "".join(fields) + "</tr>"


def _scenario_section(item: OneArmTTEScenarioSummary, time_unit: str) -> str:
    metrics = (
        ("Repetitions", str(item.repetitions)),
        ("Accrual-phase monitoring checks", str(item.total_monitoring_checks)),
        ("Scenario RNG seed", str(item.seed)),
        ("True mean/median TTE", f"{item.true_tte:.17g} {time_unit}"),
        ("Accrual rate", f"{item.accrual_rate:.17g} patients per {time_unit}"),
        (
            "Early inferiority",
            f"{_percent(item.early_inferior_probability, item.early_inferior_mcse)}; "
            f"{item.early_inferior_count} trials",
        ),
        (
            "Early superiority",
            f"{_percent(item.early_superior_probability, item.early_superior_mcse)}; "
            f"{item.early_superior_count} trials",
        ),
        (
            "Final inferiority",
            f"{_percent(item.final_inferior_probability, item.final_inferior_mcse)}; "
            f"{item.final_inferior_count} trials",
        ),
        (
            "Final superiority",
            f"{_percent(item.final_superior_probability, item.final_superior_mcse)}; "
            f"{item.final_superior_count} trials",
        ),
        ("Mean patients", f"{item.mean_sample_size:.17g}"),
        ("Mean events", f"{item.mean_events:.17g}"),
        ("Mean observed exposure", f"{item.mean_exposure:.17g} {time_unit}"),
        ("Mean accrual-stop time", f"{item.mean_accrual_stop_time:.17g} {time_unit}"),
        ("Mean final time", f"{item.mean_final_time:.17g} {time_unit}"),
        ("Sample-size quantiles (lower, median, upper)", _display(item.sample_size_quantiles)),
        (
            "Final-time quantiles (lower, median, upper)",
            f"{_display(item.duration_quantiles)} {time_unit}",
        ),
    )
    rows = "".join(
        f'<tr><th scope="row">{escape(label)}</th><td>{escape(value)}</td></tr>'
        for label, value in metrics
    )
    return f"<section><h2>{escape(item.name)}</h2><table><tbody>{rows}</tbody></table></section>"
