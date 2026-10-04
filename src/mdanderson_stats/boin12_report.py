"""Bounded reproducible reports for BOIN12 simulation workflows."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .boin import BOINDesign, _owned
from .boin12 import BOIN12Design
from .boin12_simulation import BOIN12Simulation, simulate_boin12
from .boin12_two_stage import BOIN12TwoStageSimulation, simulate_boin12_two_stage

_MAX_SCENARIOS = 20
_MAX_DOSES = 100
_MAX_TRIAL_DOSE_CELLS = 1_000_000
_MAX_PATIENT_REPLICATIONS = 2_000_000
_MAX_WORK = 2_000_000


def _label(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 120:
        raise ValueError("scenario label must be a nonempty string of at most 120 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("scenario label must not contain control characters")
    return value


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return result


def _number(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    return format(float(value), ".17g")


def _scenario_grid(value: ArrayLike) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1:] != (4,) or not 2 <= value.shape[0] <= _MAX_DOSES:
            raise ValueError("joint_probability must have shape (2..100 doses, 4 outcomes)")
        if value.dtype.kind not in "fiu":
            raise ValueError("joint_probability must contain real probabilities")
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= _MAX_DOSES or any(
            not isinstance(row, (tuple, list, np.ndarray))
            or len(row) != 4
            or any(
                isinstance(item, (tuple, list, np.ndarray, bool, np.bool_, complex)) for item in row
            )
            for row in value
        ):
            raise ValueError("joint_probability must have shape (2..100 doses, 4 outcomes)")
    else:
        raise ValueError("joint_probability must be a bounded two-dimensional probability table")
    grid = finite(value, "joint_probability")
    if grid.ndim != 2 or grid.shape[1] != 4 or np.any(grid < 0):
        raise ValueError("joint_probability must be nonnegative with four outcome columns")
    if np.any(np.abs(grid.sum(axis=1) - 1.0) > 1e-12):
        raise ValueError("each joint_probability row must sum to one")
    return tuple(tuple(float(x) for x in row) for row in grid)


@dataclass(frozen=True, slots=True)
class BOIN12ReportScenario:
    """Immutable dose-by-joint-outcome truth for one report scenario.

    Outcome columns follow BOIN12 order: no-toxicity/efficacy,
    no-toxicity/no-efficacy, toxicity/efficacy, toxicity/no-efficacy.
    """

    label: str
    joint_probability: ArrayLike
    _grid: tuple[tuple[float, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        label = _label(self.label)
        grid = _scenario_grid(self.joint_probability)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "joint_probability", _owned(grid))
        object.__setattr__(self, "_grid", grid)


@dataclass(frozen=True, slots=True)
class BOIN12ScenarioSummary:
    """Compact immutable operating-characteristic summary for one truth profile."""

    label: str
    joint_probability: tuple[tuple[float, ...], ...]
    true_toxicity: tuple[float, ...]
    true_efficacy: tuple[float, ...]
    obd_probability: tuple[float, ...]
    obd_mcse: tuple[float, ...]
    mtd_probability: tuple[float, ...]
    mtd_mcse: tuple[float, ...]
    mean_patients: tuple[float, ...]
    mean_toxicities: tuple[float, ...]
    mean_efficacies: tuple[float, ...]
    stop_reason_frequency: tuple[tuple[str, float], ...]
    transition_cohort_frequency: tuple[tuple[int, float], ...] | None
    mean_stage1_cohorts: float | None
    mean_stage2_cohorts: float | None


@dataclass(frozen=True, slots=True)
class BOIN12Report:
    """Saved design inputs and compact simulation summaries."""

    design: BOIN12Design
    scenarios: tuple[BOIN12ScenarioSummary, ...]
    cohorts: int
    cohort_size: int
    trials: int
    start_dose: int
    stage1_threshold: int | None
    seed: int

    def to_html(self) -> str:
        """Render a self-contained escaped HTML report."""
        d = self.design
        settings = (
            ("Toxicity / BOIN target", d.toxicity_limit),
            ("Efficacy futility limit", d.efficacy_limit),
            (
                "Outcome utilities (noT/E, noT/noE, T/E, T/noE)",
                ", ".join(map(_number, d.utilities)),
            ),
            ("Toxicity admissibility cutoff", d.toxicity_cutoff),
            ("Efficacy admissibility cutoff", d.efficacy_cutoff),
            ("Exploration patients", d.exploration_patients),
            ("Stay patients", d.stay_patients),
            ("Early stop patients", d.early_stop_patients),
            ("BOIN escalation boundary", d._boin.escalation_boundary),
            ("BOIN de-escalation boundary", d._boin.deescalation_boundary),
            ("Cohorts", self.cohorts),
            ("Cohort size", self.cohort_size),
            ("Maximum planned enrollment", self.cohorts * self.cohort_size),
            ("Trials per scenario", self.trials),
            ("Starting dose (one-based)", self.start_dose),
            ("Two-stage toxicity-only threshold", self.stage1_threshold),
            ("NumPy seed", self.seed),
        )
        settings_html = "".join(
            f"<tr><th>{escape(name)}</th><td>"
            f"{escape(_number(value) if not isinstance(value, str) else value)}</td></tr>"
            for name, value in settings
        )
        sections: list[str] = []
        for scenario in self.scenarios:
            truth_rows = "".join(
                f"<tr><td>{i}</td>"
                f"<td>{_number(row[0])}</td><td>{_number(row[1])}</td>"
                f"<td>{_number(row[2])}</td><td>{_number(row[3])}</td>"
                f"<td>{_number(row[2] + row[3])}</td>"
                f"<td>{_number(row[0] + row[2])}</td>"
                f"<td>{_number(row[2] + row[3] - self.design.toxicity_limit)}</td>"
                f"<td>{_number(row[0] + row[2] - self.design.efficacy_limit)}</td>"
                f"<td>{_number(np.dot(row, self.design.utilities))}</td>"
                f"<td>{_number(scenario.mean_patients[i - 1])}</td>"
                f"<td>{_number(scenario.mean_toxicities[i - 1])}</td>"
                f"<td>{_number(scenario.mean_efficacies[i - 1])}</td></tr>"
                for i, row in enumerate(scenario.joint_probability, 1)
            )
            selection_rows = "".join(
                f"<tr><td>{'No selection' if i == 0 else f'Dose {i}'}</td>"
                f"<td>{_number(scenario.obd_probability[i])}</td><td>{_number(scenario.obd_mcse[i])}</td>"
                f"<td>{_number(scenario.mtd_probability[i])}</td><td>{_number(scenario.mtd_mcse[i])}</td></tr>"
                for i in range(len(scenario.obd_probability))
            )
            stop_rows = "".join(
                f"<tr><td>{escape(reason)}</td><td>{_number(freq)}</td></tr>"
                for reason, freq in scenario.stop_reason_frequency
            )
            stage_rows = ""
            if scenario.transition_cohort_frequency is not None:
                stage_rows = (
                    "<h3>Stage transition</h3><table><thead><tr>"
                    "<th>Trigger cohort (0 = no transition)</th>"
                    "<th>Frequency</th></tr></thead><tbody>"
                    + "".join(
                        f"<tr><td>{cohort}</td><td>{_number(freq)}</td></tr>"
                        for cohort, freq in scenario.transition_cohort_frequency
                    )
                    + "</tbody></table>"
                    f"<p>Mean Stage 1 cohorts: {_number(scenario.mean_stage1_cohorts)}; "
                    f"mean Stage 2 cohorts: {_number(scenario.mean_stage2_cohorts)}.</p>"
                )
            sections.append(
                f"<section><h2>{escape(scenario.label)}</h2>"
                "<table><caption>Joint truth, margins, and mean outcomes</caption>"
                "<thead><tr><th>Dose</th><th>P(noT,E)</th><th>P(noT,noE)</th>"
                "<th>P(T,E)</th><th>P(T,noE)</th><th>True toxicity</th><th>True efficacy</th>"
                "<th>Toxicity margin</th><th>Efficacy margin</th>"
                "<th>Expected utility (0–100)</th>"
                "<th>Mean patients</th><th>Mean toxicities</th>"
                "<th>Mean efficacies</th></tr></thead>"
                f"<tbody>{truth_rows}</tbody></table>"
                "<table><caption>Final selection probabilities</caption>"
                "<thead><tr><th>Selection</th><th>OBD probability</th><th>OBD MCSE</th>"
                "<th>MTD probability</th><th>MTD MCSE</th></tr></thead>"
                f"<tbody>{selection_rows}</tbody></table>"
                "<table><caption>Stopping reasons</caption><thead><tr><th>Reason</th>"
                f"<th>Frequency</th></tr></thead><tbody>{stop_rows}</tbody></table>"
                f"{stage_rows}</section>"
            )
        stage_note = (
            "Single-stage BOIN12 simulation."
            if self.stage1_threshold is None
            else (
                "Two-stage simulation switches after the cohort that first reaches the threshold; "
                "the next assignment follows Stage 2."
            )
        )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>BOIN12 design report</title><style>body{font:16px system-ui,sans-serif;"
            "max-width:1100px;margin:2em auto;padding:0 1em;color:#202124}"
            "table{border-collapse:collapse;"
            "width:100%;margin:1em 0 2em}th,td{border:1px solid #bbc;padding:.45em;text-align:left}"
            "section{border-top:2px solid #ddd;margin-top:2em}</style></head><body>"
            "<h1>BOIN12 simulation report</h1>"
            "<p>This report records the Python design and simulation inputs. "
            "It is not a claim of native report, random-stream, or unresolved "
            "3+3 run-in precedence parity.</p>"
            f"<p>{escape(stage_note)}</p><table><caption>Design and simulation settings</caption>"
            f"<tbody>{settings_html}</tbody></table>{''.join(sections)}"
            "<p>Selection index zero denotes no selected dose. Frequencies use all trials; "
            "MCSE is the Bernoulli standard error. The joint-cell table preserves "
            "truth probabilities.</p>"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Render then atomically write this report as UTF-8 HTML."""
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


def boin12_report(
    design: BOIN12Design,
    scenarios: tuple[BOIN12ReportScenario, ...],
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1_000,
    start_dose: int = 1,
    stage1_threshold: int | None = None,
    seed: int,
) -> BOIN12Report:
    """Run validated scenarios serially and retain only compact summaries."""
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    for name in ("toxicity_limit", "efficacy_limit", "toxicity_cutoff", "efficacy_cutoff"):
        value = getattr(design, name)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"design.{name} must be a finite real scalar")
        scalar(value, f"design.{name}")
    if not 0.05 <= design.toxicity_limit <= 0.6 or not 0 < design.efficacy_limit < 1:
        raise ValueError("design limits are outside their supported ranges")
    if not 0 < design.toxicity_cutoff < 1 or not 0 < design.efficacy_cutoff < 1:
        raise ValueError("design cutoffs must lie in (0,1)")
    if (
        isinstance(design.exploration_patients, (bool, np.bool_))
        or not isinstance(design.exploration_patients, (int, np.integer))
        or design.exploration_patients < 0
    ):
        raise ValueError("design.exploration_patients must be a nonnegative integer")
    if (
        isinstance(design.stay_patients, (bool, np.bool_))
        or not isinstance(design.stay_patients, (int, np.integer))
        or design.stay_patients < 0
    ):
        raise ValueError("design.stay_patients must be a nonnegative integer")
    if design.early_stop_patients is not None and (
        isinstance(design.early_stop_patients, (bool, np.bool_))
        or not isinstance(design.early_stop_patients, (int, np.integer))
        or design.early_stop_patients <= 0
    ):
        raise ValueError("design.early_stop_patients must be a positive integer or None")
    if len(design.utilities) != 4 or any(
        isinstance(value, (bool, np.bool_)) for value in design.utilities
    ):
        raise ValueError("design.utilities must contain four real utility scores")
    for index, value in enumerate(design.utilities):
        scalar(value, f"design.utilities[{index}]")
    snapshot = BOIN12Design(
        toxicity_limit=design.toxicity_limit,
        efficacy_limit=design.efficacy_limit,
        utilities=(
            float(design.utilities[0]),
            float(design.utilities[1]),
            float(design.utilities[2]),
            float(design.utilities[3]),
        ),
        toxicity_cutoff=scalar(design.toxicity_cutoff, "design.toxicity_cutoff"),
        efficacy_cutoff=scalar(design.efficacy_cutoff, "design.efficacy_cutoff"),
        exploration_patients=design.exploration_patients,
        stay_patients=design.stay_patients,
        early_stop_patients=design.early_stop_patients,
    )
    if not isinstance(design._boin, BOINDesign):
        raise ValueError("design contains inconsistent BOIN movement boundaries")
    for name in ("target", "escalation_boundary", "deescalation_boundary"):
        actual = float(getattr(design._boin, name))
        expected = float(getattr(snapshot._boin, name))
        if not np.isfinite(actual) or abs(actual - expected) > 64 * np.finfo(float).eps * abs(
            expected
        ):
            raise ValueError("design contains inconsistent BOIN movement boundaries")
    if not isinstance(scenarios, (tuple, list)) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError("scenarios must contain 1..20 report scenarios")
    captured: list[BOIN12ReportScenario] = []
    for scenario in scenarios:
        if not isinstance(scenario, BOIN12ReportScenario):
            raise ValueError("each scenario must be a BOIN12ReportScenario")
        captured.append(BOIN12ReportScenario(scenario.label, scenario.joint_probability))
    labels = [scenario.label for scenario in captured]
    if len(set(labels)) != len(labels):
        raise ValueError("scenario labels must be unique")
    dose_count = len(captured[0]._grid)
    if any(len(item._grid) != dose_count for item in captured):
        raise ValueError("all scenarios must have the same dose count")
    ncohorts = _integer(cohorts, "cohorts", 1, 200)
    size = _integer(cohort_size, "cohort_size", 1, 200)
    repetitions = _integer(trials, "trials", 1, 1_000_000)
    start = _integer(start_dose, "start_dose", 1, dose_count)
    seed_value = _integer(seed, "seed", 0, 2**64 - 1)
    threshold = (
        None if stage1_threshold is None else _integer(stage1_threshold, "stage1_threshold", 6, 12)
    )
    planned_patients = ncohorts * size
    if planned_patients > 200:
        raise ValueError("report design may plan at most 200 patients per trial")
    scenario_count = len(captured)
    if repetitions * dose_count > _MAX_TRIAL_DOSE_CELLS:
        raise ValueError("each scenario may use at most 1000000 trial-dose cells")
    if repetitions * planned_patients * scenario_count > _MAX_PATIENT_REPLICATIONS:
        raise ValueError("report may include at most 2000000 planned patient-replications")
    if repetitions * ncohorts * dose_count * scenario_count > _MAX_WORK:
        raise ValueError(
            "report planned simulation work exceeds 2000000 trial-cohort-dose evaluations"
        )

    generator = np.random.default_rng(seed_value)
    summaries: list[BOIN12ScenarioSummary] = []
    for scenario in captured:
        grid = np.asarray(scenario._grid, dtype=np.float64)
        simulation: BOIN12Simulation | BOIN12TwoStageSimulation
        if threshold is None:
            simulation = simulate_boin12(
                snapshot,
                grid,
                cohorts=ncohorts,
                cohort_size=size,
                trials=repetitions,
                start_dose=start,
                rng=generator,
            )
            transitions = None
            mean_stage1 = mean_stage2 = None
        else:
            two_stage = simulate_boin12_two_stage(
                snapshot,
                grid,
                stage1_threshold=threshold,
                cohorts=ncohorts,
                cohort_size=size,
                trials=repetitions,
                start_dose=start,
                rng=generator,
            )
            counts = np.bincount(two_stage.transition_cohort, minlength=ncohorts + 1)
            transitions = tuple((i, float(counts[i] / repetitions)) for i in range(ncohorts + 1))
            mean_stage1 = float(np.mean(two_stage.stage1_cohorts))
            mean_stage2 = float(np.mean(two_stage.stage2_cohorts))
            simulation = two_stage
            del two_stage
        reasons = Counter(simulation.stop_reason)
        obd_prob = tuple(float(x) for x in simulation.obd_probability)
        mtd_prob = tuple(float(x) for x in simulation.mtd_probability)
        summaries.append(
            BOIN12ScenarioSummary(
                scenario.label,
                scenario._grid,
                tuple(float(row[2] + row[3]) for row in scenario._grid),
                tuple(float(row[0] + row[2]) for row in scenario._grid),
                obd_prob,
                tuple(float(np.sqrt(x * (1.0 - x) / repetitions)) for x in obd_prob),
                mtd_prob,
                tuple(float(np.sqrt(x * (1.0 - x) / repetitions)) for x in mtd_prob),
                tuple(float(x) for x in simulation.patients.mean(axis=0)),
                tuple(float(x) for x in simulation.toxicities.mean(axis=0)),
                tuple(float(x) for x in simulation.efficacies.mean(axis=0)),
                tuple((reason, float(reasons[reason] / repetitions)) for reason in sorted(reasons)),
                transitions,
                mean_stage1,
                mean_stage2,
            )
        )
        del simulation
    return BOIN12Report(
        snapshot, tuple(summaries), ncohorts, size, repetitions, start, threshold, seed_value
    )
