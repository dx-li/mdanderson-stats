"""Reproducible, bounded PoP protocol and operating-characteristic reports."""

from __future__ import annotations

import html
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .boin import _owned
from .pop_design import PoPDesign
from .pop_simulation import PoPSimulation, simulate_pop

_MAX_SCENARIOS = 20
_MAX_SCENARIO_CELLS = 100_000
_MAX_TOTAL_REPLICATIONS = 2_000_000
_MAX_REPORT_CHARS = 2_000_000
_MAX_LABEL_CHARS = 256


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_LABEL_CHARS:
        raise ValueError(f"{name} must be nonempty and at most {_MAX_LABEL_CHARS} characters")
    return value


def _integer(value: float | int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return int(parsed)


def _truth_vector(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or np.iscomplexobj(value) or value.dtype.kind == "b":
            raise ValueError(f"{name} must be a real one-dimensional vector")
        if not 2 <= value.size <= 100:
            raise ValueError(f"{name} must contain 2..100 dose probabilities")
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= 100 or any(
            isinstance(item, (list, tuple, np.ndarray, complex, np.complexfloating, bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"{name} must be a real one-dimensional vector of length 2..100")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional vector")
    result = finite(value, name)
    if np.any((result < 0) | (result > 1)) or np.any(np.diff(result) < 0):
        raise ValueError(f"{name} must be nondecreasing with probabilities in [0,1]")
    return _owned(result)


@dataclass(frozen=True, slots=True)
class PoPReportScenario:
    """One named monotone true-toxicity vector."""

    label: str
    true_toxicity: ArrayLike


@dataclass(frozen=True, slots=True)
class PoPScenarioSummary:
    """Compact immutable operating characteristics from one simulated scenario."""

    label: str
    seed: int
    true_toxicity: tuple[float, ...]
    true_mtd: int
    selection_probability: tuple[float, ...]
    selection_mcse: tuple[float, ...]
    mean_patients: tuple[float, ...]
    mean_toxicities: tuple[float, ...]
    mean_total_patients: float
    mean_total_toxicities: float
    early_stop_probability: float
    early_stop_mcse: float
    risk_under: float
    risk_over: float
    risk_under_mcse: float
    risk_over_mcse: float


@dataclass(frozen=True, slots=True)
class PoPProtocolReport:
    """Captured PoP settings, full integer cutoff table, and scenario summaries."""

    design: PoPDesign
    boundaries: tuple[tuple[int, int, int, int, int], ...]
    scenarios: tuple[PoPScenarioSummary, ...]
    total_patients: int
    cohort_size: int
    trials: int
    start_dose: int
    titration: bool
    earlyterm: bool
    risk_cutoff: float
    seed: int
    rng_convention: str

    def to_html(self) -> str:
        """Render a bounded, self-contained report with all user text escaped."""

        def esc(value: object) -> str:
            return html.escape(str(value))

        def number(value: float) -> str:
            return format(float(value), ".17g")

        boundary_rows = "".join(
            "<tr>"
            f"<td>{n}</td><td>{up}</td><td>{down}</td>"
            f"<td>{exclude_under}</td><td>{exclude_over}</td></tr>"
            for n, up, down, exclude_under, exclude_over in self.boundaries
        )
        sections: list[str] = []
        for scenario in self.scenarios:
            dose_rows = "".join(
                f"<tr><td>{dose}</td><td>{number(truth)}</td>"
                f"<td>{number(scenario.mean_patients[dose - 1])}</td>"
                f"<td>{number(scenario.mean_toxicities[dose - 1])}</td></tr>"
                for dose, truth in enumerate(scenario.true_toxicity, start=1)
            )
            selection_rows = "".join(
                f"<tr><td>{'No MTD' if index == 0 else f'Dose {index}'}</td>"
                f"<td>{number(probability)}</td><td>{number(scenario.selection_mcse[index])}</td></tr>"
                for index, probability in enumerate(scenario.selection_probability)
            )
            sections.append(
                f"<section><h2>{esc(scenario.label)}</h2>"
                f"<p>True MTD: dose {scenario.true_mtd}, defined as the first dose "
                "closest to target on an exact distance tie. Scenario seed: "
                f"{scenario.seed}.</p>"
                "<table><caption>Truth and mean allocation</caption>"
                "<thead><tr><th>Dose</th><th>True toxicity</th><th>Mean patients</th>"
                f"<th>Mean toxicities</th></tr></thead><tbody>{dose_rows}</tbody></table>"
                f"<p>Mean total patients: {number(scenario.mean_total_patients)}; "
                f"mean total toxicities: {number(scenario.mean_total_toxicities)}.</p>"
                "<table><caption>Final selection</caption>"
                "<thead><tr><th>Selection</th><th>Probability</th><th>MCSE</th></tr></thead>"
                f"<tbody>{selection_rows}</tbody></table>"
                "<table><caption>Operating characteristics</caption><tbody>"
                "<tr><th>Early stop (all doses excluded)</th>"
                f"<td>{number(scenario.early_stop_probability)}</td>"
                f"<td>{number(scenario.early_stop_mcse)}</td></tr>"
                f"<tr><th>Underdosing risk (&gt; {number(self.risk_cutoff)} of planned "
                "enrollment below true MTD)</th>"
                f"<td>{number(scenario.risk_under)}</td><td>{number(scenario.risk_under_mcse)}</td></tr>"
                f"<tr><th>Overdosing risk (&gt; {number(self.risk_cutoff)} of planned "
                "enrollment above true MTD)</th>"
                f"<td>{number(scenario.risk_over)}</td><td>{number(scenario.risk_over_mcse)}</td></tr>"
                "</tbody></table></section>"
            )
        html_text = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>PoP protocol report</title></head><body>"
            "<h1>Posterior Predictive dose-finding protocol</h1>"
            "<table><caption>Design and simulation settings</caption><tbody>"
            f"<tr><th>Target</th><td>{number(self.design.target)}</td></tr>"
            f"<tr><th>Transition cutoff</th><td>{number(self.design.cutoff)}</td></tr>"
            f"<tr><th>Exclusion cutoff</th><td>{number(self.design.exclusion_cutoff)}</td></tr>"
            f"<tr><th>Safety minimum per dose</th><td>{self.design.safety_min_patients}</td></tr>"
            f"<tr><th>Planned patients</th><td>{self.total_patients}</td></tr>"
            f"<tr><th>Cohort size</th><td>{self.cohort_size}</td></tr>"
            f"<tr><th>Starting dose</th><td>{self.start_dose}</td></tr>"
            f"<tr><th>Accelerated titration</th><td>{self.titration}</td></tr>"
            f"<tr><th>Early exclusion stop</th><td>{self.earlyterm}</td></tr>"
            f"<tr><th>Risk cutoff</th><td>{number(self.risk_cutoff)}</td></tr>"
            f"<tr><th>Trials per scenario; root seed</th><td>{self.trials}; {self.seed}</td></tr>"
            f"<tr><th>Random stream convention</th><td>{esc(self.rng_convention)}</td></tr>"
            "</tbody></table>"
            "<h2>Complete integer-count cutoff table</h2>"
            "<p>Actions use strict PrBF comparisons. Impossible actions are encoded "
            "as -1 or n+1.</p>"
            "<table><thead><tr><th>Patients n</th><th>Maximum DLTs to escalate</th>"
            "<th>Minimum DLTs to de-escalate</th><th>Maximum DLTs to exclude under target</th>"
            "<th>Minimum DLTs to exclude over target</th></tr></thead>"
            f"<tbody>{boundary_rows}</tbody></table>"
            "<p>The predictive Bayes factor is compared with the transition cutoff C for movement "
            "and exclusion cutoff E for sticky, directional dose exclusion. Titration assigns one "
            "patient at a time until a DLT or the highest dose; the first DLT applies "
            "the dose-level "
            "transition boundary. Remaining patients are assigned in cohorts, and every assigned "
            "outcome is immediately available. After enrollment, MTD selection uses "
            "Beta(0.05,0.05) "
            "posterior means fitted by inverse-variance-weighted increasing isotonic regression. "
            "The executable selector adds 1e-10 times treated-dose rank to break flat estimates, "
            "chooses the closest admissible estimate, and selects the highest dose on an exact "
            "remaining distance tie. A separate Beta(1,1) safety screen excludes a dose and higher "
            "doses when its posterior toxicity probability exceeds 0.95, after the configured "
            "minimum patient count; untreated doses cannot be selected.</p>"
            "<p>Risk is calculated from "
            "the first dose closest to target and "
            "uses strict greater-than comparison with the stated fraction of planned enrollment. "
            "Probabilities and risks are shown as fractions, not percentages. "
            "This Python report records the cutoff actually simulated. Early-stop "
            "probability means "
            "all dose levels became excluded and is separate from final safety-based MTD selection."
            "</p>"
            f"{''.join(sections)}"
            "<p>The cached package plotting documentation mentions 95% toxicity "
            "credible intervals, "
            "but the inspected executable selector returns only the MTD and isotonic estimates; "
            "its plotting routine draws estimates and the target line, not intervals. No interval "
            "calculation is inferred here. The cached app DOM contains conditional plus3 result "
            "hooks but no corresponding input control, and its comparison behavior is unresolved; "
            "this report makes no claim about it. This is an independent Python summary, "
            "not native HTML/Word templates, plots or RNG parity.</p>"
            "</body></html>"
        )
        if len(html_text) > _MAX_REPORT_CHARS:
            raise ValueError("rendered report exceeds the configured character limit")
        return html_text

    def write_html(self, path: str | Path) -> Path:
        """Atomically write this report as UTF-8 HTML."""
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


def _snapshot_design(design: PoPDesign) -> PoPDesign:
    if not isinstance(design, PoPDesign):
        raise ValueError("design must be a PoPDesign")
    snapshot = PoPDesign(
        target=design.target,
        cutoff=design.cutoff,
        exclusion_cutoff=design.exclusion_cutoff,
        safety_min_patients=design.safety_min_patients,
    )
    return snapshot


def _capture_simulation(
    name: str, seed: int, simulation: PoPSimulation, truth: np.ndarray, target: float
) -> PoPScenarioSummary:
    true_mtd = int(np.argmin(np.abs(truth - target))) + 1
    return PoPScenarioSummary(
        label=name,
        seed=seed,
        true_toxicity=tuple(float(value) for value in truth),
        true_mtd=true_mtd,
        selection_probability=tuple(float(value) for value in simulation.selection_probability),
        selection_mcse=tuple(float(value) for value in simulation.selection_mcse),
        mean_patients=tuple(float(value) for value in simulation.mean_patients),
        mean_toxicities=tuple(float(value) for value in simulation.mean_toxicities),
        mean_total_patients=float(np.sum(simulation.mean_patients)),
        mean_total_toxicities=float(np.sum(simulation.mean_toxicities)),
        early_stop_probability=float(simulation.early_stop_probability),
        early_stop_mcse=float(simulation.early_stop_mcse),
        risk_under=float(simulation.risk_under),
        risk_over=float(simulation.risk_over),
        risk_under_mcse=float(simulation.risk_under_mcse),
        risk_over_mcse=float(simulation.risk_over_mcse),
    )


def run_pop_protocol(
    design: PoPDesign,
    scenarios: tuple[PoPReportScenario, ...],
    *,
    total_patients: int,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    titration: bool = True,
    earlyterm: bool = True,
    risk_cutoff: float = 0.8,
    seed: int,
) -> PoPProtocolReport:
    """Run PoP simulations serially and capture source-defined summaries."""
    snapshot = _snapshot_design(design)
    if not isinstance(scenarios, tuple) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must be a tuple containing 1..{_MAX_SCENARIOS} scenarios")
    normalized: list[tuple[str, np.ndarray]] = []
    names: set[str] = set()
    dose_count: int | None = None
    for scenario in scenarios:
        if not isinstance(scenario, PoPReportScenario):
            raise ValueError("each scenario must be a PoPReportScenario")
        name = _label(scenario.label, "scenario label")
        if name in names:
            raise ValueError("scenario labels must be unique")
        names.add(name)
        truth = _truth_vector(scenario.true_toxicity, f"scenario {name} true_toxicity")
        if dose_count is None:
            dose_count = int(truth.size)
        elif truth.size != dose_count:
            raise ValueError("all scenarios must have the same number of dose levels")
        normalized.append((name, truth))
    assert dose_count is not None

    n_patients = _integer(total_patients, "total_patients", 1, 1000)
    size = _integer(cohort_size, "cohort_size", 1, 4)
    n_trials = _integer(trials, "trials", 1, 100_000)
    start = _integer(start_dose, "start_dose", 1, dose_count)
    root_seed = _integer(seed, "seed", 0, 2**32 - 1)
    if not isinstance(titration, (bool, np.bool_)) or not isinstance(earlyterm, (bool, np.bool_)):
        raise ValueError("titration and earlyterm must be boolean")
    risk = scalar(risk_cutoff, "risk_cutoff")
    if not 0 <= risk <= 1:
        raise ValueError("risk_cutoff must be in [0,1]")
    if n_trials * n_patients > _MAX_SCENARIO_CELLS or n_trials * dose_count > _MAX_SCENARIO_CELLS:
        raise ValueError("each scenario must satisfy trials*patients and trials*doses <= 100000")
    if n_trials * n_patients * len(normalized) > _MAX_TOTAL_REPLICATIONS:
        raise ValueError("aggregate planned patient replications exceed 2000000")

    boundaries = snapshot.boundaries(n_patients, cohort_size=1)
    seed_sequences = np.random.SeedSequence(root_seed).spawn(len(normalized))
    scenario_seeds = []
    for child in seed_sequences:
        words = child.generate_state(2, dtype=np.uint32)
        scenario_seeds.append(int(words[0]) | (int(words[1]) << 32))
    summaries: list[PoPScenarioSummary] = []
    for (name, truth), scenario_seed in zip(normalized, scenario_seeds, strict=True):
        simulation = simulate_pop(
            snapshot,
            truth,
            total_patients=n_patients,
            cohort_size=size,
            trials=n_trials,
            start_dose=start,
            titration=bool(titration),
            earlyterm=bool(earlyterm),
            risk_cutoff=risk,
            seed=scenario_seed,
        )
        summaries.append(
            _capture_simulation(name, scenario_seed, simulation, truth, snapshot.target)
        )
        del simulation

    table = tuple(
        (int(n), int(up), int(down), int(under), int(over))
        for n, up, down, under, over in zip(
            boundaries.patients,
            boundaries.escalate_max,
            boundaries.deescalate_min,
            boundaries.exclude_under_max,
            boundaries.exclude_over_min,
            strict=True,
        )
    )
    return PoPProtocolReport(
        snapshot,
        table,
        tuple(summaries),
        n_patients,
        size,
        n_trials,
        start,
        bool(titration),
        bool(earlyterm),
        risk,
        root_seed,
        "SeedSequence spawns one uint64 seed per scenario in input order; scenarios run serially.",
    )
