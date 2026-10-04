"""Bounded operating-characteristic reports for the BF-BOIN calendar simulator."""

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
from .bf_boin import BFBOINDesign
from .bf_boin_simulation import BFBOINSimulation, simulate_bf_boin
from .boin import _owned

_MAX_SCENARIOS = 20
_MAX_TRIALS = 1_000_000
_MAX_TRIAL_RECORDS = 1_000
_MAX_AGGREGATE_RECORDS = 100_000


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 120:
        raise ValueError(f"{name} must be a nonempty string of at most 120 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _cutoff(value: int, n: int) -> str:
    return str(value) if value <= n else "—"


def _finite_mean(value: ArrayLike) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    observed = np.isfinite(array)
    return None if not np.any(observed) else float(np.mean(array[observed]))


def _positive(value: float, name: str) -> float:
    parsed = scalar(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(parsed)


def _scenario_vector(
    value: ArrayLike | None, name: str, *, optional: bool = False
) -> tuple[float, ...] | None:
    if value is None and optional:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or np.iscomplexobj(value) or value.dtype.kind == "b":
            raise ValueError(f"{name} must be a real one-dimensional probability vector")
        if not 2 <= value.size <= 100:
            raise ValueError(f"{name} must contain 2..100 probabilities")
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= 100 or any(
            isinstance(item, (list, tuple, np.ndarray, complex, np.complexfloating, bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"{name} must be a real one-dimensional probability vector")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional probability vector")
    array = finite(value, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} entries must lie in [0,1]")
    return tuple(float(item) for item in array)


def _snapshot_design(design: BFBOINDesign) -> BFBOINDesign:
    if not isinstance(design, BFBOINDesign):
        raise ValueError("design must be a BFBOINDesign")
    return BFBOINDesign(
        target=design.target,
        n_cap=design.n_cap,
        n_stop=design.n_stop,
        elimination_probability=design.elimination_probability,
        extra_safe=design.extra_safe,
        safety_offset=design.safety_offset,
        bound_mtd=design.bound_mtd,
        stay_at_one_of_three=design.stay_at_one_of_three,
    )


@dataclass(frozen=True, slots=True)
class BFBOINReportScenario:
    """One truth scenario; grade-2 risk is conditional on no DLT."""

    label: str
    true_toxicity: ArrayLike
    true_response: ArrayLike
    true_grade2: ArrayLike | None = None
    _toxicity: tuple[float, ...] = field(init=False, repr=False)
    _response: tuple[float, ...] = field(init=False, repr=False)
    _grade2: tuple[float, ...] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        label = _label(self.label, "scenario label")
        toxicity = _scenario_vector(self.true_toxicity, "true_toxicity")
        response = _scenario_vector(self.true_response, "true_response")
        grade2 = _scenario_vector(self.true_grade2, "true_grade2", optional=True)
        assert toxicity is not None and response is not None
        if len(response) != len(toxicity) or (grade2 is not None and len(grade2) != len(toxicity)):
            raise ValueError("scenario truth vectors must have matching dose counts")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "true_toxicity", _owned(toxicity))
        object.__setattr__(self, "true_response", _owned(response))
        object.__setattr__(self, "true_grade2", None if grade2 is None else _owned(grade2))
        object.__setattr__(self, "_toxicity", toxicity)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_grade2", grade2)


@dataclass(frozen=True, slots=True)
class BFBOINScenarioSummary:
    """Immutable compact operating-characteristic summary for one scenario."""

    label: str
    true_toxicity: tuple[float, ...]
    true_response: tuple[float, ...]
    true_grade2_conditional_on_no_dlt: tuple[float, ...] | None
    selection_probability: tuple[float, ...]
    selection_mcse: tuple[float, ...]
    no_mtd_probability: float
    no_mtd_mcse: float
    mean_assigned_by_dose: tuple[float, ...]
    allocated_share_by_dose: tuple[float, ...]
    mean_completed_by_dose: tuple[float, ...]
    mean_dlt_by_dose: tuple[float, ...]
    mean_total_assigned: float
    mean_total_dlt: float
    stop_reason_frequency: tuple[tuple[str, float], ...]
    mean_trial_duration: float
    mean_escalation_end: float
    expansion_reason_frequency: tuple[tuple[str, float], ...]
    mean_expansion_patients: float
    expansion_end_trials: int
    mean_expansion_end: float | None
    titration_stop_reason_frequency: tuple[tuple[str, float], ...]
    mean_titration_patients: float
    mean_titration_grade2: float
    titration_end_trials: int
    mean_titration_end: float | None


@dataclass(frozen=True, slots=True)
class BFBOINDesignReport:
    """Static BF-BOIN design and simulation summary snapshot."""

    design: BFBOINDesign
    scenarios: tuple[BFBOINScenarioSummary, ...]
    cohorts: int
    cohort_size: int
    trials: int
    start_dose: int
    accrual_rate: float
    dlt_window: float
    time_unit: str
    arrival_distribution: Literal["uniform", "exponential"]
    expand_after_escalation: bool
    accelerated_titration: bool
    titration_cap: int | None
    grade2_assessment_delay: float | None
    seed: int

    def to_html(self) -> str:
        """Render an escaped self-contained report, including source conduct notes."""
        design = self.design
        settings = (
            ("Target DLT probability", design.target),
            ("Backfill assigned-patient cap n_cap", design.n_cap),
            ("Precision stop n_stop", design.n_stop),
            ("Escalation boundary", design.escalation_boundary),
            ("De-escalation boundary", design.deescalation_boundary),
            ("Safety elimination cutoff", design.elimination_probability),
            ("Extra safety", design.extra_safe),
            ("Extra-safety offset", design.safety_offset),
            ("Stay at 1 DLT among 3 at current dose", design.stay_at_one_of_three),
            ("Bound selected MTD", design.bound_mtd),
            ("Ordinary cohorts", self.cohorts),
            ("Cohort size", self.cohort_size),
            ("Trials per scenario", self.trials),
            ("Starting dose", self.start_dose),
            ("Accrual rate per time unit", self.accrual_rate),
            ("DLT assessment window", self.dlt_window),
            ("Time unit", self.time_unit),
            ("Arrival distribution", self.arrival_distribution),
            ("Post-escalation expansion", self.expand_after_escalation),
            ("Accelerated titration", self.accelerated_titration),
            ("Effective titration cap", self.titration_cap),
            ("Grade-2 assessment delay", self.grade2_assessment_delay),
            ("NumPy seed", self.seed),
        )
        setting_rows = "".join(
            f'<tr><th scope="row">{escape(label)}</th><td>{escape(_setting_value(value))}</td></tr>'
            for label, value in settings
        )
        titration_allowance = (
            self.titration_cap - self.start_dose + 1 + self.cohort_size
            if self.titration_cap is not None
            else 0
        )
        table = design.boundary_table(
            min(self.cohorts * self.cohort_size + design.n_cap + titration_allowance, 1_000)
        )
        boundary_rows = "".join(
            "<tr>"
            f"<td>{int(n)}</td><td>{int(escalate)}</td><td>{int(deescalate)}</td>"
            f"<td>{_cutoff(int(eliminate), int(n))}</td>"
            f"<td>{_cutoff(int(lowest), int(n))}</td></tr>"
            for n, escalate, deescalate, eliminate, lowest in zip(
                table.patients,
                table.escalate_max,
                table.deescalate_min,
                table.eliminate_min,
                table.lowest_stop_min,
                strict=True,
            )
        )
        sections: list[str] = []
        for scenario in self.scenarios:
            dose_rows_list: list[str] = []
            for dose in range(1, len(scenario.true_toxicity) + 1):
                grade2 = scenario.true_grade2_conditional_on_no_dlt
                grade2_value = None if grade2 is None else grade2[dose - 1]
                dose_rows_list.append(
                    f"<tr><td>{dose}</td><td>{_number(scenario.true_toxicity[dose - 1])}</td>"
                    f"<td>{_number(scenario.true_response[dose - 1])}</td>"
                    f"<td>{_number(grade2_value)}</td>"
                    f"<td>{_percent(scenario.selection_probability[dose])}</td>"
                    f"<td>{_percent(scenario.selection_mcse[dose])}</td>"
                    f"<td>{_number(scenario.mean_assigned_by_dose[dose - 1])}</td>"
                    f"<td>{_percent(scenario.allocated_share_by_dose[dose - 1])}</td>"
                    f"<td>{_number(scenario.mean_completed_by_dose[dose - 1])}</td>"
                    f"<td>{_number(scenario.mean_dlt_by_dose[dose - 1])}</td></tr>"
                )
            dose_rows = "".join(dose_rows_list)
            reason_rows = _frequency_rows(scenario.stop_reason_frequency)
            expansion_rows = _frequency_rows(scenario.expansion_reason_frequency)
            titration_rows = _frequency_rows(scenario.titration_stop_reason_frequency)
            sections.append(
                f"<section><h2>{escape(scenario.label)}</h2>"
                "<table><caption>Truth and per-dose operating characteristics</caption>"
                "<thead><tr><th>Dose</th><th>True DLT probability</th>"
                "<th>True response probability</th>"
                "<th>True grade-2 probability, conditional on no DLT</th>"
                "<th>Selection</th><th>Selection MCSE</th><th>Mean assigned</th>"
                "<th>Allocated share</th><th>Mean completed</th><th>Mean DLTs</th></tr></thead>"
                f"<tbody>{dose_rows}</tbody></table>"
                f"<p>Mean total assigned: {_number(scenario.mean_total_assigned)}; "
                f"mean total DLTs: {_number(scenario.mean_total_dlt)}; "
                f"no-MTD selection probability: {_percent(scenario.no_mtd_probability)} "
                f"(MCSE {_percent(scenario.no_mtd_mcse)}); mean duration: "
                f"{_number(scenario.mean_trial_duration)} {escape(self.time_unit)}; "
                f"mean escalation end: {_number(scenario.mean_escalation_end)} "
                f"{escape(self.time_unit)}.</p>"
                f"<p>Mean expansion end among {scenario.expansion_end_trials} trials with a finite "
                f"end time: {_number(scenario.mean_expansion_end)} {escape(self.time_unit)}.</p>"
                f"<table><caption>Trial stop reasons</caption>"
                "<thead><tr><th>Reason</th><th>Frequency</th></tr></thead>"
                f"<tbody>{reason_rows}</tbody></table>"
                f"<table><caption>Post-escalation expansion</caption>"
                f"<tbody>{expansion_rows}</tbody></table>"
                f"<p>Mean expansion patients: {_number(scenario.mean_expansion_patients)}.</p>"
                f"<table><caption>Accelerated titration</caption>"
                f"<tbody>{titration_rows}</tbody></table>"
                f"<p>Mean singleton patients: {_number(scenario.mean_titration_patients)}; "
                "mean observed grade-2 events at titration exit: "
                f"{_number(scenario.mean_titration_grade2)}; "
                f"mean titration end among {scenario.titration_end_trials} trials with a finite "
                f"end time: {_number(scenario.mean_titration_end)} {escape(self.time_unit)}.</p>"
                "</section>"
            )
        return (
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>BF-BOIN design report</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em;"
            "width:100%}th,td{border:1px solid #b8bec5;padding:.45em .65em;text-align:left}"
            "caption{text-align:left;font-weight:700;margin-bottom:.5em}"
            "section{border-top:2px solid #ddd;margin-top:2em}</style></head><body>\n"
            "<h1>BF-BOIN operating-characteristic report</h1>"
            "<p>This Python report composes the validated BF-BOIN calendar simulator. It records "
            "no-MTD selection separately from stop reasons and reports the allocated share as "
            "the ratio of mean assigned counts, not the mean of trial-specific shares. It is not "
            "a claim of native application or CRAN random-stream/report parity.</p>"
            "<table><caption>Design and simulation settings</caption><tbody>"
            f"{setting_rows}</tbody></table>"
            "<table><caption>Underlying BOIN cutoffs by patients at a dose</caption>"
            "<thead><tr><th>Patients at dose</th><th>Escalate at most</th>"
            "<th>De-escalate at least</th><th>Eliminate dose and above at least</th>"
            "<th>Effective lowest-dose stop at least</th></tr></thead>"
            f"<tbody>{boundary_rows}</tbody></table>"
            f"{''.join(sections)}"
            "<section><h2>Conduct and timing conventions</h2><ul>"
            "<li>At the current dose, escalate at a completed DLT rate no greater than the "
            "escalation boundary, de-escalate at a rate at least the de-escalation boundary, "
            "and otherwise stay. The Guide-defined optional 1-DLT-of-3 modification is "
            "available only at target 0.25 and changes that individual action to stay. For "
            "backfill conflicts, this Python implementation applies the modifier to the "
            "individual action before the existing conflict-pooling rule; the source does not "
            "specify this interaction explicitly. "
            "The safety posterior uses Beta(1,1); after at least three patients, a posterior "
            "overdose probability strictly above the cutoff eliminates that dose and all "
            "higher doses. An extra-safe rule may stop at the lowest dose; `n_stop` is a "
            "precision stop when the next action would stay and assigned count reaches `n_stop`. "
            "The BF extra-safe stop requires n>3 at dose 1, in addition to the ordinary "
            "n>=3 elimination rule. `bound_mtd` requires a fitted probability strictly below "
            "the de-escalation boundary.</li>"
            "<li>Backfill activity is established by an observed response at that dose or below. "
            "Empirical backfill closure requires both the dose-specific and adjacent pooled "
            "evaluable DLT rates to exceed the de-escalation boundary; closures may reopen. "
            "Pending outcomes are not treated as non-DLTs. The assigned-patient cap applies "
            "to backfill eligibility, not ordinary escalation assignments. When a lower "
            "backfilled dose conflicts with the current action, the implementation pools "
            "from the highest conflicting dose through the current dose and applies the "
            "paper's pooled movement rule.</li>"
            "<li>Movement waits for the current escalation cohort's DLT assessments. Final "
            "follow-up includes all enrolled patients. Optional expansion targets one dose "
            "below the last dose treated for escalation and continues under backfill "
            "eligibility.</li>"
            "<li>Arrivals use the selected renewal distribution; the first arrival is at time "
            "zero. "
            "DLT times follow the Python Weibull calibration F(window)=p and F(window/2)=p/2; "
            "assessment occurs at the earlier of event time and window. Responses are sampled "
            "at enrollment and observed at arrival plus the DLT window.</li>"
            "<li>In accelerated titration, grade-2 probability is conditional on no DLT. The "
            "grade-2 probability model and assessment delay are caller inputs because the guide "
            "does not specify them. The fixed Python delay is shown above.</li>"
            "<li>Scenario simulations run serially from one NumPy Generator initialized with "
            "the recorded seed. This stream and Python completion-time decision schedule do "
            "not claim native app or third-party CRAN parity.</li>"
            "<li>Final MTD selection uses all completed DLT data and the BOIN isotonic estimate; "
            "`bound_mtd` requires a fitted probability strictly below the de-escalation "
            "boundary.</li>"
            "</ul></section>"
            "<p>Native Figure 15 labels a `% Pts treated` field; this report's `Allocated share` "
            "uses sum of mean assigned counts divided by total mean assigned count because the "
            "native averaging formula is not specified in the available source. Its displayed "
            "early-stopping percentage is not assumed to equal the Python no-MTD probability.</p>"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Render before atomically replacing the requested UTF-8 HTML destination."""
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


def bf_boin_design_report(
    design: BFBOINDesign,
    scenarios: tuple[BFBOINReportScenario, ...],
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1_000,
    start_dose: int = 1,
    accrual_rate: float = 1.0,
    dlt_window: float = 1.0,
    time_unit: str = "month",
    arrival_distribution: Literal["uniform", "exponential"] = "uniform",
    expand_after_escalation: bool = False,
    accelerated_titration: bool = False,
    titration_cap: int | None = None,
    grade2_assessment_delay: float | None = None,
    seed: int,
) -> BFBOINDesignReport:
    """Run bounded scenarios serially and retain only operating-characteristic summaries."""
    snapshot = _snapshot_design(design)
    if not isinstance(scenarios, tuple) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must be a tuple containing 1..{_MAX_SCENARIOS} values")
    if any(not isinstance(scenario, BFBOINReportScenario) for scenario in scenarios):
        raise ValueError("every scenario must be a BFBOINReportScenario")
    dose_count = len(scenarios[0]._toxicity)
    if any(
        len(scenario._toxicity) != dose_count
        or len(scenario._response) != dose_count
        or (scenario._grade2 is not None and len(scenario._grade2) != dose_count)
        for scenario in scenarios
    ):
        raise ValueError("all scenarios must have matching dose counts")
    if len({scenario.label for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario labels must be unique")
    n_cohorts = _integer(cohorts, "cohorts", 1, 1_000)
    size = _integer(cohort_size, "cohort_size", 1, 1_000)
    n_trials = _integer(trials, "trials", 1, _MAX_TRIALS)
    start = _integer(start_dose, "start_dose", 1, dose_count)
    rate = _positive(accrual_rate, "accrual_rate")
    window = _positive(dlt_window, "dlt_window")
    unit = _label(time_unit, "time_unit")
    if arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
    if not isinstance(expand_after_escalation, (bool, np.bool_)):
        raise ValueError("expand_after_escalation must be boolean")
    if not isinstance(accelerated_titration, (bool, np.bool_)):
        raise ValueError("accelerated_titration must be boolean")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer in [0,2**32-1]")
    seed_value = scalar(seed, "seed")
    if seed_value != int(seed_value) or not 0 <= seed_value <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0,2**32-1]")

    effective_titration_cap: int | None = None
    grade_delay: float | None = None
    if accelerated_titration:
        effective_titration_cap = (
            dose_count
            if titration_cap is None
            else _integer(titration_cap, "titration_cap", start, dose_count)
        )
        if grade2_assessment_delay is None:
            raise ValueError("accelerated titration requires grade2_assessment_delay")
        grade_delay = _positive(grade2_assessment_delay, "grade2_assessment_delay")
        if any(scenario._grade2 is None for scenario in scenarios):
            raise ValueError("accelerated titration requires true_grade2 in every scenario")
    elif (
        titration_cap is not None
        or grade2_assessment_delay is not None
        or any(scenario._grade2 is not None for scenario in scenarios)
    ):
        raise ValueError(
            "titration settings and scenario grade2 vectors require accelerated_titration=True"
        )

    max_cohort_patients = n_cohorts * size
    titration_allowance = (
        effective_titration_cap - start + 1 + size if effective_titration_cap is not None else 0
    )
    records_per_trial = max_cohort_patients + dose_count * snapshot.n_cap + titration_allowance
    if records_per_trial > _MAX_TRIAL_RECORDS:
        raise ValueError("one scenario trial may retain at most 1000 patient records")
    total_records = records_per_trial * n_trials * len(scenarios)
    if total_records > _MAX_AGGREGATE_RECORDS:
        raise ValueError("all scenario simulations may retain at most 100000 patient records")

    generator = np.random.default_rng(int(seed_value))
    summaries: list[BFBOINScenarioSummary] = []
    for scenario in scenarios:
        simulation: BFBOINSimulation = simulate_bf_boin(
            snapshot,
            scenario._toxicity,
            scenario._response,
            cohorts=n_cohorts,
            cohort_size=size,
            trials=n_trials,
            start_dose=start,
            accrual_rate=rate,
            dlt_window=window,
            arrival_distribution=arrival_distribution,
            expand_after_escalation=bool(expand_after_escalation),
            accelerated_titration=bool(accelerated_titration),
            titration_cap=effective_titration_cap,
            true_grade2=scenario._grade2,
            grade2_assessment_delay=grade_delay,
            rng=generator,
        )
        assigned = np.asarray(simulation.mean_assigned, dtype=np.float64)
        total_assigned = float(assigned.sum())
        allocation_share = (
            tuple(float(value) for value in assigned / total_assigned)
            if total_assigned > 0
            else tuple(0.0 for _ in assigned)
        )
        no_mtd = float(simulation.selection_probability[0])
        stop_counts = Counter(simulation.stop_reason)
        expansion_counts = Counter(simulation.expansion_stop_reason)
        titration_counts = Counter(
            simulation.titration_stop_reason or ("not_requested",) * n_trials
        )
        summaries.append(
            BFBOINScenarioSummary(
                scenario.label,
                scenario._toxicity,
                scenario._response,
                scenario._grade2,
                tuple(float(value) for value in simulation.selection_probability),
                tuple(float(value) for value in simulation.selection_mcse),
                no_mtd,
                float(simulation.selection_mcse[0]),
                tuple(float(value) for value in assigned),
                allocation_share,
                tuple(float(value) for value in simulation.mean_patients),
                tuple(float(value) for value in simulation.mean_toxicities),
                total_assigned,
                float(simulation.mean_toxicities.sum()),
                tuple((reason, stop_counts[reason] / n_trials) for reason in sorted(stop_counts)),
                float(np.mean(simulation.trial_duration)),
                float(np.mean(simulation.escalation_end)),
                tuple(
                    (reason, expansion_counts[reason] / n_trials)
                    for reason in sorted(expansion_counts)
                ),
                float(np.mean(simulation.expansion_patients)),
                int(np.count_nonzero(np.isfinite(simulation.expansion_end))),
                _finite_mean(simulation.expansion_end),
                tuple(
                    (reason, titration_counts[reason] / n_trials)
                    for reason in sorted(titration_counts)
                ),
                0.0
                if simulation.titration_patients is None
                else float(np.mean(simulation.titration_patients)),
                0.0
                if simulation.titration_grade2 is None
                else float(np.mean(simulation.titration_grade2)),
                0
                if simulation.titration_end is None
                else int(np.count_nonzero(np.isfinite(simulation.titration_end))),
                None
                if simulation.titration_end is None
                else _finite_mean(simulation.titration_end),
            )
        )
        del simulation

    return BFBOINDesignReport(
        snapshot,
        tuple(summaries),
        n_cohorts,
        size,
        n_trials,
        start,
        rate,
        window,
        unit,
        arrival_distribution,
        bool(expand_after_escalation),
        bool(accelerated_titration),
        effective_titration_cap,
        grade_delay,
        int(seed_value),
    )


def _setting_value(value: float | int | bool | str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    return str(value) if isinstance(value, str) else _number(value)


def _percent(value: float) -> str:
    return f"{_number(100 * value)}%"


def _frequency_rows(values: tuple[tuple[str, float], ...]) -> str:
    if not values:
        return '<tr><td colspan="2">Not applicable</td></tr>'
    return "".join(
        f"<tr><td>{escape(label)}</td><td>{_percent(frequency)}</td></tr>"
        for label, frequency in values
    )
