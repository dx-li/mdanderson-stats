"""Saved BARD BF-BOIN studies and self-contained operating-characteristic reports."""

from __future__ import annotations

import html
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from ._validation import scalar
from .bard_bf_boin_simulation import (
    BARDBFBOINSimulation,
    BARDCategoryFrequency,
    BARDMeanEstimate,
    BARDProportion,
    BARDSelectionAccuracy,
)
from .bard_study import BARDStudySpecification

_MAX_SCENARIOS = 20
_MAX_REPORT_INPUT_CHARS = 1_000_000
_MAX_REPORT_CELLS = 250_000
_MAX_REPORT_CHARS = 7_000_000
_MAX_PATIENT_WORK = 100_000_000


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _vector_snapshot(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"captured {name} must be an immutable tuple")
    return cast(tuple[float, ...], value)


def _matrix_snapshot(value: object, name: str) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, tuple) or not value or not isinstance(value[0], tuple):
        raise TypeError(f"captured {name} must be an immutable tuple of tuples")
    return cast(tuple[tuple[object, ...], ...], value)


def _frequency_rows(values: tuple[BARDCategoryFrequency, ...]) -> str:
    return "".join(
        "<tr>"
        f"<td>{html.escape(str(item.category))}</td>"
        f"<td>{item.count}</td><td>{item.denominator}</td>"
        f"<td>{_number(item.proportion)}</td><td>{_number(item.mcse)}</td></tr>"
        for item in values
    )


def _mean_cells(value: BARDMeanEstimate) -> str:
    return (
        f"<td>{_number(value.mean)}</td><td>{_number(value.sample_sd)}</td>"
        f"<td>{_number(value.mcse)}</td><td>{value.count}</td>"
    )


def _json_value(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        return item

    return json.dumps(normalize(value), allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def _setting_value(value: object) -> str:
    if isinstance(value, np.ndarray) and value.size > 16:
        return f"full array in captured JSON ({value.size} values)"
    if isinstance(value, tuple) and len(value) > 16:
        return f"full sequence in captured JSON ({len(value)} values)"
    return _json_value(value)


def _report_cell_count(specifications: tuple[BARDStudySpecification, ...]) -> int:
    cells = 0
    for item in specifications:
        doses = len(_vector_snapshot(item.true_toxicity, "true_toxicity"))
        titration_allowance = 0
        if item.accelerated_titration:
            cap = doses if item.titration_cap is None else item.titration_cap
            titration_allowance = cap - item.start_dose + 1 + item.cohort_size
        max_boundary_n = min(
            item.cohorts * item.cohort_size + item.design.n_cap + titration_allowance,
            1_000,
        )
        profiles = _matrix_snapshot(item.factor_profiles, "factor_profiles")
        cells += 5 * max_boundary_n + 7 * (doses + 1) + 5 * len(profiles[0]) + 200
    return cells


def _settings_rows(specification: BARDStudySpecification) -> str:
    design = specification.design
    stage_two = specification.stage_two
    profiles = _matrix_snapshot(specification.factor_profiles, "factor_profiles")
    eligible = np.asarray(stage_two.eligible_profiles, dtype=bool)
    pair = (
        "selected stage-one MTD and adjacent lower dose"
        if stage_two.dose_pair is None
        else stage_two.dose_pair
    )
    stage2_weights = (
        "response-model profile distribution"
        if stage_two.stage_two_profile_probabilities is None
        else _setting_value(stage_two.stage_two_profile_probabilities)
    )
    stage2_joint = stage_two.stage_two_joint_toxicity_response_probability
    joint_policy = (
        "stage-one joint table inherited"
        if stage2_joint is None and specification.joint_toxicity_response_probability is not None
        else "conditional independence"
        if stage2_joint is None
        else "stage-two joint table override"
    )
    rows: list[tuple[str, object]] = [
        ("Study label", specification.label),
        ("Scenario label", specification.scenario_label or specification.label),
        ("Random seed / trials", (specification.seed, specification.trials)),
        (
            "True OBD labels: noninferiority / utility",
            (
                specification.true_obd_noninferiority,
                specification.true_obd_utility,
            ),
        ),
        ("Stage-one BF-BOIN target / assigned-patient cap", (design.target, design.n_cap)),
        ("Stage-one precision stop n_stop", design.n_stop),
        (
            "Escalation / de-escalation boundaries",
            (
                design.escalation_boundary,
                design.deescalation_boundary,
            ),
        ),
        (
            "Elimination probability / safety offset",
            (
                design.elimination_probability,
                design.safety_offset,
            ),
        ),
        (
            "Extra-safe / bound-MTD / 1-of-3 / 2-of-6 rules",
            (
                design.extra_safe,
                design.bound_mtd,
                design.stay_at_one_of_three,
                design.deescalate_at_two_of_six,
            ),
        ),
        (
            "Stage-one cohorts / cohort size / start dose",
            (
                specification.cohorts,
                specification.cohort_size,
                specification.start_dose,
            ),
        ),
        (
            "Stage-one accrual rate / DLT window / arrival law",
            (
                specification.stage_one_accrual_rate,
                specification.dlt_window,
                specification.stage_one_arrival_distribution,
            ),
        ),
        (
            "Expansion / accelerated titration / effective titration cap",
            (
                specification.expand_after_escalation,
                specification.accelerated_titration,
                (
                    len(_vector_snapshot(specification.true_toxicity, "true_toxicity"))
                    if specification.titration_cap is None
                    else specification.titration_cap
                )
                if specification.accelerated_titration
                else None,
            ),
        ),
        (
            "Grade-2 probabilities / assessment delay",
            (
                specification.true_grade2,
                specification.grade2_assessment_delay,
            ),
        ),
        (
            "True toxicity / population response by dose",
            (
                specification.true_toxicity,
                specification.population_response,
            ),
        ),
        (
            "Response-profile count / factor count",
            (
                len(profiles),
                len(profiles[0]),
            ),
        ),
        ("Factor profiles (one-based categories)", _setting_value(specification.factor_profiles)),
        ("Population profile probabilities", _setting_value(specification.profile_probabilities)),
        (
            "Response odds ratios by factor and category",
            _setting_value(specification.response_odds_ratios),
        ),
        (
            "Stage-one joint toxicity-response policy",
            (
                "conditional independence"
                if specification.joint_toxicity_response_probability is None
                else "explicit joint table in captured JSON",
            ),
        ),
        ("Stage-two target / dose pair", (stage_two.total_target, pair)),
        (
            "Stage-two eligible profiles",
            (
                int(np.count_nonzero(eligible)),
                int(eligible.size),
            ),
        ),
        ("Stage-two profile weights", stage2_weights),
        ("Stage-two prior / safety weights", (stage_two.prior, stage_two.safety_weights)),
        (
            "Stage-two toxicity limit / efficacy limit",
            (
                stage_two.toxicity_limit,
                stage_two.efficacy_limit,
            ),
        ),
        (
            "Stage-two safety cutoff / efficacy cutoff",
            (
                stage_two.safety_cutoff,
                stage_two.efficacy_cutoff,
            ),
        ),
        (
            "Stage-two utilities / noninferiority margin / tie arm",
            (
                stage_two.utilities,
                stage_two.margin,
                stage_two.tie_arm,
            ),
        ),
        (
            "Allocation probability / tie probability",
            (
                stage_two.allocation_probability,
                stage_two.tie_probability,
            ),
        ),
        (
            "Balanced factor columns (zero-based) / stage-two arrival",
            (
                stage_two.balanced_factors,
                stage_two.arrival_distribution,
            ),
        ),
        (
            "Stage-two accrual rate / joint-endpoint policy",
            (
                stage_two.stage_two_accrual_rate,
                joint_policy,
            ),
        ),
        ("Aggregate patient-work bound", specification.patient_work_bound),
    ]
    return "".join(
        "<tr>"
        f'<th scope="row">{html.escape(label)}</th>'
        f"<td>{html.escape(_setting_value(value))}</td></tr>"
        for label, value in rows
    )


def _accuracy_html(title: str, value: BARDSelectionAccuracy) -> str:
    selected = value.correct_given_selection
    selected_html = (
        "not estimable (no selected trials)"
        if selected is None
        else (
            f"{_number(selected.probability)} (MCSE {_number(selected.mcse)}; "
            f"n={selected.denominator})"
        )
    )
    all_trials = value.correct_all_trials
    return (
        f"<p><strong>{html.escape(title)}</strong>: correct/all trials "
        f"{_number(all_trials.probability)} (MCSE {_number(all_trials.mcse)}; "
        f"n={all_trials.denominator}); correct among selected {selected_html}; "
        f"selected trials={value.selected_trials}.</p>"
    )


@dataclass(frozen=True, slots=True)
class BARDDesignReport:
    """Immutable captured study inputs and their compact OC summaries."""

    specifications: tuple[BARDStudySpecification, ...]
    summaries: tuple[BARDBFBOINSimulation, ...]
    specification_json: tuple[str, ...]
    patient_work_bound: int
    max_patient_work: int

    def __post_init__(self) -> None:
        if not (
            isinstance(self.specifications, tuple)
            and isinstance(self.summaries, tuple)
            and isinstance(self.specification_json, tuple)
        ):
            raise TypeError("report components must be immutable tuples")
        if not 1 <= len(self.specifications) <= _MAX_SCENARIOS or not (
            len(self.specifications) == len(self.summaries) == len(self.specification_json)
        ):
            raise ValueError("report components must have matching lengths in 1..20")
        if any(not isinstance(item, BARDStudySpecification) for item in self.specifications):
            raise TypeError("report specifications must be BARDStudySpecification values")
        if any(not isinstance(item, BARDBFBOINSimulation) for item in self.summaries):
            raise TypeError("report summaries must be BARDBFBOINSimulation values")
        if any(not isinstance(item, str) for item in self.specification_json):
            raise TypeError("captured specifications must be JSON text")
        if sum(len(item) for item in self.specification_json) > _MAX_REPORT_INPUT_CHARS:
            raise ValueError("captured study inputs exceed the report character limit")
        report_cells = _report_cell_count(self.specifications)
        if report_cells > _MAX_REPORT_CELLS:
            raise ValueError("requested report table cells exceed the 250,000-cell limit")
        estimated_chars = (
            6 * sum(len(item) for item in self.specification_json)
            + 40 * report_cells
            + 10_000 * len(self.specifications)
        )
        if estimated_chars > _MAX_REPORT_CHARS:
            raise ValueError("estimated HTML exceeds the 7,000,000-character report limit")
        if (
            isinstance(self.patient_work_bound, bool)
            or not isinstance(self.patient_work_bound, int)
            or not 0 <= self.patient_work_bound <= _MAX_PATIENT_WORK
            or isinstance(self.max_patient_work, bool)
            or not isinstance(self.max_patient_work, int)
            or not 1 <= self.max_patient_work <= _MAX_PATIENT_WORK
            or self.patient_work_bound > self.max_patient_work
        ):
            raise ValueError("report patient-work bounds are invalid")

    def to_html(self) -> str:
        """Render full-precision captured settings and OC summaries as HTML."""
        sections: list[str] = []
        for specification, summary, input_json in zip(
            self.specifications, self.summaries, self.specification_json, strict=True
        ):
            toxicity = _vector_snapshot(specification.true_toxicity, "true_toxicity")
            response = _vector_snapshot(specification.population_response, "population_response")
            doses = len(toxicity)
            titration_allowance = 0
            if specification.accelerated_titration:
                cap = doses if specification.titration_cap is None else specification.titration_cap
                titration_allowance = cap - specification.start_dose + 1 + specification.cohort_size
            maximum_patients_at_dose = min(
                specification.cohorts * specification.cohort_size
                + specification.design.n_cap
                + titration_allowance,
                1_000,
            )
            boundary = specification.design.boundary_table(maximum_patients_at_dose)
            boundary_rows = "".join(
                "<tr>"
                f"<td>{int(n)}</td><td>{int(up)}</td><td>{int(down)}</td>"
                f"<td>{_dash_if_beyond(int(eliminate), int(n))}</td>"
                f"<td>{_dash_if_beyond(int(lowest), int(n))}</td></tr>"
                for n, up, down, eliminate, lowest in zip(
                    boundary.patients,
                    boundary.escalate_max,
                    boundary.deescalate_min,
                    boundary.eliminate_min,
                    boundary.lowest_stop_min,
                    strict=True,
                )
            )
            dose_rows = "".join(
                "<tr>"
                f"<td>{dose}</td><td>{_number(toxicity[dose - 1])}</td>"
                f"<td>{_number(response[dose - 1])}</td>"
                f"{_dose_result_cells(summary, dose)}</tr>"
                for dose in range(1, doses + 1)
            )
            dose_rows += (
                "<tr><td>No selection</td><td>—</td><td>—</td>"
                f"{_proportion_cells(summary.noninferiority_no_selection)}"
                f"{_proportion_cells(summary.utility_no_selection)}</tr>"
            )
            factor_rows = "".join(
                f"<tr><td>Factor {index}</td>{_mean_cells(value)}</tr>"
                for index, value in enumerate(
                    summary.mean_factor_level1_proportion_imbalance, start=1
                )
            )
            count_gap = summary.mean_pair_arm_count_imbalance
            sections.append(
                "<section>"
                f"<h2>{html.escape(specification.label)}</h2>"
                "<h3>Protocol and simulation settings</h3>"
                "<table><tbody>"
                f"{_settings_rows(specification)}</tbody></table>"
                "<details><summary>Complete captured study inputs (versioned JSON)</summary>"
                f"<pre>{html.escape(input_json)}</pre></details>"
                "<h3>Stage-one BF-BOIN cutoffs</h3>"
                "<table><thead><tr><th>Patients with completed outcomes at dose</th>"
                "<th>Escalate at most if safe</th><th>De-escalate at least if safe</th>"
                "<th>Eliminate dose and above at least</th>"
                "<th>Effective lowest-dose stop at least</th></tr></thead>"
                f"<tbody>{boundary_rows}</tbody></table>"
                "<h3>Dose-level truth and final OBD selections</h3>"
                f"<p>Trials: {summary.trials}.</p>"
                "<table><thead><tr><th>Dose / selection outcome</th><th>True toxicity</th>"
                "<th>Population response</th><th>NI selection probability</th>"
                "<th>NI MCSE</th><th>Utility selection probability</th>"
                "<th>Utility MCSE</th></tr></thead>"
                f"<tbody>{dose_rows}</tbody></table>"
                + _accuracy_html("Noninferiority OBD", summary.noninferiority_accuracy)
                + _accuracy_html("Utility OBD", summary.utility_accuracy)
                + "<h3>Enrollment and duration</h3><table><thead><tr><th>Measure</th>"
                "<th>Mean</th><th>Sample SD</th><th>MCSE</th><th>Trials</th></tr></thead><tbody>"
                f"<tr><td>Total enrolled</td>{_mean_cells(summary.mean_total_enrollment)}</tr>"
                f"<tr><td>Duration</td>{_mean_cells(summary.mean_duration)}</tr>"
                "</tbody></table>"
                "<h3>Pair allocation imbalance</h3><table><thead><tr><th>Measure</th>"
                "<th>Mean</th><th>Sample SD</th><th>MCSE</th>"
                "<th>Defined trials</th></tr></thead><tbody>"
                f"<tr><td>Absolute arm-count difference</td>{_mean_cells(count_gap)}</tr>"
                "</tbody></table>"
                "<p>Arm-count metric denominator: "
                f"{summary.arm_count_metric_trials}; no pair: {summary.no_pair_trials}; "
                "safety-rejected pair: "
                f"{summary.safety_rejected_pair_trials}.</p>"
                "<h3>Per-factor level-1 allocation imbalance</h3>"
                "<table><thead><tr><th>Factor column (1-based)</th>"
                "<th>Mean absolute proportion gap</th>"
                "<th>Sample SD</th><th>MCSE</th><th>Defined trials</th></tr></thead>"
                f"<tbody>{factor_rows}</tbody></table>"
                "<p>Factor metric denominator: "
                f"{summary.factor_metric_trials}; trials excluded for an empty arm: "
                f"{summary.empty_arm_factor_metric_trials}. "
                "Differences are proportions in [0,1].</p>"
                + _status_table("Trial status", summary.status_frequency)
                + _status_table("Stage-one stop reason", summary.stage_one_stop_reason_frequency)
                + _status_table(
                    "Noninferiority selection status", summary.noninferiority_status_frequency
                )
                + _status_table("Utility selection status", summary.utility_status_frequency)
                + "</section>"
            )
        content = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            "<title>BARD design study report</title><style>"
            "body{font:16px system-ui,sans-serif;max-width:1250px;margin:2em auto;"
            "padding:0 1em;color:#202124}"
            "table{border-collapse:collapse;margin:1em 0 2em;width:100%;font-size:.9em}"
            "th,td{border:1px solid #b8bec5;padding:.4em .55em;text-align:left;vertical-align:top}"
            "th{background:#f3f5f7}section{border-top:2px solid #ddd;"
            "margin-top:2em;padding-top:1em}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:.8em 0}"
            "@media print{body{max-width:none;margin:0}.no-print{display:none}}"
            "</style></head><body>"
            "<h1>BARD BF-BOIN design report</h1>"
            "<p>This report combines saved, versioned Python study inputs with the summaries "
            "returned by the streaming simulator. Scenario seeds are independent: reordering "
            "or adding another scenario does not change an existing scenario's stream.</p>"
            f"<p>Aggregate patient-work bound: {self.patient_work_bound}; "
            f"configured maximum: {self.max_patient_work}.</p>"
            + "".join(sections)
            + _conventions_html()
            + "</body></html>"
        )
        if len(content) > _MAX_REPORT_CHARS:
            raise ValueError("rendered report exceeds the 7,000,000-character limit")
        return content

    def write_html(self, path: str | Path) -> Path:
        """Atomically write a self-contained UTF-8 HTML report."""
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


def _dash_if_beyond(value: int, maximum: int) -> str:
    return str(value) if value <= maximum else "—"


def _dose_result_cells(summary: BARDBFBOINSimulation, dose: int) -> str:
    ni = summary.noninferiority_selection_by_dose[dose - 1]
    utility = summary.utility_selection_by_dose[dose - 1]
    return (
        f"<td>{_number(ni.proportion)}</td><td>{_number(ni.mcse)}</td>"
        f"<td>{_number(utility.proportion)}</td><td>{_number(utility.mcse)}</td>"
    )


def _proportion_cells(value: BARDProportion) -> str:
    return f"<td>{_number(value.probability)}</td><td>{_number(value.mcse)}</td>"


def _status_table(title: str, values: tuple[BARDCategoryFrequency, ...]) -> str:
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table><thead><tr><th>Category</th><th>Count</th><th>Trials</th>"
        "<th>Proportion</th><th>MCSE</th></tr></thead>"
        f"<tbody>{_frequency_rows(values)}</tbody></table>"
    )


def _conventions_html() -> str:
    return (
        "<section><h2>Interpretation and implementation conventions</h2><ul>"
        "<li>Correct OBD probability is unconditional over all trials; accuracy among selected "
        "trials is separate and carries its own denominator. No-selection trials are not "
        "silently dropped from unconditional accuracy.</li>"
        "<li>Factor imbalance is the paper-defined absolute difference in the proportion at "
        "factor level 1. All modeled factors are shown. The arm-count difference is defined "
        "even when one arm is empty; factor-proportion differences require two nonempty arms. "
        "No-pair, safety-rejected, and empty-arm trials are separately counted.</li>"
        "<li>The paper and guide do not identify a joint DLT-response law or complete calendar "
        "timing policy. When no joint endpoint table is supplied, the Python simulator uses "
        "conditional independence; its stage-two arrivals begin after stage-one follow-up and "
        "use the configured renewal law.</li>"
        "<li>Stage-one pair-eligible patients are carried into stage two under the captured "
        "eligibility mask. New stage-two profile weights are conditioned on eligible profiles "
        "and renormalized by the trial runner.</li>"
        "<li>The guide describes two OC tables and downloadable HTML/Word protocol templates. "
        "This Python report records captured settings and OC summaries; it does not claim the "
        "native file layout, defaults, or random-stream parity.</li>"
        "<li>This workflow reports the BF-BOIN operating-characteristic simulator. It does not "
        "simulate the separate stochastic BF-BLRM design.</li>"
        "</ul></section>"
    )


def bard_design_report(
    specifications: Sequence[BARDStudySpecification],
    *,
    max_patient_work: int = _MAX_PATIENT_WORK,
) -> BARDDesignReport:
    """Validate all saved studies and their aggregate budget before running any trial."""
    if not isinstance(specifications, (list, tuple)):
        raise ValueError("specifications must be a bounded list or tuple")
    if not 1 <= len(specifications) <= _MAX_SCENARIOS:
        raise ValueError(f"provide 1..{_MAX_SCENARIOS} saved study specifications")
    if isinstance(max_patient_work, (bool, np.bool_)):
        raise ValueError("max_patient_work must be an integer bound")
    budget_value = scalar(max_patient_work, "max_patient_work")
    if budget_value != int(budget_value) or not 1 <= budget_value <= _MAX_PATIENT_WORK:
        raise ValueError(f"max_patient_work must be an integer in 1..{_MAX_PATIENT_WORK}")
    items = tuple(specifications)
    if any(not isinstance(item, BARDStudySpecification) for item in items):
        raise TypeError("each item must be a BARDStudySpecification")
    labels = [item.label for item in items]
    if len(set(labels)) != len(labels):
        raise ValueError("study labels must be unique")
    for item in items:
        item.validate()
    work_bound = sum(item.patient_work_bound for item in items)
    if work_bound > int(budget_value):
        raise ValueError(
            f"aggregate patient-work bound {work_bound} exceeds "
            f"max_patient_work={int(budget_value)}"
        )
    input_json_values: list[str] = []
    input_chars = 0
    for item in items:
        snapshot = item.to_json()
        if len(snapshot) > _MAX_REPORT_INPUT_CHARS:
            raise ValueError("a captured study input exceeds the 1,000,000-character report limit")
        input_chars += len(snapshot)
        if input_chars > _MAX_REPORT_INPUT_CHARS:
            raise ValueError("captured study inputs exceed the 1,000,000-character report limit")
        input_json_values.append(snapshot)
    input_json = tuple(input_json_values)
    report_cells = _report_cell_count(items)
    if report_cells > _MAX_REPORT_CELLS:
        raise ValueError("requested report table cells exceed the 250,000-cell limit")
    estimated_chars = 6 * input_chars + 40 * report_cells + 10_000 * len(items)
    if estimated_chars > _MAX_REPORT_CHARS:
        raise ValueError("estimated HTML exceeds the 7,000,000-character report limit")
    summaries = tuple(item.run() for item in items)
    return BARDDesignReport(
        items,
        summaries,
        input_json,
        work_bound,
        int(budget_value),
    )
