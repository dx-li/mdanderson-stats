"""Python-defined reports for existing BOP2-DC analyses and simulations.

These reports describe results from the package's numerical APIs. They do not
claim to reproduce the unavailable native app's report files or controls.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from numbers import Integral
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite
from .bop2_dc import BOP2DCDesign
from .bop2_dc_categorical import BOP2DCCategoricalDesign
from .bop2_dc_categorical_simulation import simulate_bop2_dc_categorical
from .bop2_dc_normal import BOP2DCNormalDesign, simulate_bop2_dc_normal
from .bop2_dc_paired import BOP2DCPairedDesign
from .bop2_dc_randomized_binary import BOP2DCRandomizedBinaryDesign
from .bop2_dc_randomized_normal import BOP2DCRandomizedNormalDesign
from .bop2_dc_randomized_normal_simulation import simulate_bop2_dc_randomized_normal
from .bop2_dc_randomized_paired import BOP2DCRandomizedPairedDesign
from .bop2_dc_randomized_paired_simulation import simulate_bop2_dc_randomized_paired
from .bop2_dc_randomized_survival import BOP2DCRandomizedSurvivalDesign
from .bop2_dc_randomized_survival_simulation import simulate_bop2_dc_randomized_survival
from .bop2_dc_survival import BOP2DCSurvivalDesign
from .bop2_dc_survival_trial import simulate_bop2_dc_survival

_MAX_REPORT_SCENARIOS = 20
_MAX_PATIENT_WORK = 1_000_000


def _label(value: str, name: str = "scenario label") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError(f"{name} must be a nonempty string of at most 120 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _array_text(value: ArrayLike) -> str:
    return ", ".join(format(float(item), ".17g") for item in np.asarray(value).ravel())


def _scenarios(values: Sequence[object]) -> None:
    if isinstance(values, (str, bytes)) or not 1 <= len(values) <= _MAX_REPORT_SCENARIOS:
        raise ValueError(f"reports require 1..{_MAX_REPORT_SCENARIOS} scenarios")


class _SeededScenario(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def n_trials(self) -> int: ...

    @property
    def seed(self) -> int: ...


def _simulation_scenarios(values: Sequence[_SeededScenario], max_subjects: int) -> tuple[str, ...]:
    """Preflight the complete saved batch, including explicit replay seeds."""
    _scenarios(values)
    labels = tuple(_label(item.label) for item in values)
    _unique_labels(labels)
    work = 0
    for item in values:
        trials = item.n_trials
        seed = item.seed
        if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, Integral):
            raise ValueError("n_trials must be an integer in [1,5000]")
        trial_value = int(trials)
        if not 1 <= trial_value <= 5_000:
            raise ValueError("n_trials must be an integer in [1,5000]")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError(
                "saved reports require an explicit nonnegative integer seed below 2**53"
            )
        seed_value = int(seed)
        if not 0 <= seed_value < 2**53:
            raise ValueError(
                "saved reports require an explicit nonnegative integer seed below 2**53"
            )
        work += trial_value * max_subjects
    if work > _MAX_PATIENT_WORK:
        raise ValueError("aggregate report simulation exceeds the patient-path work bound")
    return labels


def _unique_labels(labels: Sequence[str]) -> None:
    normalized = tuple(_label(label) for label in labels)
    if len(set(normalized)) != len(normalized):
        raise ValueError("scenario labels must be unique")


@dataclass(frozen=True)
class BOP2DCReportAction:
    name: str
    probability: float
    mcse: float | None = None
    count: int | None = None


@dataclass(frozen=True)
class BOP2DCReportLook:
    look: int
    reached: int | None
    actions: tuple[BOP2DCReportAction, ...]


@dataclass(frozen=True)
class BOP2DCReportCase:
    label: str
    truth: tuple[tuple[str, str], ...]
    actions: tuple[BOP2DCReportAction, ...]
    denominator: int | None
    method: str
    metrics: tuple[tuple[str, str], ...] = ()
    look_summaries: tuple[BOP2DCReportLook, ...] = ()
    replay_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class BOP2DCReport:
    """Compact immutable report snapshot made from a constructed design and its analyses."""

    endpoint: str
    max_subjects: int
    settings: tuple[tuple[str, str], ...]
    looks: tuple[int, ...]
    decision_rule: str
    cases: tuple[BOP2DCReportCase, ...]
    provenance: str = (
        "Python BOP2-DC community report. Native app report controls and file format were "
        "not available in the cached source set; no native report parity is claimed."
    )

    def to_html(self) -> str:
        settings = "".join(
            f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>"
            for name, value in self.settings
        )
        sections: list[str] = []
        for case in self.cases:
            look_caption = (
                "Per-look exact unconditional action probabilities"
                if case.denominator is None
                else "Per-look actions (probabilities and MCSE use all simulated trials; "
                "reached is the conditional denominator)"
            )
            truth = "".join(
                f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>"
                for name, value in case.truth
            )
            actions = "".join(
                "<tr>"
                f"<th>{escape(action.name)}</th><td>{_number(action.count)}</td>"
                f"<td>{_number(action.probability)}</td><td>{_number(action.mcse)}</td>"
                "</tr>"
                for action in case.actions
            )
            metrics = "".join(
                f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>"
                for name, value in case.metrics
            )
            look_rows_parts = []
            for look in case.look_summaries:
                action_text = "; ".join(
                    f"{action.name}: count={_number(action.count)}, "
                    f"p={_number(action.probability)}, MCSE={_number(action.mcse)}"
                    for action in look.actions
                )
                look_rows_parts.append(
                    "<tr>"
                    f"<th>{look.look}</th><td>{_number(look.reached)}</td>"
                    f"<td>{escape(action_text)}</td></tr>"
                )
            look_rows = "".join(look_rows_parts)
            seed_text = ", ".join(str(int(seed)) for seed in case.replay_seeds) or "—"
            sections.append(
                f"<section><h2>{escape(case.label)}</h2>"
                f"<p><strong>Method:</strong> {escape(case.method)}; "
                f"<strong>trial denominator:</strong> {_number(case.denominator)}; "
                f"<strong>replay seeds:</strong> {escape(seed_text)}</p>"
                f"<table><caption>Truth</caption><tbody>{truth}</tbody></table>"
                "<table><caption>Terminal decisions</caption>"
                "<thead><tr><th>Action</th><th>Count</th><th>Probability</th><th>MCSE</th>"
                f"</tr></thead><tbody>{actions}</tbody></table>"
                f"<table><caption>Additional results</caption><tbody>{metrics}</tbody></table>"
                f"<table><caption>{escape(look_caption)}</caption>"
                "<thead><tr><th>Look</th><th>Reached</th>"
                "<th>Action count, probability and MCSE</th></tr></thead>"
                f"<tbody>{look_rows}</tbody></table></section>"
            )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>BOP2-DC {escape(self.endpoint)} report</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.4em .6em;text-align:left}"
            "th{background:#f4f6f8}</style></head><body>"
            f"<h1>BOP2-DC {escape(self.endpoint)} report</h1>"
            f"<p>{escape(self.provenance)}</p><p>{escape(self.decision_rule)}</p>"
            f"<p>Planned maximum enrollment: {self.max_subjects}; looks: "
            f"{escape(', '.join(str(look) for look in self.looks))}</p>"
            f"<table><caption>Captured design and prior</caption><tbody>{settings}</tbody></table>"
            + "".join(sections)
            + "</body></html>\n"
        )

    def write_html(self, path: str | os.PathLike[str]) -> Path:
        """Atomically save the static report; its parent directory must exist."""
        destination = Path(path)
        if not destination.parent.is_dir():
            raise ValueError("report parent directory must already exist")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(self.to_html())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            raise
        return destination


@dataclass(frozen=True)
class BOP2DCNormalScenario:
    label: str
    true_mean: float
    true_sd: float
    n_trials: int
    seed: int


@dataclass(frozen=True)
class BOP2DCSurvivalScenario:
    label: str
    true_median: float
    accrual_rate: float
    final_followup: float
    n_trials: int
    seed: int
    arrival: str = "fixed"


@dataclass(frozen=True)
class BOP2DCCategoricalScenario:
    label: str
    truth_probabilities: ArrayLike
    seed: int
    n_trials: int = 100


@dataclass(frozen=True)
class BOP2DCRandomizedNormalScenario:
    label: str
    control_mean: float
    control_sd: float
    treatment_mean: float
    treatment_sd: float
    seed: int
    n_trials: int = 100


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalScenario:
    label: str
    control_median: float
    treatment_median: float
    accrual_rate: float
    final_followup: float
    seed: int
    n_trials: int = 100
    arrival: str = "fixed"


@dataclass(frozen=True)
class BOP2DCRandomizedPairedScenario:
    label: str
    control_probabilities: ArrayLike
    treatment_probabilities: ArrayLike
    seed: int
    n_trials: int = 100


def _simulation_actions(
    labels: tuple[str, ...], counts: ArrayLike, probabilities: ArrayLike, mcse: ArrayLike
) -> tuple[BOP2DCReportAction, ...]:
    count_array = np.asarray(counts)
    probability_array = np.asarray(probabilities)
    error_array = np.asarray(mcse)
    return tuple(
        BOP2DCReportAction(
            name,
            float(probability_array[index]),
            float(error_array[index]),
            int(count_array[index]),
        )
        for index, name in enumerate(labels)
    )


def _look_summaries(
    looks: ArrayLike,
    reached: ArrayLike,
    names: Sequence[str],
    counts: ArrayLike,
    probabilities: ArrayLike,
    mcse: ArrayLike,
) -> tuple[BOP2DCReportLook, ...]:
    count_array = np.asarray(counts)
    probability_array = np.asarray(probabilities)
    error_array = np.asarray(mcse)
    reached_array = np.asarray(reached)
    return tuple(
        BOP2DCReportLook(
            int(look),
            int(reached_array[index]),
            tuple(
                BOP2DCReportAction(
                    name,
                    float(probability_array[index, action]),
                    float(error_array[index, action]),
                    int(count_array[index, action]),
                )
                for action, name in enumerate(names)
            ),
        )
        for index, look in enumerate(np.asarray(looks))
    )


def _terminal_look_summaries(
    looks: ArrayLike,
    labels: Sequence[str],
    probability: ArrayLike,
    mcse: ArrayLike,
    trials: int,
) -> tuple[BOP2DCReportLook, ...]:
    """Adapt a terminal-by-look table whose probabilities use all trials."""
    probabilities = np.asarray(probability)
    errors = np.asarray(mcse)
    reached = trials
    summaries = []
    for index, look in enumerate(np.asarray(looks)):
        counts = np.rint(probabilities[index] * trials).astype(np.int64)
        summaries.append(
            BOP2DCReportLook(
                int(look),
                reached,
                tuple(
                    BOP2DCReportAction(
                        name,
                        float(probabilities[index, j]),
                        float(errors[index, j]),
                        int(counts[j]),
                    )
                    for j, name in enumerate(labels)
                ),
            )
        )
        reached -= int(np.sum(counts))
    return tuple(summaries)


def bop2_dc_binary_report(
    design: BOP2DCDesign,
    scenarios: Sequence[tuple[str, float]],
) -> BOP2DCReport:
    """Calculate exact binomial OCs for named response-probability scenarios."""
    _scenarios(scenarios)
    labels = tuple(_label(item[0]) for item in scenarios)
    _unique_labels(labels)
    truths = finite(tuple(item[1] for item in scenarios), "response probabilities")
    if np.any((truths < 0) | (truths > 1)):
        raise ValueError("response probabilities must be in [0,1]")
    oc = design.operating_characteristics(truths)
    cases = []
    for index, label in enumerate(labels):
        terminal = (
            BOP2DCReportAction("stop_no_go", float(np.sum(oc.stop_no_go[index]))),
            BOP2DCReportAction("final_go", float(oc.final_go[index])),
            BOP2DCReportAction("final_consider", float(oc.final_consider[index])),
            BOP2DCReportAction("final_no_go", float(oc.final_no_go[index])),
        )
        cases.append(
            BOP2DCReportCase(
                label,
                (("response_probability", _number(truths[index])),),
                terminal,
                None,
                "Exact binomial recursion",
                metrics=(
                    (
                        "total_no_go_probability_including_early_stops",
                        _number(oc.no_go_probability[index]),
                    ),
                    ("expected_sample_size", _number(oc.expected_sample_size[index])),
                    (
                        "sample_size_probability_by_look",
                        _array_text(oc.sample_size_probability[index]),
                    ),
                ),
                look_summaries=tuple(
                    BOP2DCReportLook(
                        int(look),
                        None,
                        (BOP2DCReportAction("stop_no_go", float(oc.stop_no_go[index, j])),),
                    )
                    for j, look in enumerate(design.looks[:-1])
                ),
            )
        )
    settings = (
        ("lrv", _number(design.lrv)),
        ("cmv", _number(design.cmv)),
        ("prior beta shapes", _array_text(design.prior)),
        ("lambda_lrv", _number(design.lambda_lrv)),
        ("lambda_cmv", _number(design.lambda_cmv)),
        ("gamma_lrv", _number(design.gamma_lrv)),
        ("gamma_cmv", _number(design.gamma_cmv)),
        ("no-go boundary at interim looks (LRV)", _array_text(design.no_go_lrv)),
        ("no-go boundary at interim looks (CMV)", _array_text(design.no_go_cmv)),
    )
    return BOP2DCReport(
        "single-arm binary",
        design.max_subjects,
        settings,
        tuple(int(x) for x in design.looks),
        "Interim stops and final go/consider/no-go use the documented strict two-criterion rules.",
        tuple(cases),
    )


def bop2_dc_normal_report(
    design: BOP2DCNormalDesign,
    scenarios: Sequence[BOP2DCNormalScenario],
) -> BOP2DCReport:
    """Simulate named Normal truths through the existing bounded OC engine."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_normal(
            design, item.true_mean, item.true_sd, n_trials=item.n_trials, rng=item.seed
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                (("true_mean", _number(item.true_mean)), ("true_sd", _number(item.true_sd))),
                _simulation_actions(
                    result.decision_labels,
                    result.decision_count,
                    result.decision_probability,
                    result.decision_mcse,
                ),
                result.trials,
                "Serial Monte Carlo",
                metrics=(
                    ("mean_enrollment", _number(result.mean_enrollment)),
                    ("enrollment_mcse", _number(result.enrollment_mcse)),
                    (
                        "sample_size_probability_at_configured_looks_in_look_order",
                        _array_text(
                            np.asarray(
                                [
                                    np.count_nonzero(result.sample_size == look)
                                    for look in design.looks
                                ]
                            )
                            / result.trials
                        ),
                    ),
                    ("rng_seed", str(result.rng_seed)),
                ),
            )
        )
    settings = (
        ("LRV mean threshold", _number(design.theta_lrv)),
        ("CMV mean threshold", _number(design.theta_cmv)),
        ("NIG prior mean", _number(design.prior_mean)),
        ("NIG prior precision", _number(design.prior_precision)),
        ("NIG prior shape", _number(design.prior_shape)),
        ("NIG prior scale", _number(design.prior_scale)),
        ("lambda_lrv", _number(design.lambda_lrv)),
        ("lambda_cmv", _number(design.lambda_cmv)),
        ("gamma_lrv", _number(design.gamma_lrv)),
        ("gamma_cmv", _number(design.gamma_cmv)),
    )
    return BOP2DCReport(
        "single-arm Normal",
        design.max_subjects,
        settings,
        tuple(int(x) for x in design.looks),
        "Normal-Inverse-Gamma posterior tails; interim futility and final go/consider/no-go.",
        tuple(cases),
    )


def bop2_dc_survival_report(
    design: BOP2DCSurvivalDesign,
    scenarios: Sequence[BOP2DCSurvivalScenario],
) -> BOP2DCReport:
    """Simulate named exponential-survival truths with explicit accrual/follow-up."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_survival(
            design,
            item.true_median,
            accrual_rate=item.accrual_rate,
            final_followup=item.final_followup,
            n_trials=item.n_trials,
            arrival=item.arrival,
            rng=item.seed,
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                (
                    ("true_median", _number(item.true_median)),
                    ("accrual_rate", _number(item.accrual_rate)),
                    ("final_followup", _number(item.final_followup)),
                    ("arrival", item.arrival),
                ),
                _simulation_actions(
                    result.decision_labels,
                    result.decision_count,
                    result.decision_probability,
                    result.decision_mcse,
                ),
                result.trials,
                "Serial Monte Carlo; exponential survival and administrative censoring",
                metrics=(
                    ("mean_enrollment", _number(result.mean_enrollment)),
                    ("enrollment_mcse", _number(result.enrollment_mcse)),
                    ("mean_events", _number(result.mean_events)),
                    ("events_mcse", _number(result.events_mcse)),
                    ("mean_total_time", _number(result.mean_total_time)),
                    ("total_time_mcse", _number(result.total_time_mcse)),
                    ("mean_duration", _number(result.mean_duration)),
                    ("duration_mcse", _number(result.duration_mcse)),
                    ("rng_seed", str(result.rng_seed)),
                ),
            )
        )
    settings = (
        ("LRV median", _number(design.lrv)),
        ("CMV median", _number(design.cmv)),
        ("inverse-gamma prior shape", _number(design.prior_shape)),
        ("inverse-gamma prior scale", _number(design.prior_scale)),
        ("lambda_lrv", _number(design.lambda_lrv)),
        ("lambda_cmv", _number(design.lambda_cmv)),
        ("gamma_lrv", _number(design.gamma_lrv)),
        ("gamma_cmv", _number(design.gamma_cmv)),
    )
    return BOP2DCReport(
        "single-arm survival",
        design.max_subjects,
        settings,
        tuple(int(x) for x in design.looks),
        "Inverse-gamma posterior median-survival tails with scheduled no-go stops.",
        tuple(cases),
    )


def bop2_dc_paired_report(
    design: BOP2DCPairedDesign,
    scenarios: Sequence[tuple[str, ArrayLike]],
) -> BOP2DCReport:
    """Calculate exact joint-category OCs for the established two-endpoint design."""
    _scenarios(scenarios)
    labels = tuple(_label(item[0]) for item in scenarios)
    _unique_labels(labels)
    truths = np.stack(
        [
            finite(item[1], f"{label} category probabilities")
            for label, item in zip(labels, scenarios, strict=True)
        ]
    )
    oc = design.operating_characteristics(truths)
    cases = []
    for index, label in enumerate(labels):
        actions = (
            BOP2DCReportAction("stop_no_go", float(np.sum(oc.stop_no_go[index]))),
            BOP2DCReportAction("final_go", float(oc.final_go[index])),
            BOP2DCReportAction("final_consider", float(oc.final_consider[index])),
            BOP2DCReportAction("final_no_go", float(oc.final_no_go[index])),
        )
        category_names = (
            "efficacy and toxicity, efficacy without toxicity, toxicity without efficacy, neither"
            if design.endpoint == "efficacy_toxicity"
            else "both endpoints, first only, second only, neither"
        )
        cases.append(
            BOP2DCReportCase(
                label,
                ((f"joint category probabilities ({category_names})", _array_text(truths[index])),),
                actions,
                None,
                "Exact bivariate recursion",
                metrics=(
                    ("expected_sample_size", _number(oc.expected_sample_size[index])),
                    (
                        "sample_size_probability_by_look",
                        _array_text(oc.sample_size_probability[index]),
                    ),
                ),
                look_summaries=tuple(
                    BOP2DCReportLook(
                        int(look),
                        None,
                        (BOP2DCReportAction("stop_no_go", float(oc.stop_no_go[index, j])),),
                    )
                    for j, look in enumerate(design.looks[:-1])
                ),
            )
        )
    settings = (
        ("endpoint mode", design.endpoint),
        ("LRV", _array_text(design.lrv)),
        ("CMV", _array_text(design.cmv)),
        ("success LRV", _array_text(design.success_lrv)),
        ("success CMV", _array_text(design.success_cmv)),
        ("Dirichlet prior", _array_text(design.prior)),
        ("lambda LRV", _array_text(design.lambda_lrv)),
        ("lambda CMV", _array_text(design.lambda_cmv)),
        ("gamma LRV", _array_text(design.gamma_lrv)),
        ("gamma CMV", _array_text(design.gamma_cmv)),
    )
    return BOP2DCReport(
        "paired binary endpoints",
        design.max_subjects,
        settings,
        tuple(int(x) for x in design.looks),
        (
            "Multiple efficacy uses OR-go/AND-no-go; efficacy/toxicity uses AND-go/OR-no-go, "
            "with toxicity interpreted as a lower-is-better endpoint."
        ),
        tuple(cases),
    )


def bop2_dc_randomized_binary_report(
    design: BOP2DCRandomizedBinaryDesign,
    scenarios: Sequence[tuple[str, float, float]],
) -> BOP2DCReport:
    """Calculate exact randomized-binary OCs for control/treatment truths."""
    _scenarios(scenarios)
    labels = tuple(_label(item[0]) for item in scenarios)
    _unique_labels(labels)
    control = finite(tuple(item[1] for item in scenarios), "control probabilities")
    treatment = finite(tuple(item[2] for item in scenarios), "treatment probabilities")
    if np.any((control < 0) | (control > 1) | (treatment < 0) | (treatment > 1)):
        raise ValueError("response probabilities must be in [0,1]")
    oc = design.operating_characteristics(control, treatment)
    cases = []
    for index, label in enumerate(labels):
        actions = (
            BOP2DCReportAction("stop_no_go", float(np.sum(oc.stop_no_go[index]))),
            BOP2DCReportAction("graduate", float(np.sum(oc.graduate[index]))),
            BOP2DCReportAction("final_go", float(oc.final_go[index])),
            BOP2DCReportAction("final_consider", float(oc.final_consider[index])),
            BOP2DCReportAction("final_no_go", float(oc.final_no_go[index])),
        )
        cases.append(
            BOP2DCReportCase(
                label,
                (
                    ("control response probability", _number(control[index])),
                    ("treatment response probability", _number(treatment[index])),
                ),
                actions,
                None,
                "Exact randomized binomial recursion",
                metrics=(
                    ("expected_sample_size", _number(oc.expected_sample_size[index])),
                    (
                        "sample_size_probability_by_look",
                        _array_text(oc.sample_size_probability[index]),
                    ),
                    ("maximum_comparison_error", _array_text(oc.maximum_comparison_error)),
                ),
                look_summaries=tuple(
                    BOP2DCReportLook(
                        int(look),
                        None,
                        (
                            BOP2DCReportAction("stop_no_go", float(oc.stop_no_go[index, j])),
                            BOP2DCReportAction("graduate", float(oc.graduate[index, j])),
                        ),
                    )
                    for j, look in enumerate(design.looks[:-1])
                ),
            )
        )
    settings = (
        ("LRV treatment-control difference", _number(design.theta_lrv)),
        ("CMV treatment-control difference", _number(design.theta_cmv)),
        ("control Beta prior", _array_text(design.control_prior)),
        ("treatment Beta prior", _array_text(design.treatment_prior)),
        ("fixed arm allocation (0 control, 1 treatment)", _array_text(design.arm_assignments)),
        ("lambda LRV/CMV", _array_text((design.lambda_lrv, design.lambda_cmv))),
        ("gamma LRV/CMV", _array_text((design.gamma_lrv, design.gamma_cmv))),
        ("graduate at interim", str(design.graduate_at_interim)),
        ("comparison tolerance", _number(design.comparison_tolerance)),
    )
    return BOP2DCReport(
        "randomized binary",
        design.max_subjects,
        settings,
        tuple(map(int, design.looks)),
        "Distinct interim stop/graduate probabilities and final go/consider/no-go.",
        tuple(cases),
    )


def bop2_dc_categorical_report(
    design: BOP2DCCategoricalDesign,
    scenarios: Sequence[BOP2DCCategoricalScenario],
) -> BOP2DCReport:
    """Simulate joint categorical endpoint truths for one- or two-arm designs."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_categorical(
            design, item.truth_probabilities, n_trials=item.n_trials, rng=item.seed
        )
        truth = result.truth_probabilities
        truth_lines = (
            tuple(
                (f"{arm} category probabilities", _array_text(row))
                for arm, row in zip(("control", "treatment"), truth, strict=False)
            )
            if design.randomized
            else (("category probabilities", _array_text(truth)),)
        )
        category_map = tuple(
            f"category {index}: "
            + ", ".join(
                f"endpoint {endpoint + 1}={'yes' if selected else 'no'}"
                for endpoint, selected in enumerate(row)
            )
            for index, row in enumerate(design.indicators.T)
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                truth_lines,
                _simulation_actions(
                    result.terminal_names,
                    result.terminal_counts,
                    result.terminal_probabilities,
                    result.terminal_mcse,
                ),
                result.terminal_counts.sum().item(),
                "Serial Monte Carlo",
                metrics=(
                    ("category mapping", " | ".join(category_map)),
                    ("expected_sample_size", _number(result.expected_sample_size)),
                    ("expected_sample_size_mcse", _number(result.expected_sample_size_mcse)),
                    ("master_rng_seed", str(result.rng_seed)),
                ),
                look_summaries=_look_summaries(
                    design.looks,
                    result.look_reached_counts,
                    result.look_action_names,
                    result.look_action_counts,
                    result.look_action_probabilities,
                    result.look_action_mcse,
                ),
                replay_seeds=tuple(int(seed) for seed in result.trial_seeds),
            )
        )
    settings = _categorical_settings(design)
    decision = (
        f"Combination={design.combination}; directions={', '.join(design.directions)}; "
        f"early graduation={design.graduate_at_interim}."
    )
    return BOP2DCReport(
        "joint categorical endpoints",
        design.max_subjects,
        settings,
        tuple(map(int, design.looks)),
        decision,
        tuple(cases),
    )


def _categorical_settings(design: BOP2DCCategoricalDesign) -> tuple[tuple[str, str], ...]:
    settings: list[tuple[str, str]] = [
        (
            "endpoint indicator matrix (rows=endpoint, columns=category)",
            repr(design.indicators.astype(int).tolist()),
        ),
        (
            "endpoint labels",
            ", ".join(f"endpoint {index + 1}" for index in range(design.n_endpoints)),
        ),
        ("endpoint directions", ", ".join(design.directions)),
        ("combination rule", design.combination),
        ("treatment Dirichlet prior", _array_text(design.prior)),
        ("LRV", _array_text(design.lrv)),
        ("CMV", _array_text(design.cmv)),
        ("lambda LRV/CMV", _array_text((design.lambda_lrv, design.lambda_cmv))),
        ("gamma LRV/CMV", _array_text((design.gamma_lrv, design.gamma_cmv))),
        ("graduate at interim", str(design.graduate_at_interim)),
        ("comparison tolerance", _number(design.comparison_tolerance)),
    ]
    if design.randomized:
        assert design.control_prior is not None and design.arm_assignments is not None
        settings.extend(
            (
                ("control Dirichlet prior", _array_text(design.control_prior)),
                (
                    "fixed arm allocation (0 control, 1 treatment)",
                    _array_text(design.arm_assignments),
                ),
            )
        )
    return tuple(settings)


def bop2_dc_randomized_paired_report(
    design: BOP2DCRandomizedPairedDesign,
    scenarios: Sequence[BOP2DCRandomizedPairedScenario],
) -> BOP2DCReport:
    """Simulate fixed-allocation paired endpoint-category truths."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_randomized_paired(
            design,
            item.control_probabilities,
            item.treatment_probabilities,
            n_trials=item.n_trials,
            rng=item.seed,
        )
        categories = (
            (
                "efficacy and toxicity",
                "efficacy without toxicity",
                "toxicity without efficacy",
                "neither",
            )
            if design.endpoint == "efficacy_toxicity"
            else ("both endpoints positive", "endpoint 1 only", "endpoint 2 only", "neither")
        )
        truth = (
            (
                "control category probabilities (" + "; ".join(categories) + ")",
                _array_text(result.control_probabilities),
            ),
            (
                "treatment category probabilities (" + "; ".join(categories) + ")",
                _array_text(result.treatment_probabilities),
            ),
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                truth,
                _simulation_actions(
                    result.terminal_names,
                    result.terminal_counts,
                    result.terminal_probabilities,
                    result.terminal_mcse,
                ),
                int(np.sum(result.terminal_counts)),
                "Serial Monte Carlo",
                metrics=(
                    ("expected_sample_size", _number(result.expected_sample_size)),
                    ("expected_sample_size_mcse", _number(result.expected_sample_size_mcse)),
                    ("maximum_quadrature_error", _number(result.maximum_quadrature_error)),
                    ("master_rng_seed", str(item.seed)),
                ),
                look_summaries=_look_summaries(
                    design.looks,
                    result.look_reached_counts,
                    result.look_action_names,
                    result.look_action_counts,
                    result.look_action_probabilities,
                    result.look_action_mcse,
                ),
                replay_seeds=tuple(int(seed) for seed in result.trial_seeds),
            )
        )
    settings = (
        ("endpoint mode", design.endpoint),
        ("control Dirichlet prior", _array_text(design.control_prior)),
        ("treatment Dirichlet prior", _array_text(design.treatment_prior)),
        ("fixed arm allocation (0 control, 1 treatment)", _array_text(design.arm_assignments)),
        ("LRV", _array_text(design.lrv)),
        ("CMV", _array_text(design.cmv)),
        ("lambda LRV/CMV", _array_text((design.lambda_lrv, design.lambda_cmv))),
        ("gamma LRV/CMV", _array_text((design.gamma_lrv, design.gamma_cmv))),
        ("graduate at interim", str(design.graduate_at_interim)),
        ("comparison tolerance", _number(design.comparison_tolerance)),
    )
    return BOP2DCReport(
        "randomized paired endpoints",
        design.max_subjects,
        settings,
        tuple(map(int, design.looks)),
        f"{design.endpoint} joint-category decision rule; control/treatment "
        "order is both, endpoint 1 only, endpoint 2 only, neither.",
        tuple(cases),
    )


def bop2_dc_randomized_normal_report(
    design: BOP2DCRandomizedNormalDesign,
    scenarios: Sequence[BOP2DCRandomizedNormalScenario],
) -> BOP2DCReport:
    """Run and report bounded Monte Carlo randomized-Normal scenarios."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_randomized_normal(
            design,
            item.control_mean,
            item.control_sd,
            item.treatment_mean,
            item.treatment_sd,
            n_trials=item.n_trials,
            rng=item.seed,
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                (
                    ("control_mean", _number(item.control_mean)),
                    ("control_sd", _number(item.control_sd)),
                    ("treatment_mean", _number(item.treatment_mean)),
                    ("treatment_sd", _number(item.treatment_sd)),
                ),
                _simulation_actions(
                    result.decision_labels,
                    result.decision_count,
                    result.decision_probability,
                    result.decision_mcse,
                ),
                item.n_trials,
                "Serial Monte Carlo",
                metrics=(
                    ("expected_sample_size", _number(result.expected_sample_size)),
                    ("enrollment_mcse", _number(result.enrollment_mcse)),
                    (
                        "sample_size_probability_by_look",
                        _array_text(result.sample_size_probability),
                    ),
                    ("maximum_quadrature_error", _number(result.maximum_quadrature_error)),
                    ("rng_seed", str(result.rng_seed)),
                ),
                look_summaries=_terminal_look_summaries(
                    result.looks,
                    result.decision_labels,
                    result.look_decision_probability,
                    result.look_decision_mcse,
                    item.n_trials,
                ),
            )
        )
    settings = (
        ("LRV treatment-control mean difference", _number(design.theta_lrv)),
        ("CMV treatment-control mean difference", _number(design.theta_cmv)),
        ("control NIG prior", _array_text(design.control_prior)),
        ("treatment NIG prior", _array_text(design.treatment_prior)),
        ("fixed arm allocation (0 control, 1 treatment)", _array_text(design.arm_assignments)),
        ("lambda LRV/CMV", _array_text((design.lambda_lrv, design.lambda_cmv))),
        ("gamma LRV/CMV", _array_text((design.gamma_lrv, design.gamma_cmv))),
        ("graduate at interim", str(design.graduate_at_interim)),
        ("comparison tolerance", _number(design.comparison_tolerance)),
        ("quadrature limit", str(design.quadrature_limit)),
    )
    return BOP2DCReport(
        "randomized Normal",
        design.max_subjects,
        settings,
        tuple(map(int, design.looks)),
        "Independent arm Normal-Inverse-Gamma posteriors with quadrature-based difference tails.",
        tuple(cases),
    )


def bop2_dc_randomized_survival_report(
    design: BOP2DCRandomizedSurvivalDesign,
    scenarios: Sequence[BOP2DCRandomizedSurvivalScenario],
) -> BOP2DCReport:
    """Run and report bounded exponential-survival randomized scenarios."""
    _simulation_scenarios(scenarios, design.max_subjects)
    cases = []
    for item in scenarios:
        result = simulate_bop2_dc_randomized_survival(
            design,
            item.control_median,
            item.treatment_median,
            accrual_rate=item.accrual_rate,
            final_followup=item.final_followup,
            n_trials=item.n_trials,
            arrival=item.arrival,
            rng=item.seed,
        )
        cases.append(
            BOP2DCReportCase(
                item.label,
                (
                    ("control_true_median", _number(item.control_median)),
                    ("treatment_true_median", _number(item.treatment_median)),
                    ("accrual_rate", _number(item.accrual_rate)),
                    ("final_followup", _number(item.final_followup)),
                    ("arrival", item.arrival),
                ),
                _simulation_actions(
                    result.decision_labels,
                    result.decision_count,
                    result.decision_probability,
                    result.decision_mcse,
                ),
                result.trials,
                "Serial Monte Carlo; exponential arm survival and administrative censoring",
                metrics=(
                    ("mean_enrollment", _number(result.mean_enrollment)),
                    ("enrollment_mcse", _number(result.enrollment_mcse)),
                    ("mean_events_by_arm", _array_text(result.mean_events)),
                    ("events_mcse_by_arm", _array_text(result.events_mcse)),
                    ("mean_exposure_by_arm", _array_text(result.mean_exposure)),
                    ("exposure_mcse_by_arm", _array_text(result.exposure_mcse)),
                    ("mean_duration", _number(result.mean_duration)),
                    ("duration_mcse", _number(result.duration_mcse)),
                    ("rng_seed", str(result.rng_seed)),
                ),
            )
        )
    settings = (
        ("LRV treatment-control median difference", _number(design.median_lrv)),
        ("CMV treatment-control median difference", _number(design.median_cmv)),
        ("control inverse-gamma prior", _array_text(design.control_prior)),
        ("treatment inverse-gamma prior", _array_text(design.treatment_prior)),
        ("fixed arm allocation (0 control, 1 treatment)", _array_text(design.arm_assignments)),
        ("lambda LRV/CMV", _array_text((design.lambda_lrv, design.lambda_cmv))),
        ("gamma LRV/CMV", _array_text((design.gamma_lrv, design.gamma_cmv))),
        ("graduate at interim", str(design.graduate_at_interim)),
        ("comparison tolerance", _number(design.comparison_tolerance)),
    )
    return BOP2DCReport(
        "randomized survival",
        design.max_subjects,
        settings,
        tuple(map(int, design.looks)),
        "Independent arm inverse-gamma models; exponential event times and fixed/Poisson arrival "
        "are Python simulation conventions.",
        tuple(cases),
    )
