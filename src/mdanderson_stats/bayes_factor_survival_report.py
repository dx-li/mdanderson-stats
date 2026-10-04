"""Reproducible HTML reports for Bayes Factor TTE analyses."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite, scalar
from .bayes_factor_survival import (
    BayesFactorSurvivalBoundaries,
    bayes_factor_survival_boundaries,
)
from .bayes_factor_survival_calendar import simulate_bayes_factor_survival

_MAX_SCENARIOS = 20
_MAX_BOUNDARY_ROWS = 501
_MAX_BOUNDARY_EVENT = 500
_MAX_REPORT_WORK = 1_000_000
_MAX_REPORT_QUADRATURES = 100_000


def _label(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float) -> str:
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    if np.isnan(value):
        return "undefined"
    return format(float(value), ".17g")


def _bounded_vector(value: ArrayLike, name: str, maximum: int) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > maximum or value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > maximum or any(
            isinstance(item, (list, tuple, np.ndarray)) or np.ndim(item) != 0 for item in value
        ):
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    values = finite(value, name)
    if values.ndim != 1 or values.size > maximum:
        raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    return np.array(values, dtype=np.float64, copy=True)


def _terminal_stop_flags(
    early_inferiority: NDArray[np.bool_],
    early_superiority: NDArray[np.bool_],
    final_inferiority: NDArray[np.bool_],
    final_superiority: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Use an early stop as terminal even if later follow-up reverses it."""
    no_early = ~(early_inferiority | early_superiority)
    return (
        early_inferiority | (no_early & final_inferiority),
        early_superiority | (no_early & final_superiority),
    )


@dataclass(frozen=True)
class BayesFactorSurvivalScenarioSummary:
    """Compact operating-characteristic summary for one true median."""

    true_median: float
    true_mean: float
    seed: int
    terminal_inferiority_probability: float
    terminal_inferiority_mcse: float
    terminal_superiority_probability: float
    terminal_superiority_mcse: float
    early_inferiority_probability: float
    early_inferiority_mcse: float
    early_superiority_probability: float
    early_superiority_mcse: float
    final_inferiority_probability: float
    final_inferiority_mcse: float
    final_superiority_probability: float
    final_superiority_mcse: float
    mean_patients_enrolled: float
    patient_count_quantiles: tuple[float, float, float]


@dataclass(frozen=True)
class BayesFactorSurvivalReport:
    """Immutable report snapshot from explicit Python timing assumptions."""

    null_median: float
    alternative_median_mode: float
    inferiority_cutoff: float
    superiority_cutoff: float
    accrual_rate: float
    max_patients: int
    repetitions: int
    check_times: tuple[float, ...]
    final_followup: float
    time_unit: str
    seed: int
    max_total_work: int
    max_total_quadratures: int
    max_boundary_rows: int
    title: str
    scenarios: tuple[BayesFactorSurvivalScenarioSummary, ...]
    boundaries: BayesFactorSurvivalBoundaries | None

    def to_html(self) -> str:
        """Render a self-contained HTML record with full-precision inputs."""
        title = escape(_label(self.title, "title", 120))
        unit = escape(_label(self.time_unit, "time_unit", 40))
        settings = (
            ("Null median TTE", f"{_number(self.null_median)} {unit}"),
            ("Alternative median mode TTE", f"{_number(self.alternative_median_mode)} {unit}"),
            ("Inferiority cutoff", _number(self.inferiority_cutoff)),
            ("Superiority cutoff", _number(self.superiority_cutoff)),
            ("Accrual rate", f"{_number(self.accrual_rate)} patients per {unit}"),
            ("Maximum patients", str(self.max_patients)),
            ("Repetitions per scenario", str(self.repetitions)),
            ("Check times", ", ".join(_number(x) for x in self.check_times) or "none"),
            (
                "Final follow-up after the later of planned last arrival and last check",
                _number(self.final_followup),
            ),
            ("Top-level seed", str(self.seed)),
            ("Aggregate worst-case simulation work cap", str(self.max_total_work)),
            ("Aggregate worst-case Bayes-factor evaluation cap", str(self.max_total_quadratures)),
            ("Boundary row cap", str(self.max_boundary_rows)),
        )
        settings_rows = "\n".join(
            f'<tr><th scope="row">{escape(label)}</th><td>{value}</td></tr>'
            for label, value in settings
        )
        scenario_rows: list[str] = []
        for index, row in enumerate(self.scenarios, start=1):
            q10, q50, q90 = row.patient_count_quantiles
            scenario_rows.append(
                "<tr>"
                f"<td>{index}</td><td>{_number(row.true_median)}</td>"
                f"<td>{_number(row.true_mean)}</td><td>{row.seed}</td>"
                f"<td>{_number(row.terminal_superiority_probability)}</td>"
                f"<td>{_number(row.terminal_superiority_mcse)}</td>"
                f"<td>{_number(row.terminal_inferiority_probability)}</td>"
                f"<td>{_number(row.terminal_inferiority_mcse)}</td>"
                f"<td>{_number(row.early_superiority_probability)}</td>"
                f"<td>{_number(row.early_superiority_mcse)}</td>"
                f"<td>{_number(row.early_inferiority_probability)}</td>"
                f"<td>{_number(row.early_inferiority_mcse)}</td>"
                f"<td>{_number(row.final_superiority_probability)}</td>"
                f"<td>{_number(row.final_superiority_mcse)}</td>"
                f"<td>{_number(row.final_inferiority_probability)}</td>"
                f"<td>{_number(row.final_inferiority_mcse)}</td>"
                f"<td>{_number(row.mean_patients_enrolled)}</td>"
                f"<td>{_number(q10)}, {_number(q50)}, {_number(q90)}</td>"
                "</tr>"
            )
        boundary_section = ""
        if self.boundaries is not None:
            rows = "\n".join(
                "<tr>"
                f"<td>{int(event)}</td>"
                f"<td>{_number(lower)} {unit}</td>"
                f"<td>{_number(upper)} {unit}</td>"
                "</tr>"
                for event, lower, upper in zip(
                    self.boundaries.events,
                    self.boundaries.inferiority_time,
                    self.boundaries.superiority_time,
                    strict=True,
                )
            )
            boundary_section = (
                "<h2>Continuous total time on test boundaries</h2>"
                "<p>Inferiority requires exposure below its boundary; superiority requires "
                "above the other. These are continuous roots in the "
                f"unit; no integer-day rounding or conversion is applied.</p><table><thead><tr>"
                "<th>Observed events</th><th>Inferiority boundary (&lt;)</th>"
                f"<th>Superiority boundary (&gt;)</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1400px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.4em .55em;text-align:left}"
            "</style></head><body>"
            f"<h1>{title}</h1><p>Bayes Factor single-arm time-to-event analysis. "
            "The exponential model uses the null median and alternative prior-mode median below, "
            "equal prior odds, and the specified one-sided iMOM prior. "
            "This is a saved Python snapshot.</p>"
            "<h2>Effective design inputs and Python resource settings</h2>"
            f"<table><tbody>{settings_rows}</tbody></table>"
            "<h2>Scenario operating characteristics</h2>"
            "<p>Each scenario uses its own recorded seed. Terminal stopping selects the first "
            "early boundary crossed; only trials without an early stop use the final decision. "
            "Early and final columns show those monitoring stages. Monte Carlo errors are plug-in "
            "binomial standard errors. Patient quantiles are 10th, 50th, and 90th percentiles.</p>"
            "<table><thead><tr><th>Scenario</th><th>True median</th><th>True mean</th>"
            "<th>Child seed</th><th>Terminal superiority</th><th>MCSE</th>"
            "<th>Terminal inferiority</th><th>MCSE</th><th>Early superiority</th><th>MCSE</th>"
            "<th>Early inferiority</th><th>MCSE</th><th>Final-monitor superiority</th><th>MCSE</th>"
            "<th>Final-monitor inferiority</th><th>MCSE</th><th>Mean patients enrolled</th>"
            "<th>Patient count quantiles (10%, 50%, 90%)</th></tr></thead><tbody>"
            f"{''.join(scenario_rows)}</tbody></table>"
            f"{boundary_section}"
            "<h2>Timing and numerical conventions</h2>"
            "<p>Calendar simulation is a Python policy: the first arrival is at time zero. Later "
            "interarrival gaps are exponential at the stated rate; event times use each median. "
            "Checks use the supplied absolute schedule. Follow-up is added after the later of "
            "the last planned arrival and last check. Early decisions stop accrual, but follow-up "
            "change the final-monitor decision. Native arrival generation, check scheduling, and "
            "final-follow-up timing are unspecified, so native calendar parity is not claimed. "
            "The guide prints integer-day boundaries; this report keeps continuous roots "
            "without inferring native day-rounding behavior.</p>"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically write the rendered report as UTF-8 HTML."""
        destination = Path(path)
        content = self.to_html()
        temporary_path: Path | None = None
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
                temporary_path = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination


def bayes_factor_survival_report(
    *,
    null_median: float,
    alternative_median_mode: float,
    accrual_rate: float,
    max_patients: int,
    repetitions: int,
    check_times: ArrayLike,
    final_followup: float,
    true_medians: ArrayLike,
    seed: int,
    time_unit: str,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
    boundary_events: ArrayLike | None = None,
    max_total_work: int = 20_000,
    max_total_quadratures: int = 1_000,
    max_boundary_rows: int = 50,
    title: str = "Bayes Factor TTE analysis report",
) -> BayesFactorSurvivalReport:
    """Run bounded serial scenarios and create an immutable report snapshot.

    Simulation budgets are worst-case sums across all scenarios. Boundary rows
    are optional and limited by ``max_boundary_rows`` before any root solve;
    quadrature work per boundary row is adaptive and is not represented as an
    exact evaluation count.
    """
    unit = _label(time_unit, "time_unit", 40)
    report_title = _label(title, "title", 120)
    maximum_value = scalar(max_patients, "max_patients")
    repetitions_value = scalar(repetitions, "repetitions")
    rate = scalar(accrual_rate, "accrual_rate")
    followup = scalar(final_followup, "final_followup")
    work_value = scalar(max_total_work, "max_total_work")
    quadrature_value = scalar(max_total_quadratures, "max_total_quadratures")
    boundary_rows_value = scalar(max_boundary_rows, "max_boundary_rows")
    m0 = scalar(null_median, "null_median")
    m1 = scalar(alternative_median_mode, "alternative_median_mode")
    low = scalar(inferiority_cutoff, "inferiority_cutoff")
    high = scalar(superiority_cutoff, "superiority_cutoff")
    if (
        isinstance(max_patients, (bool, np.bool_))
        or isinstance(repetitions, (bool, np.bool_))
        or isinstance(seed, (bool, np.bool_))
        or isinstance(max_total_work, (bool, np.bool_))
        or isinstance(max_total_quadratures, (bool, np.bool_))
        or isinstance(max_boundary_rows, (bool, np.bool_))
        or maximum_value != int(maximum_value)
        or not 1 <= maximum_value <= 500
        or repetitions_value != int(repetitions_value)
        or not 1 <= repetitions_value <= 20_000
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
        or rate <= 0
        or followup < 0
        or not 0 < m0 < m1
        or not 0 <= low < high <= 1
        or work_value != int(work_value)
        or not 1 <= work_value <= _MAX_REPORT_WORK
        or quadrature_value != int(quadrature_value)
        or not 1 <= quadrature_value <= _MAX_REPORT_QUADRATURES
        or boundary_rows_value != int(boundary_rows_value)
        or not 1 <= boundary_rows_value <= _MAX_BOUNDARY_ROWS
    ):
        raise ValueError("invalid Bayes Factor TTE report inputs or resource limits")

    maximum = int(maximum_value)
    reps = int(repetitions_value)
    max_work = int(work_value)
    max_quadratures = int(quadrature_value)
    row_limit = int(boundary_rows_value)
    seed_int = int(seed)
    checks = _bounded_vector(check_times, "check_times", 1_000)
    if np.any(checks < 0) or np.any(np.diff(checks) <= 0):
        raise ValueError("check_times must be nonnegative and strictly increasing")
    truths = _bounded_vector(true_medians, "true_medians", _MAX_SCENARIOS)
    if truths.size == 0 or np.any(truths <= 0):
        raise ValueError("true_medians must contain 1..20 positive values")
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        truth_means = truths / np.log(2)
        gap_mean = 1.0 / rate
    if not np.all(np.isfinite(truth_means)) or not np.isfinite(gap_mean):
        raise ArithmeticError("scenario mean survival or accrual scale is not representable")
    scenario_count = int(truths.size)
    check_count = int(checks.size)
    aggregate_work = scenario_count * reps * (maximum + check_count + 1)
    aggregate_quadratures = scenario_count * reps * (check_count + 1)
    if aggregate_work > max_work:
        raise ValueError("aggregate worst-case scenario work exceeds max_total_work")
    if aggregate_quadratures > max_quadratures:
        raise ValueError("aggregate worst-case scenario evaluations exceed max_total_quadratures")

    boundary_values: NDArray[np.float64] | None = None
    if boundary_events is not None:
        boundary_values = _bounded_vector(boundary_events, "boundary_events", _MAX_BOUNDARY_ROWS)
        if boundary_values.size > row_limit:
            raise ValueError("boundary_events exceeds max_boundary_rows")
        if boundary_values.size > _MAX_BOUNDARY_ROWS or np.any(
            (boundary_values < 0)
            | (boundary_values > _MAX_BOUNDARY_EVENT)
            | (boundary_values != np.floor(boundary_values))
        ):
            raise ValueError("boundary_events must be integer counts in 0..500")

    # Validate model settings and compute optional boundaries before any RNG use.
    boundaries = (
        None
        if boundary_values is None
        else bayes_factor_survival_boundaries(
            count(boundary_values, "boundary_events"),
            null_median=m0,
            alternative_median_mode=m1,
            inferiority_cutoff=low,
            superiority_cutoff=high,
        )
    )

    children = np.random.SeedSequence(seed_int).spawn(scenario_count)
    summaries: list[BayesFactorSurvivalScenarioSummary] = []
    for true_median, true_mean, child in zip(truths, truth_means, children, strict=True):
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        result = simulate_bayes_factor_survival(
            null_median=m0,
            alternative_median_mode=m1,
            true_median=float(true_median),
            accrual_rate=rate,
            max_patients=maximum,
            repetitions=reps,
            check_times=checks,
            final_followup=followup,
            seed=child_seed,
            inferiority_cutoff=low,
            superiority_cutoff=high,
            max_total_work=max_work,
            max_total_quadratures=max_quadratures,
        )
        early_i = result.early_inferiority
        early_s = result.early_superiority
        terminal_i, terminal_s = _terminal_stop_flags(
            early_i,
            early_s,
            result.final_inferiority,
            result.final_superiority,
        )

        def estimate(values: NDArray[np.bool_]) -> tuple[float, float]:
            probability = float(np.mean(values))
            error = (
                float(np.sqrt(probability * (1 - probability) / reps)) if reps > 1 else float("nan")
            )
            return probability, error

        ti, ti_se = estimate(terminal_i)
        ts, ts_se = estimate(terminal_s)
        ei, ei_se = estimate(early_i)
        es, es_se = estimate(early_s)
        fi, fi_se = estimate(result.final_inferiority)
        fs, fs_se = estimate(result.final_superiority)
        quantiles = tuple(float(value) for value in result.patient_count_quantiles)
        summaries.append(
            BayesFactorSurvivalScenarioSummary(
                float(true_median),
                float(true_mean),
                child_seed,
                ti,
                ti_se,
                ts,
                ts_se,
                ei,
                ei_se,
                es,
                es_se,
                fi,
                fi_se,
                fs,
                fs_se,
                result.mean_patients_enrolled,
                (quantiles[0], quantiles[1], quantiles[2]),
            )
        )
        del result, early_i, early_s, terminal_i, terminal_s

    return BayesFactorSurvivalReport(
        m0,
        m1,
        low,
        high,
        rate,
        maximum,
        reps,
        tuple(float(value) for value in checks),
        followup,
        unit,
        seed_int,
        max_work,
        max_quadratures,
        row_limit,
        report_title,
        tuple(summaries),
        boundaries,
    )
