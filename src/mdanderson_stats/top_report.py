"""Bounded Python reports composed from the existing TOP APIs.

The HTML is a community output format; it does not claim parity with native
TOP protocol templates or report controls.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .top_binary import TOPBinaryDesign
from .top_endpoints import TOPMultiEndpointDesign
from .top_multi_simulation import TOPMultiEndpointSimulation, simulate_top_multiendpoint
from .top_simulation import simulate_top_binary
from .toxicity_timing import toxicity_time_quantile

_MAX_REPORT_SCENARIOS = 10
_MAX_REPORT_CELLS = 2_000_000
_BINARY_ACTIONS = ("success", "stop_futility")
_MULTI_ACTIONS = ("success", "stop_futility", "stop_toxicity", "stop_futility_toxicity")


def _label(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError("scenario labels must be nonempty strings of at most 120 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("scenario labels must not contain control characters")
    return value


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _array_text(value: ArrayLike) -> str:
    return ", ".join(format(float(item), ".17g") for item in np.asarray(value).ravel())


def _mean_mcse(values: ArrayLike) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    if scale == 0.0:
        return 0.0, 0.0 if array.size > 1 else None
    scaled = array / scale
    mean = float(np.mean(scaled) * scale)
    mcse = float(np.std(scaled, ddof=1) / np.sqrt(array.size) * scale) if array.size > 1 else None
    if not np.isfinite(mean) or (mcse is not None and not np.isfinite(mcse)):
        raise ValueError("summary mean or MCSE is not representable as a finite float")
    return mean, mcse


def _is_scalar_value(value: object) -> bool:
    return np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)


def _bounded_shape(value: ArrayLike, expected: tuple[int, ...], name: str) -> None:
    """Reject wrong list/array shapes before coercing values to NumPy arrays."""
    if isinstance(value, np.ndarray):
        actual = value.shape
    elif isinstance(value, (list, tuple)):
        if len(expected) == 1:
            actual = (
                expected
                if len(value) == expected[0] and all(_is_scalar_value(item) for item in value)
                else ()
            )
        elif len(value) == expected[0] and all(
            isinstance(row, (list, tuple, np.ndarray))
            and len(row) == expected[1]
            and all(_is_scalar_value(item) for item in row)
            for row in value
        ):
            actual = expected
        else:
            raise ValueError(f"{name} must have shape {expected}")
    else:
        raise ValueError(f"{name} must be a list, tuple, or NumPy array of shape {expected}")
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}")


def _preflight_scenarios(
    scenarios: Sequence[TOPBinaryScenario | TOPMultiEndpointScenario],
    max_subjects: int,
    cells_per_patient: int,
) -> tuple[str, ...]:
    if isinstance(scenarios, (str, bytes)) or not 1 <= len(scenarios) <= _MAX_REPORT_SCENARIOS:
        raise ValueError(f"reports accept 1..{_MAX_REPORT_SCENARIOS} named scenarios")
    labels = tuple(_label(scenario.label) for scenario in scenarios)
    if len(set(labels)) != len(labels):
        raise ValueError("scenario labels must be unique")
    total_cells = 0
    for scenario in scenarios:
        if isinstance(scenario.trials, (bool, np.bool_)) or not isinstance(
            scenario.trials, Integral
        ):
            raise ValueError("trials must be a positive integer")
        if not 1 <= int(scenario.trials) <= 100_000:
            raise ValueError("trials must be in [1,100000]")
        if isinstance(scenario.seed, (bool, np.bool_)) or not isinstance(scenario.seed, Integral):
            raise ValueError("saved reports require an explicit integer seed")
        if not 0 <= int(scenario.seed) <= np.iinfo(np.uint64).max:
            raise ValueError("saved report seeds must be in [0, 2**64-1]")
        total_cells += int(scenario.trials) * max_subjects * cells_per_patient
    if total_cells > _MAX_REPORT_CELLS:
        raise ValueError("aggregate report simulation exceeds the 2,000,000-cell limit")
    return labels


@dataclass(frozen=True)
class TOPBinaryScenario:
    """Named single-binary truth and calendar simulation inputs."""

    label: str
    response_probability: float
    window: float
    accrual_rate: float
    trials: int
    seed: int
    arrival: str = "exponential"
    response_distribution: str = "uniform"
    late_probability: float | None = None


@dataclass(frozen=True)
class TOPMultiEndpointScenario:
    """Named joint-outcome truth and calendar simulation inputs."""

    label: str
    joint_probabilities: tuple[float, float, float, float]
    accrual_rate: float
    trials: int
    seed: int
    arrival: str = "exponential"
    truth_timing_probabilities: (
        tuple[float, float, float]
        | tuple[tuple[float, float, float], tuple[float, float, float]]
        | None
    ) = None


@dataclass(frozen=True)
class TOPReportAction:
    name: str
    count: int
    probability: float
    mcse: float
    denominator: int


@dataclass(frozen=True)
class TOPReportTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TOPReportCase:
    label: str
    inputs: tuple[tuple[str, str], ...]
    actions: tuple[TOPReportAction, ...]
    summaries: tuple[tuple[str, str, str], ...]
    trials: int
    seed: int
    method: str


@dataclass(frozen=True)
class TOPReport:
    """Immutable static report from a TOP design and named scenario batch."""

    endpoint: str
    settings: tuple[tuple[str, str], ...]
    boundaries: TOPReportTable
    cases: tuple[TOPReportCase, ...]
    limitations: str = (
        "Python community report built from this package's TOP APIs. Cached source material "
        "does not include a native protocol-template file or report schema; native file, "
        "control, scheduling, and RNG parity are not claimed."
    )

    def to_html(self) -> str:
        settings_rows = "".join(
            f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>"
            for name, value in self.settings
        )
        boundary_head = "".join(f"<th>{escape(name)}</th>" for name in self.boundaries.columns)
        boundary_rows = "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
            for row in self.boundaries.rows
        )
        sections = []
        for case in self.cases:
            inputs = "".join(
                f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>"
                for name, value in case.inputs
            )
            actions = "".join(
                "<tr>"
                f"<th>{escape(action.name)}</th><td>{action.count}</td>"
                f"<td>{_number(action.probability)}</td><td>{_number(action.mcse)}</td>"
                f"<td>{action.denominator}</td></tr>"
                for action in case.actions
            )
            summaries = "".join(
                f"<tr><th>{escape(name)}</th><td>{escape(value)}</td>"
                f"<td>{escape(uncertainty)}</td></tr>"
                for name, value, uncertainty in case.summaries
            )
            sections.append(
                f"<section><h2>{escape(case.label)}</h2>"
                f"<p><strong>Method:</strong> {escape(case.method)}; "
                f"<strong>trials:</strong> {case.trials}; <strong>seed:</strong> {case.seed}</p>"
                f"<table><caption>Scenario inputs</caption><tbody>{inputs}</tbody></table>"
                "<table><caption>Terminal actions; MCSE is the binomial Monte Carlo SE across "
                "independent simulated trials</caption>"
                "<thead><tr><th>Action</th><th>Count</th><th>Probability</th><th>MCSE</th>"
                f"<th>Denominator</th></tr></thead><tbody>{actions}</tbody></table>"
                "<table><caption>Enrollment and calendar summaries (mean; MCSE)</caption>"
                f"<tbody>{summaries}</tbody></table></section>"
            )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>TOP {escape(self.endpoint)} report</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1150px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.4em .6em;text-align:left}"
            "th{background:#f4f6f8}</style></head><body>"
            f"<h1>TOP {escape(self.endpoint)} report</h1>"
            f"<p>{escape(self.limitations)}</p>"
            "<table><caption>Captured design and prior</caption>"
            f"<tbody>{settings_rows}</tbody></table>"
            f"<table><caption>{escape(self.boundaries.title)}</caption>"
            f"<thead><tr>{boundary_head}</tr></thead><tbody>{boundary_rows}</tbody></table>"
            + "".join(sections)
            + "</body></html>\n"
        )

    def write_html(self, path: str | os.PathLike[str]) -> Path:
        """Atomically write HTML in an existing directory."""
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


def _action_rows(decisions: ArrayLike, labels: tuple[str, ...]) -> tuple[TOPReportAction, ...]:
    values = np.asarray(decisions)
    trials = int(values.size)
    rows = []
    for label in labels:
        count = int(np.count_nonzero(values == label))
        probability = count / trials
        mcse = float(np.sqrt(probability * (1 - probability) / trials))
        rows.append(TOPReportAction(label, count, probability, mcse, trials))
    return tuple(rows)


def _summary_rows(values: Sequence[tuple[str, ArrayLike]]) -> tuple[tuple[str, str, str], ...]:
    rows = []
    for name, array in values:
        mean, mcse = _mean_mcse(array)
        rows.append((name, _number(mean), _number(mcse)))
    return tuple(rows)


def _binary_boundary_table(design: TOPBinaryDesign) -> TOPReportTable:
    boundary = design.boundaries()
    rows = []
    for look_index, raw_look in enumerate(boundary.patients):
        look = int(raw_look)
        roots = boundary.futility_effective_size[look_index, : look + 1]
        rows.append(
            (
                str(look),
                str(int(boundary.complete_go_min[look_index])),
                str(int(boundary.suspend_pending_min[look_index])),
                _array_text(roots),
            )
        )
    return TOPReportTable(
        "Exact design boundaries (futility crossings; stop strictly above crossing)",
        (
            "look n",
            "complete-data response minimum",
            "suspend at pending count",
            "ESS futility crossing by r",
        ),
        tuple(rows),
    )


def _multi_boundary_table(design: TOPMultiEndpointDesign) -> TOPReportTable:
    boundary = design.boundaries()
    rows = []
    for look_index, raw_look in enumerate(boundary.patients):
        look = int(raw_look)
        for endpoint in range(2):
            roots = boundary.effective_size_crossing[look_index, endpoint, : look + 1]
            rows.append(
                (
                    str(look),
                    "efficacy"
                    if endpoint == 0
                    else ("toxicity" if design.mode == "efficacy_toxicity" else "endpoint 2"),
                    str(int(boundary.complete_event_threshold[look_index, endpoint])),
                    str(int(boundary.suspend_pending_min[look_index])),
                    _array_text(roots),
                )
            )
    return TOPReportTable(
        "Exact endpoint boundaries (crossings by event count; evaluate() determines actions)",
        (
            "look n",
            "endpoint",
            "complete-data event threshold",
            "suspend at pending count",
            "ESS crossing by events",
        ),
        tuple(rows),
    )


def _binary_settings(design: TOPBinaryDesign) -> tuple[tuple[str, str], ...]:
    assert design.prior is not None and design.looks is not None
    assert design.timing_probabilities is not None
    return (
        ("maximum patients", str(design.max_subjects)),
        ("null response rate", _number(design.null_rate)),
        ("cutoff scale C", _number(design.cutoff_scale)),
        ("gamma", _number(design.gamma)),
        ("Beta prior shapes", _array_text(design.prior)),
        ("looks", _array_text(design.looks)),
        ("suspension convention", design.suspension),
        (
            "analysis timing mixture by response-window third",
            _array_text(design.timing_probabilities),
        ),
    )


def _multi_settings(design: TOPMultiEndpointDesign) -> tuple[tuple[str, str], ...]:
    assert design.looks is not None and design.timing_probabilities is not None
    return (
        ("maximum patients", str(design.max_subjects)),
        ("mode", design.mode),
        ("null joint probabilities (11,10,01,00)", _array_text(design.null_joint_probabilities)),
        ("prior concentration", _number(design.prior_concentration)),
        ("marginal Beta priors", _array_text(design.marginal_prior)),
        ("marginal null thresholds", _array_text(design.marginal_null)),
        ("cutoff scale C", _number(design.cutoff_scale)),
        ("gamma", _number(design.gamma)),
        ("assessment windows by endpoint", _array_text(design.windows)),
        ("looks", _array_text(design.looks)),
        ("suspension convention", design.suspension),
        (
            "analysis timing mixture by endpoint and response-window third",
            _array_text(design.timing_probabilities),
        ),
    )


def top_binary_report(design: TOPBinaryDesign, scenarios: Sequence[TOPBinaryScenario]) -> TOPReport:
    """Simulate named binary truths and save design boundaries with summaries."""
    if not isinstance(design, TOPBinaryDesign):
        raise TypeError("design must be a TOPBinaryDesign")
    labels = _preflight_scenarios(scenarios, design.max_subjects, 1)
    prepared: list[tuple[float, float, float, str, str, float | None, int, int]] = []
    for scenario in scenarios:
        probability = scalar(scenario.response_probability, "response_probability")
        window = scalar(scenario.window, "window")
        rate = scalar(scenario.accrual_rate, "accrual_rate")
        if not 0 <= probability <= 1 or window <= 0 or rate <= 0 or not np.isfinite(1 / rate):
            raise ValueError(
                "require response probability in [0,1] and representable positive window/rate"
            )
        if scenario.arrival not in ("fixed", "exponential"):
            raise ValueError("arrival must be 'fixed' or 'exponential'")
        toxicity_time_quantile(
            0.5,
            probability,
            window,
            distribution=scenario.response_distribution,
            late_probability=scenario.late_probability,
        )
        late = (
            None
            if scenario.late_probability is None
            else scalar(scenario.late_probability, "late_probability")
        )
        prepared.append(
            (
                probability,
                window,
                rate,
                scenario.arrival,
                scenario.response_distribution,
                late,
                int(scenario.trials),
                int(scenario.seed),
            )
        )
    cases = []
    for scenario, label, settings in zip(scenarios, labels, prepared):
        probability, window, rate, arrival, response_distribution, late, trials, seed = settings
        simulation = simulate_top_binary(
            design,
            probability,
            window,
            rate,
            trials=trials,
            arrival=arrival,
            response_distribution=response_distribution,
            late_probability=late,
            rng=seed,
        )
        cases.append(
            TOPReportCase(
                label,
                (
                    ("true response probability", _number(probability)),
                    ("response window", _number(window)),
                    ("accrual rate", _number(rate)),
                    ("arrival gaps", arrival),
                    ("response-time truth model", response_distribution),
                    ("late-half probability (if applicable)", _number(late)),
                ),
                _action_rows(simulation.decision, _BINARY_ACTIONS),
                _summary_rows(
                    (
                        ("mean enrollment", simulation.patients),
                        ("mean observed responses at terminal decision", simulation.responses),
                        ("mean pending at terminal decision", simulation.pending),
                        ("mean total duration", simulation.duration),
                        ("mean interim accrual suspension", simulation.suspension_time),
                        ("mean final follow-up wait", simulation.final_followup_time),
                    )
                ),
                trials,
                seed,
                "Existing simulate_top_binary engine; independent Monte Carlo trials",
            )
        )
    return TOPReport(
        "binary efficacy", _binary_settings(design), _binary_boundary_table(design), tuple(cases)
    )


def _joint_probabilities(value: ArrayLike) -> tuple[float, float, float, float]:
    _bounded_shape(value, (4,), "joint_probabilities (11,10,01,00)")
    probabilities = finite(value, "joint_probabilities")
    total = float(np.sum(probabilities))
    if (
        np.any(probabilities < 0)
        or not np.isfinite(total)
        or abs(total - 1) > 32 * np.finfo(float).eps
    ):
        raise ValueError("joint_probabilities must be nonnegative and sum to one")
    return (
        float(probabilities[0] / total),
        float(probabilities[1] / total),
        float(probabilities[2] / total),
        float(probabilities[3] / total),
    )


def _truth_timing(
    value: ArrayLike | None, design: TOPMultiEndpointDesign
) -> tuple[tuple[float, ...], ...]:
    if value is None:
        timing = np.asarray(design.timing_probabilities)
    else:
        if isinstance(value, np.ndarray):
            shape = value.shape
        elif (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(_is_scalar_value(item) for item in value)
        ):
            shape = (3,)
        elif (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(row, (list, tuple, np.ndarray))
                and len(row) == 3
                and all(_is_scalar_value(item) for item in row)
                for row in value
            )
        ):
            shape = (2, 3)
        else:
            shape = ()
        if shape not in ((3,), (2, 3)):
            raise ValueError("truth_timing_probabilities must be a triple or a 2-by-3 matrix")
        timing = finite(value, "truth_timing_probabilities").copy()
        if timing.shape == (3,):
            timing = np.broadcast_to(timing, (2, 3)).copy()
        if np.any(timing < 0) or np.any(np.abs(timing.sum(axis=1) - 1) > 32 * np.finfo(float).eps):
            raise ValueError("each endpoint timing triple must be nonnegative and sum to one")
        timing /= timing.sum(axis=1, keepdims=True)
    return tuple(tuple(float(item) for item in row) for row in timing)


def top_multiendpoint_report(
    design: TOPMultiEndpointDesign, scenarios: Sequence[TOPMultiEndpointScenario]
) -> TOPReport:
    """Simulate named joint truths in co-primary or efficacy/toxicity mode."""
    if not isinstance(design, TOPMultiEndpointDesign):
        raise TypeError("design must be a TOPMultiEndpointDesign")
    labels = _preflight_scenarios(scenarios, design.max_subjects, 2)
    prepared: list[
        tuple[
            tuple[float, float, float, float], float, str, tuple[tuple[float, ...], ...], int, int
        ]
    ] = []
    for scenario in scenarios:
        probabilities = _joint_probabilities(scenario.joint_probabilities)
        if scenario.arrival not in ("fixed", "exponential"):
            raise ValueError("arrival must be 'fixed' or 'exponential'")
        rate = scalar(scenario.accrual_rate, "accrual_rate")
        if rate <= 0 or not np.isfinite(1 / rate):
            raise ValueError("accrual_rate must be positive with representable mean gap")
        truth_timing = _truth_timing(scenario.truth_timing_probabilities, design)
        prepared.append(
            (
                probabilities,
                rate,
                scenario.arrival,
                truth_timing,
                int(scenario.trials),
                int(scenario.seed),
            )
        )
    cases = []
    for scenario, label, settings in zip(scenarios, labels, prepared):
        probabilities, rate, arrival, truth_timing, trials, seed = settings
        simulation: TOPMultiEndpointSimulation = simulate_top_multiendpoint(
            design,
            probabilities,
            rate,
            trials=trials,
            arrival=arrival,
            truth_timing_probabilities=truth_timing,
            rng=seed,
        )
        inputs = (
            ("joint truth probabilities (11,10,01,00)", _array_text(probabilities)),
            ("accrual rate", _number(rate)),
            ("arrival gaps", arrival),
            ("truth timing by endpoint and window third", _array_text(truth_timing)),
        )
        cases.append(
            TOPReportCase(
                label,
                inputs,
                _action_rows(simulation.decision, _MULTI_ACTIONS),
                _summary_rows(
                    (
                        ("mean enrollment", simulation.patients),
                        ("mean efficacy events at decision", simulation.events[:, 0]),
                        (
                            "mean toxicity events at decision"
                            if design.mode == "efficacy_toxicity"
                            else "mean endpoint-2 events at decision",
                            simulation.events[:, 1],
                        ),
                        ("mean endpoint-1 pending at decision", simulation.pending[:, 0]),
                        ("mean endpoint-2 pending at decision", simulation.pending[:, 1]),
                        ("mean total duration", simulation.duration),
                        ("mean interim accrual suspension", simulation.interim_suspension_time),
                        ("mean final follow-up wait", simulation.final_followup_time),
                    )
                ),
                trials,
                seed,
                "Existing simulate_top_multiendpoint engine; independent Monte Carlo trials",
            )
        )
    endpoint_name = "co-primary efficacy" if design.mode == "coprimary" else "efficacy/toxicity"
    return TOPReport(
        endpoint_name, _multi_settings(design), _multi_boundary_table(design), tuple(cases)
    )
