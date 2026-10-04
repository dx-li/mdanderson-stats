"""Reproducible, bounded BOP2 design and operating-characteristic reports."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .bayesian_monitoring import BayesianMonitoringDesign, _owned
from .bop2_binary import _success as _binary_success
from .bop2_binary import bop2_binary_design
from .bop2_efftox import bop2_efftox_design
from .bop2_paired import BOP2PairedDesign, _cells, bop2_paired_design
from .bop2_survival import bop2_survival_design
from .bop2_survival_trial import simulate_bop2_survival

_MAX_SCENARIOS = 20
_MAX_EXACT_WORK = 50_000_000
_MAX_SURVIVAL_PATIENT_TRIALS = 10_000_000


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValueError(f"{name} must be a nonempty string of at most 100 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float | int | None) -> str:
    return "—" if value is None else format(float(value), ".17g")


def _probability(value: float, name: str, *, open_interval: bool = False) -> float:
    result = scalar(value, name)
    invalid = (result <= 0 or result >= 1) if open_interval else (result < 0 or result > 1)
    if invalid:
        interval = "(0,1)" if open_interval else "[0,1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _positive_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return result


def _probability_vector(value: ArrayLike, name: str, size: int) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 1
            or value.size != size
            or np.iscomplexobj(value)
            or value.dtype.kind == "b"
        ):
            raise ValueError(f"{name} must be a real vector of length {size}")
    elif isinstance(value, (tuple, list)):
        if len(value) != size or any(
            isinstance(item, (tuple, list, np.ndarray, complex, np.complexfloating, bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"{name} must be a real vector of length {size}")
    else:
        raise ValueError(f"{name} must be a bounded real vector of length {size}")
    array = finite(value, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} entries must lie in [0,1]")
    return tuple(float(item) for item in array)


def _probability_vector_allow_positive(value: ArrayLike, name: str, size: int) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 1
            or value.size != size
            or np.iscomplexobj(value)
            or value.dtype.kind == "b"
        ):
            raise ValueError(f"{name} must be a real vector of length {size}")
    elif isinstance(value, (tuple, list)):
        if len(value) != size or any(
            isinstance(item, (tuple, list, np.ndarray, complex, np.complexfloating, bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"{name} must be a real vector of length {size}")
    else:
        raise ValueError(f"{name} must be a bounded real vector of length {size}")
    array = finite(value, name)
    return tuple(float(item) for item in array)


def _positive_vector(value: ArrayLike | None, name: str, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = _probability_vector_allow_positive(value, name, size)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} entries must be positive")
    return result


def _schedule(value: ArrayLike | None, maximum: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 1
            or value.size > maximum
            or np.iscomplexobj(value)
            or value.dtype.kind == "b"
        ):
            raise ValueError("looks must be a bounded one-dimensional integer vector")
    elif isinstance(value, (list, tuple)):
        if len(value) > maximum or any(
            isinstance(item, (list, tuple, np.ndarray, bool, np.bool_, complex, np.complexfloating))
            for item in value
        ):
            raise ValueError("looks must be a bounded one-dimensional integer vector")
    else:
        raise ValueError("looks must be a bounded one-dimensional integer vector")
    values = finite(value, "looks")
    if values.ndim != 1 or not values.size or np.any(values != np.floor(values)):
        raise ValueError("looks must be a nonempty vector of integers")
    return tuple(int(x) for x in values)


@dataclass(frozen=True, slots=True)
class BOP2ProtocolScenario:
    """One named truth; probabilities use the endpoint-specific documented order."""

    label: str
    probabilities: ArrayLike | float
    joint_probability: float | None = None
    _rates: tuple[float, ...] = ()
    _joint: float | None = None

    def __post_init__(self) -> None:
        label = _label(self.label, "scenario label")
        raw = self.probabilities
        rates: tuple[float, ...]
        if isinstance(raw, (float, int, np.floating, np.integer)) and not isinstance(
            raw, (bool, np.bool_)
        ):
            rates = (_probability(float(raw), "scenario probability"),)
        else:
            if isinstance(raw, np.ndarray):
                size = raw.size if raw.ndim == 1 else -1
            elif isinstance(raw, (list, tuple)):
                size = len(raw)
            else:
                size = -1
            if size not in (1, 2):
                raise ValueError(
                    "scenario probabilities must be a scalar or a one/two-element vector"
                )
            rates = _probability_vector(raw, "scenario probabilities", size)
        joint = (
            None
            if self.joint_probability is None
            else _probability(self.joint_probability, "joint_probability")
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "probabilities", _owned(rates) if len(rates) > 1 else rates[0])
        object.__setattr__(self, "_rates", rates)
        object.__setattr__(self, "_joint", joint)


@dataclass(frozen=True, slots=True)
class BOP2ProtocolScenarioSummary:
    """Compact immutable OC summary for one captured truth scenario."""

    label: str
    truth_values: tuple[float, ...]
    category_probabilities: tuple[float, ...] | None
    success_probability: float
    success_mcse: float | None
    stop_probability_by_look: tuple[float, ...]
    sample_size_probability: tuple[float, ...]
    expected_sample_size: float
    sample_size_sd: float
    details: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class BOP2ProtocolReport:
    """Static report snapshot; it neither reruns nor restores a design session."""

    endpoint: str
    max_subjects: int
    settings: tuple[tuple[str, str], ...]
    boundary_columns: tuple[str, ...]
    boundary_rows: tuple[tuple[str, ...], ...]
    scenarios: tuple[BOP2ProtocolScenarioSummary, ...]
    provenance: str = "Specified design; this report does not claim an optimized design."
    decision_rule: str = ""

    def to_html(self) -> str:
        settings_rows = "".join(
            f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"
            for key, value in self.settings
        )
        boundary_head = "".join(f"<th>{escape(item)}</th>" for item in self.boundary_columns)
        boundary_body = "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
            for row in self.boundary_rows
        )
        scenario_parts: list[str] = []
        for scenario in self.scenarios:
            truth = escape(", ".join(_number(x) for x in scenario.truth_values))
            category = (
                "—"
                if scenario.category_probabilities is None
                else ", ".join(_number(x) for x in scenario.category_probabilities)
            )
            stop = ", ".join(_number(x) for x in scenario.stop_probability_by_look)
            pmf = ", ".join(_number(x) for x in scenario.sample_size_probability)
            details = "".join(
                f"<tr><th>{escape(key)}</th><td>{_number(value)}</td></tr>"
                for key, value in scenario.details
            )
            scenario_parts.append(
                "<section><h3>" + escape(scenario.label) + "</h3><table><tbody>"
                f"<tr><th>Truth values</th><td>{truth}</td></tr>"
                f"<tr><th>Category probabilities</th><td>{escape(category)}</td></tr>"
                "<tr><th>Success probability</th>"
                f"<td>{_number(scenario.success_probability)}</td></tr>"
                f"<tr><th>Success Monte Carlo SE</th><td>{_number(scenario.success_mcse)}</td></tr>"
                f"<tr><th>Stop probability by look</th><td>{escape(stop)}</td></tr>"
                f"<tr><th>Sample-size probability by look</th><td>{escape(pmf)}</td></tr>"
                "<tr><th>Expected sample size</th>"
                f"<td>{_number(scenario.expected_sample_size)}</td></tr>"
                f"<tr><th>Sample-size SD</th><td>{_number(scenario.sample_size_sd)}</td></tr>"
                f"{details}</tbody></table></section>"
            )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>BOP2 {escape(self.endpoint)} protocol report</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2em auto;"
            "padding:0 1em;color:#202124}"
            "table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.4em .6em;text-align:left}"
            "th{background:#f4f6f8}</style></head><body>"
            f"<h1>BOP2 {escape(self.endpoint)} protocol report</h1>"
            f"<p>{escape(self.provenance)}</p><p>{escape(self.decision_rule)}</p>"
            f"<p>Maximum enrollment: {self.max_subjects}</p>"
            "<table><caption>Captured design settings</caption>"
            f"<tbody>{settings_rows}</tbody></table>"
            "<table><caption>Decision boundaries from the constructed design</caption>"
            f"<thead><tr>{boundary_head}</tr></thead><tbody>{boundary_body}</tbody></table>"
            "<h2>Operating characteristics</h2>"
            + "".join(scenario_parts)
            + "<p>Exact OC rows use the existing Python recursion. Survival rows use Monte Carlo; "
            "the standard error is descriptive and does not imply control guarantees. This static "
            "report is not a native Word/Chinese report or animation.</p>"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        destination = Path(path)
        content = self.to_html()
        temp: Path | None = None
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
                temp = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, destination)
            temp = None
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return destination


def _settings(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple((key, repr(value)) for key, value in values.items())


def _look_rows(design: BayesianMonitoringDesign) -> tuple[tuple[str, ...], ...]:
    rows = []
    for i, n in enumerate(design.looks):
        rows.append(
            (str(int(n)), str(int(design.futility_max[i])), str(int(design.positive_min[i])))
            if n < design.max_subjects
            else (str(int(n)), "final", str(int(design.final_positive_min)))
        )
    return tuple(rows)


def _binary_summary(
    design: BayesianMonitoringDesign, scenario: BOP2ProtocolScenario
) -> BOP2ProtocolScenarioSummary:
    rate = scenario._rates[0]
    oc = design.operating_characteristics(rate)
    success = float(_binary_success(design, oc))
    pmf = tuple(float(x) for x in oc.sample_size_probability)
    if design.method == "bop2_binary_toxicity":
        failure_by_look = oc.stop_high.copy()
        failure_by_look[-1] += float(oc.complete_positive)
    else:
        failure_by_look = oc.stop_low.copy()
        failure_by_look[-1] += float(oc.complete_negative)
    stops = tuple(float(x) for x in failure_by_look)
    details = (
        ("Futility stop probability", float(oc.stop_low.sum())),
        ("High-event stop probability", float(oc.stop_high.sum())),
        ("Expected events", float(oc.expected_events)),
        ("Expected observed event rate", float(oc.expected_observed_rate)),
    )
    return BOP2ProtocolScenarioSummary(
        scenario.label,
        (rate,),
        None,
        success,
        None,
        stops,
        pmf,
        float(oc.expected_sample_size),
        float(oc.sample_size_sd),
        details,
    )


def bop2_binary_efficacy_report(
    max_subjects: int,
    null_rate: float,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    *,
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Build an exact OC report for specified binary efficacy truth rates."""
    return _binary_report(
        "binary efficacy",
        "efficacy",
        max_subjects,
        null_rate,
        scenarios,
        cutoff_scale,
        gamma,
        looks,
        prior,
        min_subjects,
        cohort_size,
    )


def bop2_binary_toxicity_report(
    max_subjects: int,
    null_rate: float,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    *,
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Build an exact OC report; scenario probabilities are adverse-event rates."""
    return _binary_report(
        "binary toxicity",
        "toxicity",
        max_subjects,
        null_rate,
        scenarios,
        cutoff_scale,
        gamma,
        looks,
        prior,
        min_subjects,
        cohort_size,
    )


def _binary_report(
    endpoint: str,
    method: str,
    max_subjects: int,
    null_rate: float,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None,
    prior: ArrayLike | None,
    min_subjects: int,
    cohort_size: int,
) -> BOP2ProtocolReport:
    n = _positive_integer(max_subjects, "max_subjects", 200)
    captured = _scenario_tuple(scenarios, 1)
    if len(captured) * (n + 1) ** 2 > _MAX_EXACT_WORK:
        raise ValueError("binary report exceeds the exact-work budget")
    p0 = _probability(null_rate, "null_rate", open_interval=True)
    scale, exponent = scalar(cutoff_scale, "cutoff_scale"), scalar(gamma, "gamma")
    schedule = _schedule(looks, n)
    prior_values = _positive_vector(prior, "prior", 2)
    design = bop2_binary_design(
        n,
        p0,
        cutoff_scale=scale,
        gamma=exponent,
        endpoint=method,
        looks=schedule,
        prior=prior_values,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    summary = tuple(_binary_summary(design, s) for s in captured)
    settings = _settings(
        null_rate=p0,
        cutoff_scale=scale,
        gamma=exponent,
        prior_input=prior_values,
        effective_prior=design.prior,
        looks=tuple(int(x) for x in design.looks),
        min_subjects=min_subjects,
        cohort_size=cohort_size,
        endpoint=method,
        max_subjects=n,
    )
    return BOP2ProtocolReport(
        endpoint,
        n,
        settings,
        ("Look N", "Maximum futility events", "Minimum high-event events"),
        _look_rows(design),
        summary,
        decision_rule=(
            "Binary efficacy stops for posterior futility at scheduled looks and "
            "makes its positive/negative conclusion at N; equality with the "
            "posterior cutoff continues."
            if method == "efficacy"
            else "Toxicity rates count adverse events. A low posterior safety "
            "probability stops for unsafe toxicity; safe success is the "
            "complete-negative conclusion at N. Equality with the cutoff continues."
        ),
    )


def _scenario_tuple(
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario], rate_count: int
) -> tuple[BOP2ProtocolScenario, ...]:
    if not isinstance(scenarios, (tuple, list)) or not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} entries")
    if any(
        not isinstance(s, BOP2ProtocolScenario) or len(s._rates) != rate_count for s in scenarios
    ):
        raise ValueError(f"each scenario must have {rate_count} truth rate(s)")
    if len({s.label for s in scenarios}) != len(scenarios):
        raise ValueError("scenario labels must be unique")
    return tuple(scenarios)


def _paired_summary(
    design: BOP2PairedDesign, scenario: BOP2ProtocolScenario, category: tuple[float, ...]
) -> BOP2ProtocolScenarioSummary:
    oc = design.operating_characteristics(category)
    pmf = tuple(float(x) for x in oc.sample_size_probability)
    return BOP2ProtocolScenarioSummary(
        scenario.label,
        scenario._rates,
        category,
        float(oc.success_probability),
        None,
        tuple(float(x) for x in oc.stop_probability),
        pmf,
        float(oc.expected_sample_size),
        float(oc.sample_size_sd),
    )


def _paired_report(
    endpoint: Literal["ordinal", "multiple"],
    max_subjects: int,
    null_rates: ArrayLike,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    *,
    cutoff_scale: float,
    gamma: float,
    null_joint_rate: float | None,
    looks: ArrayLike | None,
    prior: ArrayLike | None,
    min_subjects: int,
    cohort_size: int,
) -> BOP2ProtocolReport:
    n = _positive_integer(max_subjects, "max_subjects", 200)
    captured = _scenario_tuple(scenarios, 2)
    category_count = 3 if endpoint == "ordinal" else 4
    forward_cells = category_count * len(captured) * (n * (n + 1) * (2 * n + 1) // 6)
    if forward_cells > _MAX_EXACT_WORK:
        raise ValueError("paired report exceeds the exact-work budget")
    rates = _probability_vector(null_rates, "null_rates", 2)
    categories = [tuple(float(x) for x in _cells(s._rates, s._joint, endpoint)) for s in captured]
    schedule = _schedule(looks, n)
    prior_values = _positive_vector(prior, "prior", category_count)
    scale, exponent = scalar(cutoff_scale, "cutoff_scale"), scalar(gamma, "gamma")
    design = bop2_paired_design(
        n,
        rates,
        cutoff_scale=scale,
        gamma=exponent,
        endpoint=endpoint,
        null_joint_rate=null_joint_rate,
        looks=schedule,
        prior=prior_values,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    summaries = []
    for scenario, cells in zip(captured, categories):
        summaries.append(_paired_summary(design, scenario, cells))
    settings = _settings(
        null_rates=rates,
        null_joint_rate=null_joint_rate,
        cutoff_scale=scale,
        gamma=exponent,
        prior_input=prior_values,
        effective_prior=tuple(float(x) for x in design.prior),
        looks=tuple(int(x) for x in design.looks),
        endpoint=endpoint,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
        max_subjects=n,
    )
    rows = tuple(
        (str(int(nlook)), str(int(design.futility_max[i, 0])), str(int(design.futility_max[i, 1])))
        if nlook < n
        else (
            str(int(nlook)),
            str(int(design.marginals[0].final_positive_min - 1)),
            str(int(design.marginals[1].final_positive_min - 1)),
        )
        for i, nlook in enumerate(design.looks)
    )
    columns = ("Look N", "Maximum endpoint-1 futility events", "Maximum endpoint-2 futility events")
    rule = (
        "The exact recursion stops for futility only when both marginal criteria fail; "
        "either endpoint passing avoids futility. At the final analysis, either "
        "marginal success criterion is sufficient for a positive conclusion."
    )
    return BOP2ProtocolReport(
        endpoint, n, settings, columns, rows, tuple(summaries), decision_rule=rule
    )


def bop2_ordinal_report(
    max_subjects: int,
    null_rates: ArrayLike,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    *,
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Exact OC report; rates are CR and CR+PR, category order CR/PR/other."""
    return _paired_report(
        "ordinal",
        max_subjects,
        null_rates,
        scenarios,
        cutoff_scale=cutoff_scale,
        gamma=gamma,
        null_joint_rate=None,
        looks=looks,
        prior=prior,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )


def bop2_multiple_report(
    max_subjects: int,
    null_rates: ArrayLike,
    scenarios: tuple[BOP2ProtocolScenario, ...] | list[BOP2ProtocolScenario],
    *,
    cutoff_scale: float,
    gamma: float,
    null_joint_rate: float,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Exact OC report; rates are endpoint 1/2 and explicit joint probability."""
    return _paired_report(
        "multiple",
        max_subjects,
        null_rates,
        scenarios,
        cutoff_scale=cutoff_scale,
        gamma=gamma,
        null_joint_rate=null_joint_rate,
        looks=looks,
        prior=prior,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )


def bop2_efftox_report(
    max_subjects: int,
    null_rates: ArrayLike,
    alternative_rates: ArrayLike,
    *,
    cutoff_scales: ArrayLike,
    gamma: float,
    joint_rates: ArrayLike | None = None,
    efficacy_looks: ArrayLike | None = None,
    toxicity_looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    toxicity_exponent_factor: float = 1 / 3,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Exact four-scenario EffTox OC report in H00/H01/H10/H11 order."""
    n = _positive_integer(max_subjects, "max_subjects", 200)
    null = _probability_vector(null_rates, "null_rates", 2)
    alt = _probability_vector(alternative_rates, "alternative_rates", 2)
    joints = None if joint_rates is None else _probability_vector(joint_rates, "joint_rates", 4)
    if 4 * 4 * (n * (n + 1) * (2 * n + 1) // 6) > _MAX_EXACT_WORK:
        raise ValueError("EffTox report exceeds the exact-work budget")
    scenario_specs = (
        ("H00", null, 0),
        ("H01", (null[0], alt[1]), 1),
        ("H10", (alt[0], null[1]), 2),
        ("H11", alt, 3),
    )
    scenarios = []
    for label, pair, index in scenario_specs:
        joint = None if joints is None else joints[index]
        if joint is None:
            joint = pair[0] * pair[1]
        cells = _cells(pair, joint, "multiple")
        scenarios.append((label, pair, joint, tuple(float(x) for x in cells)))
    scale_values = _probability_vector(cutoff_scales, "cutoff_scales", 2)
    scale_values = tuple(
        _probability(value, "cutoff_scale", open_interval=True) for value in scale_values
    )
    look_e = _schedule(efficacy_looks, n)
    look_t = _schedule(toxicity_looks, n)
    prior_values = _positive_vector(prior, "prior", 4)
    design = bop2_efftox_design(
        n,
        null,
        cutoff_scales=scale_values,
        gamma=gamma,
        null_joint_rate=None if joints is None else joints[0],
        efficacy_looks=look_e,
        toxicity_looks=look_t,
        prior=prior_values,
        toxicity_exponent_factor=toxicity_exponent_factor,
        equality_continues=equality_continues,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    summaries = tuple(
        _paired_summary(design, BOP2ProtocolScenario(label, pair, joint), cells)
        for label, pair, joint, cells in scenarios
    )
    settings = _settings(
        null_rates=null,
        alternative_rates=alt,
        joint_rates=joints,
        cutoff_scales=scale_values,
        gamma=scalar(gamma, "gamma"),
        toxicity_exponent_factor=scalar(toxicity_exponent_factor, "toxicity_exponent_factor"),
        equality_continues=equality_continues,
        prior_input=prior_values,
        effective_prior=tuple(float(x) for x in design.prior),
        efficacy_looks=tuple(int(x) for x in design.marginals[0].looks),
        toxicity_looks=tuple(int(x) for x in design.marginals[1].looks),
        min_subjects=min_subjects,
        cohort_size=cohort_size,
        max_subjects=n,
    )
    rows = tuple(
        (str(int(look)), str(int(design.futility_max[i, 0])), str(int(design.toxicity_min[i])))
        if look < n
        else (
            str(int(look)),
            str(int(design.marginals[0].final_positive_min - 1)),
            str(int(design.marginals[1].final_positive_min)),
        )
        for i, look in enumerate(design.looks)
    )
    return BOP2ProtocolReport(
        "joint efficacy/toxicity",
        n,
        settings,
        ("Look N", "Maximum efficacy-futility events", "Minimum unsafe toxicity events"),
        rows,
        summaries,
        decision_rule=(
            "EffTox stops on either assessed failure; both efficacy and safety "
            "must pass at the final analysis."
        ),
    )


def bop2_survival_report(
    max_subjects: int,
    null_median: float,
    alternative_median: float,
    *,
    cutoff_scale: float,
    gamma: float,
    n_trials: int = 10000,
    seed: int,
    accrual_rate: float = 1.0,
    final_followup: float = 0.0,
    arrival: str = "fixed",
    looks: ArrayLike | None = None,
    prior_effective_events: float = 0.05,
    prior: ArrayLike | None = None,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2ProtocolReport:
    """Monte Carlo OC report for null and alternative exponential medians."""
    n = _positive_integer(max_subjects, "max_subjects", 200)
    trials = _positive_integer(n_trials, "n_trials", 100000)
    if 2 * n * trials > _MAX_SURVIVAL_PATIENT_TRIALS:
        raise ValueError("survival report exceeds the aggregate patient-trial budget")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    medians = (
        _probability_allow_positive(null_median, "null_median"),
        _probability_allow_positive(alternative_median, "alternative_median"),
    )
    if not medians[0] < medians[1]:
        raise ValueError("alternative_median must exceed null_median")
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be fixed or poisson")
    rate = _probability_allow_positive(accrual_rate, "accrual_rate")
    followup = scalar(final_followup, "final_followup")
    if followup < 0:
        raise ValueError("final_followup must be nonnegative")
    ess = scalar(prior_effective_events, "prior_effective_events")
    if prior is None and ess <= 0:
        raise ValueError("prior_effective_events must be positive")
    prior_input = None if prior is None else _probability_vector_allow_positive(prior, "prior", 2)
    if prior_input is not None and any(x <= 0 for x in prior_input):
        raise ValueError("prior entries must be positive")
    schedule = _schedule(looks, n)
    design = bop2_survival_design(
        n,
        medians[0],
        cutoff_scale=cutoff_scale,
        gamma=gamma,
        looks=schedule,
        prior_effective_events=ess,
        prior=prior_input,
        equality_continues=equality_continues,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    generator = np.random.default_rng(int(seed))
    summaries = []
    for label, median in zip(("null", "alternative"), medians):
        result = simulate_bop2_survival(
            design,
            median,
            accrual_rate=rate,
            final_followup=followup,
            n_trials=trials,
            arrival=arrival,
            rng=generator,
        )
        pmf = tuple(
            float(np.count_nonzero(result.sample_size == look) / trials) for look in design.looks
        )
        final_failure = float(
            np.count_nonzero((result.sample_size == design.max_subjects) & (result.success == 0))
            / trials
        )
        stop = tuple(float(value) for value in pmf[:-1]) + (final_failure,)
        summaries.append(
            BOP2ProtocolScenarioSummary(
                label,
                (median,),
                None,
                result.success_probability,
                result.success_mcse,
                stop,
                pmf,
                result.expected_sample_size,
                float(np.std(result.sample_size, ddof=0)),
                (
                    ("Expected events", float(result.events.mean())),
                    ("Mean calendar stop time", float(result.calendar_time.mean())),
                ),
            )
        )
        del result
    settings = _settings(
        null_median=medians[0],
        alternative_median=medians[1],
        cutoff_scale=design.cutoff_scale,
        gamma=design.gamma,
        prior_input=prior_input,
        prior_effective_events=ess if prior_input is None else None,
        prior_shape=design.prior_shape,
        prior_scale_ratio=design.prior_scale_ratio,
        looks=tuple(int(x) for x in design.looks),
        equality_continues=design.equality_continues,
        accrual_rate=rate,
        final_followup=followup,
        arrival=arrival,
        n_trials=trials,
        seed=int(seed),
        min_subjects=min_subjects,
        cohort_size=cohort_size,
        max_subjects=n,
    )
    boundary_rows: list[tuple[str, ...]] = []
    for look in design.looks:
        events = np.arange(int(look) + 1)
        boundaries = design.total_time_boundary(events, int(look))
        boundary_rows.extend(
            (str(int(look)), str(int(d)), _number(float(t))) for d, t in zip(events, boundaries)
        )
    return BOP2ProtocolReport(
        "single-arm time-to-event",
        n,
        settings,
        ("Look N", "Events", "Total-time futility boundary"),
        tuple(boundary_rows),
        tuple(summaries),
        decision_rule=(
            "At a scheduled look, futility occurs when total observation time is "
            f"{'strictly below' if design.equality_continues else 'at or below'} "
            "the displayed boundary. "
            "Use a consistent time unit for medians, accrual rate, follow-up and total exposure. "
            "The final result is positive or negative at maximum enrollment."
        ),
    )


def _probability_allow_positive(value: float, name: str) -> float:
    result = scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result
