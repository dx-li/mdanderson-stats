"""Reproducible named STPLAN forward and bounded-inverse community studies."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from inspect import signature
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import poisson

from .stplan_planning import STPLAN_METHODS, STPLANSolution, _estimated_work, stplan_solve

_MAX_STUDY_BYTES = 1_048_576
_MAX_CASES = 20
_MAX_GROUPS = 100
_MAX_TOTAL_EVALUATIONS = 20_000
_MAX_TOTAL_WORK = 5_000_000
_DEFAULT_CASE_EVALUATIONS = 1_000
_LIMITATIONS = (
    "Python community report built from existing STPLAN methods. Inverse bounds and root branch "
    "are caller-specified; native automatic-bound defaults, report-template parity, and integer "
    "allocation for proportional K-group totals are not claimed."
)


def _label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 120
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("study and case names must be 1..120 printable characters")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a finite numeric scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite numeric scalar")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value: object, name: str, *, length: int | None = None) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional vector")
        size = value.size
        if not 2 <= size <= _MAX_GROUPS or (length is not None and size != length):
            raise ValueError(
                f"{name} must contain 2..{_MAX_GROUPS} values"
                if length is None
                else f"{name} must match group count"
            )
        raw_items = value.tolist()
    elif isinstance(value, (list, tuple)):
        size = len(value)
        raw_items = value
    else:
        raise ValueError(f"{name} must be a list, tuple, or one-dimensional array")
    if not 2 <= size <= _MAX_GROUPS or (length is not None and size != length):
        raise ValueError(
            f"{name} must contain 2..{_MAX_GROUPS} values"
            if length is None
            else f"{name} must match group count"
        )
    return tuple(_number(item, name) for item in raw_items)


def _group_count(value: object) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return int(value.size)
    raise ValueError("STPLAN group parameters must be one-dimensional vectors")


def _freeze_inputs(method: str, parameters: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    if len(parameters) > len(STPLAN_METHODS[method].parameters):
        raise ValueError("too many method parameters")
    if any(not isinstance(key, str) for key in parameters):
        raise ValueError("method parameter names must be strings")
    spec = STPLAN_METHODS[method]
    unknown = set(parameters) - set(spec.parameters)
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unknown))}")
    values: dict[str, object] = {}
    for key, value in parameters.items():
        if key in spec.group_parameters:
            values[key] = _vector(value, key)
        elif key == "continued_followup":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("continued_followup must be boolean")
            values[key] = bool(value)
        elif key == "model_arm":
            if not isinstance(value, str) or value not in ("lower_hazard", "higher_hazard"):
                raise ValueError("model_arm must be 'lower_hazard' or 'higher_hazard'")
            values[key] = value
        else:
            values[key] = _number(value, key)
    return values


def _case_work(method: str, kwargs: Mapping[str, object], *, evaluations: int = 1) -> int:
    """Conservative term budget for one forward call or bounded solve."""
    if method == "stplan_matched_case_control_power":
        candidate_max = _number(kwargs["n_pairs"], "n_pairs")
        terms = int(candidate_max) + 1
    elif method == "stplan_binomial_k_sample_power":
        terms = _group_count(kwargs["probabilities"])
    elif method == "stplan_poisson_two_sample_power":
        names = ("rate1", "rate2", "exposure1", "exposure2")
        upper = {name: _number(kwargs[name], name) for name in names}
        mean = upper["rate1"] * upper["exposure1"] + upper["rate2"] * upper["exposure2"]
        if not isfinite(mean):
            return _MAX_TOTAL_WORK + 1
        cutoff = float(
            poisson.isf(_number(kwargs.get("tail_tolerance", 1e-10), "tail_tolerance"), mean)
        )
        if not isfinite(cutoff):
            return _MAX_TOTAL_WORK + 1
        terms = int(np.ceil(cutoff)) + 1
    elif method == "stplan_censored_exponential_one_sample_power":
        event_mean = _number(kwargs["accrual_rate"], "accrual_rate") * _number(
            kwargs["accrual_duration"], "accrual_duration"
        )
        if not isfinite(event_mean):
            return _MAX_TOTAL_WORK + 1
        cutoff = float(
            poisson.isf(_number(kwargs.get("tail_tolerance", 1e-10), "tail_tolerance"), event_mean)
        )
        if not isfinite(cutoff):
            return _MAX_TOTAL_WORK + 1
        terms = int(np.ceil(cutoff)) + 1
    else:
        terms = _estimated_work(method, kwargs)
    if terms < 1 or terms > _MAX_TOTAL_WORK:
        return _MAX_TOTAL_WORK + 1
    return terms * evaluations


def _inverse_work_envelope(case: STPLANInverseCase, fixed: Mapping[str, object]) -> int:
    spec = STPLAN_METHODS[case.method]
    keys = (case.compute,) if isinstance(case.compute, str) else tuple(case.compute)
    whole_k = (
        case.method == "stplan_binomial_k_sample_power"
        and keys == ("sample_sizes",)
        and case.index is None
    )
    candidates: list[dict[str, object]] = []
    for endpoint in case.bounds:
        kwargs = dict(fixed)
        if whole_k:
            weights = np.asarray(
                case.allocation_weights
                if case.allocation_weights is not None
                else np.ones(_group_count(kwargs["probabilities"])),
                dtype=float,
            )
            scaled = weights / np.max(weights)
            allocation = scaled / np.sum(scaled)
            kwargs["sample_sizes"] = endpoint * allocation
        elif case.index is not None and keys[0] in ("probabilities", "sample_sizes"):
            raw_vector = kwargs[keys[0]]
            if not isinstance(raw_vector, tuple):
                raise ValueError("STPLAN group parameter was not normalized")
            vector = list(raw_vector)
            vector[case.index] = endpoint
            kwargs[keys[0]] = tuple(vector)
        else:
            for key in keys:
                kwargs[key] = endpoint
        bound = signature(spec.function).bind(**kwargs)
        bound.apply_defaults()
        candidates.append(dict(bound.arguments))
    if case.method in (
        "stplan_poisson_two_sample_power",
        "stplan_censored_exponential_one_sample_power",
    ):
        # Independently maximize positive factors to bound any interior product maximum.
        upper_kwargs = dict(candidates[0])
        for key in ("rate1", "rate2", "exposure1", "exposure2", "accrual_rate", "accrual_duration"):
            if key in upper_kwargs:
                if key in keys:
                    upper_kwargs[key] = max(case.bounds)
        candidates.append(upper_kwargs)
    per_call = max(_case_work(case.method, kwargs) for kwargs in candidates)
    # Root solving can use the configured number of evaluations plus a final residual check.
    return per_call * (int(case.max_evaluations) + 1)


@dataclass(frozen=True)
class STPLANForwardCase:
    """One named forward power calculation using fixed method inputs."""

    name: str
    method: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in STPLAN_METHODS:
            raise ValueError("unknown STPLAN method")
        object.__setattr__(self, "name", _label(self.name))
        object.__setattr__(
            self, "parameters", MappingProxyType(_freeze_inputs(self.method, self.parameters))
        )


@dataclass(frozen=True)
class STPLANInverseCase:
    """One named bounded solve; bounds, branch and integer policy are explicit."""

    name: str
    method: str
    compute: str | tuple[str, ...]
    target_power: float
    bounds: tuple[float, float]
    parameters: Mapping[str, object]
    index: int | None = None
    allocation_weights: ArrayLike | None = None
    integer: bool | None = None
    integer_goal: Literal["smallest", "largest"] | None = None
    power_tolerance: float = 1e-8
    max_evaluations: int = _DEFAULT_CASE_EVALUATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in STPLAN_METHODS:
            raise ValueError("unknown STPLAN method")
        object.__setattr__(self, "name", _label(self.name))
        object.__setattr__(
            self, "parameters", MappingProxyType(_freeze_inputs(self.method, self.parameters))
        )
        if isinstance(self.compute, (list, tuple)):
            if not 1 <= len(self.compute) <= 2:
                raise ValueError("compute must contain one or two parameter names")
            if isinstance(self.compute, list):
                object.__setattr__(self, "compute", tuple(self.compute))
        elif not isinstance(self.compute, str):
            raise ValueError("compute must be a parameter name or a short tuple")
        if isinstance(self.compute, tuple) and any(
            not isinstance(item, str) for item in self.compute
        ):
            raise ValueError("compute tuple must contain parameter names")
        if not isinstance(self.bounds, (list, tuple)) or len(self.bounds) != 2:
            raise ValueError("bounds must contain exactly two values")
        object.__setattr__(self, "bounds", tuple(_number(value, "bounds") for value in self.bounds))
        object.__setattr__(self, "target_power", _number(self.target_power, "target_power"))
        object.__setattr__(
            self, "power_tolerance", _number(self.power_tolerance, "power_tolerance")
        )
        if self.allocation_weights is not None:
            object.__setattr__(
                self, "allocation_weights", _vector(self.allocation_weights, "allocation_weights")
            )


STPLANCase = STPLANForwardCase | STPLANInverseCase


@dataclass(frozen=True)
class STPLANStudySpecification:
    """A small, portable collection of named STPLAN calculations."""

    name: str
    cases: tuple[STPLANCase, ...]
    format_version: int = 1

    def __post_init__(self) -> None:
        _label(self.name)
        if not isinstance(self.cases, (list, tuple)) or not 1 <= len(self.cases) <= _MAX_CASES:
            raise ValueError(f"a study must have 1..{_MAX_CASES} cases")
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, Integral)
            or self.format_version != 1
        ):
            raise ValueError("unsupported STPLAN study format_version")
        object.__setattr__(self, "cases", tuple(self.cases))

    def _preflight(self) -> tuple[STPLANCase, ...]:
        _label(self.name)
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, Integral)
            or self.format_version != 1
        ):
            raise ValueError("unsupported STPLAN study format_version")
        if not isinstance(self.cases, (tuple, list)) or not 1 <= len(self.cases) <= _MAX_CASES:
            raise ValueError(f"a study must have 1..{_MAX_CASES} cases")
        names: set[str] = set()
        normalized: list[STPLANCase] = []
        aggregate_evaluations = 0
        aggregate_work = 0
        for case in self.cases:
            if not isinstance(case, (STPLANForwardCase, STPLANInverseCase)):
                raise TypeError("cases must be STPLANForwardCase or STPLANInverseCase")
            name = _label(case.name)
            if name in names:
                raise ValueError("case names must be unique")
            names.add(name)
            if not isinstance(case.method, str) or case.method not in STPLAN_METHODS:
                raise ValueError("unknown STPLAN method")
            fixed = _freeze_inputs(case.method, case.parameters)
            spec = STPLAN_METHODS[case.method]
            if isinstance(case, STPLANForwardCase):
                bound = signature(spec.function).bind(**fixed)
                bound.apply_defaults()
                forward_inputs = dict(bound.arguments)
                work = _case_work(case.method, forward_inputs)
                aggregate_work += work
                if aggregate_work > _MAX_TOTAL_WORK:
                    raise ValueError("study exceeds the five-million-term aggregate work budget")
                normalized.append(STPLANForwardCase(name, case.method, forward_inputs))
                continue
            compute = case.compute
            keys = (compute,) if isinstance(compute, str) else tuple(compute)
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("compute must name one or a supported tied parameter tuple")
            if keys not in spec.tied_parameters and (
                len(keys) != 1 or keys[0] not in spec.compute_parameters
            ):
                raise ValueError("compute is not supported by this STPLAN method")
            if set(keys) & fixed.keys():
                if not (case.index is not None and case.method == "stplan_binomial_k_sample_power"):
                    raise ValueError("omit computed parameters from fixed parameters")
            missing = set(spec.required_parameters) - fixed.keys() - set(keys)
            whole_k = (
                case.method == "stplan_binomial_k_sample_power"
                and keys == ("sample_sizes",)
                and case.index is None
            )
            indexed_k = (
                case.method == "stplan_binomial_k_sample_power"
                and case.index is not None
                and len(keys) == 1
                and keys[0] in ("probabilities", "sample_sizes")
            )
            if case.index is not None and not indexed_k:
                raise ValueError("index is supported only for indexed K-sample parameters")
            if case.allocation_weights is not None and not whole_k:
                raise ValueError(
                    "allocation_weights is only valid for whole K-sample total-size solves"
                )
            if whole_k and "probabilities" not in fixed:
                raise ValueError("whole K-sample total-size solve requires fixed probabilities")
            if indexed_k:
                vector = fixed.get(keys[0])
                if not isinstance(vector, tuple) or case.index is None or case.index >= len(vector):
                    raise ValueError("index is outside the fixed K-sample vector")
            if whole_k and case.integer is True:
                raise ValueError("whole K-sample total-size scaling is continuous")
            if whole_k:
                missing.discard("sample_sizes")
            if missing:
                raise ValueError(f"missing required parameters: {', '.join(sorted(missing))}")
            bound_inputs = signature(spec.function).bind_partial(**fixed)
            bound_inputs.apply_defaults()
            fixed = dict(bound_inputs.arguments)
            target = _number(case.target_power, "target_power")
            if not 0 < target < 1:
                raise ValueError("target_power must lie strictly between 0 and 1")
            if not isinstance(case.bounds, tuple) or len(case.bounds) != 2:
                raise ValueError("bounds must be a two-value tuple")
            bounds = tuple(_number(item, "bounds") for item in case.bounds)
            if not bounds[0] < bounds[1]:
                raise ValueError("bounds must be strictly increasing")
            if case.index is not None and (
                isinstance(case.index, bool)
                or not isinstance(case.index, Integral)
                or case.index < 0
            ):
                raise ValueError("index must be a nonnegative integer")
            if case.integer is not None and not isinstance(case.integer, bool):
                raise ValueError("integer must be boolean or None")
            if case.integer_goal is not None and (
                not isinstance(case.integer_goal, str)
                or case.integer_goal not in ("smallest", "largest")
            ):
                raise ValueError("invalid integer_goal")
            tolerance = _number(case.power_tolerance, "power_tolerance")
            if not 0 < tolerance < 0.1:
                raise ValueError("power_tolerance must lie strictly between 0 and 0.1")
            if (
                isinstance(case.max_evaluations, bool)
                or not isinstance(case.max_evaluations, Integral)
                or not 1 <= case.max_evaluations <= _MAX_TOTAL_EVALUATIONS
            ):
                raise ValueError("max_evaluations must be in 1..20000")
            aggregate_evaluations += int(case.max_evaluations)
            if aggregate_evaluations > _MAX_TOTAL_EVALUATIONS:
                raise ValueError("study exceeds the aggregate 20,000 inverse-evaluation budget")
            weights = None
            if case.allocation_weights is not None:
                length = len(fixed["probabilities"]) if whole_k else None
                weights = _vector(case.allocation_weights, "allocation_weights", length=length)
                if any(weight <= 0 for weight in weights):
                    raise ValueError("allocation_weights must be positive")
            normalized_case = STPLANInverseCase(
                name,
                case.method,
                case.compute,
                target,
                (bounds[0], bounds[1]),
                fixed,
                None if case.index is None else int(case.index),
                weights,
                case.integer,
                case.integer_goal,
                tolerance,
                int(case.max_evaluations),
            )
            aggregate_work += _inverse_work_envelope(normalized_case, fixed)
            if aggregate_work > _MAX_TOTAL_WORK:
                raise ValueError("study exceeds the five-million-term aggregate work budget")
            normalized.append(normalized_case)
        return tuple(normalized)

    def run(self) -> STPLANStudy:
        cases = self._preflight()
        results: list[STPLANCaseResult] = []
        for case in cases:
            if isinstance(case, STPLANForwardCase):
                power_array = np.asarray(
                    STPLAN_METHODS[case.method].function(**case.parameters), dtype=float
                )
                if power_array.ndim != 0 or not np.isfinite(power_array):
                    raise ValueError("saved studies require scalar forward-power results")
                power = float(power_array)
                if not 0 <= power <= 1:
                    raise ArithmeticError("forward method returned power outside [0,1]")
                results.append(
                    STPLANCaseResult(
                        case.name,
                        case.method,
                        "forward",
                        case.parameters,
                        power,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            else:
                solution = stplan_solve(
                    case.method,
                    compute=case.compute,
                    target_power=case.target_power,
                    bounds=case.bounds,
                    parameters=case.parameters,
                    index=case.index,
                    allocation_weights=case.allocation_weights,
                    integer=case.integer,
                    integer_goal=case.integer_goal,
                    power_tolerance=case.power_tolerance,
                    max_evaluations=case.max_evaluations,
                )
                results.append(_solution_result(case, solution))
        return STPLANStudy(self.name, tuple(results), cases)

    def to_json(self) -> str:
        cases = self._preflight()
        data = {
            "format_version": 1,
            "name": self.name,
            "cases": [_case_json(case) for case in cases],
        }
        text = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if len(text.encode("utf-8")) > _MAX_STUDY_BYTES:
            raise ValueError("STPLAN study JSON exceeds the 1 MiB limit")
        return text

    @classmethod
    def from_json(cls, text: str) -> STPLANStudySpecification:
        if not isinstance(text, str) or len(text) > _MAX_STUDY_BYTES:
            raise ValueError("STPLAN study JSON must be text no larger than 1 MiB")
        if len(text.encode("utf-8")) > _MAX_STUDY_BYTES:
            raise ValueError("STPLAN study JSON must be text no larger than 1 MiB")

        def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number: {value}")

        raw = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
        if not isinstance(raw, dict) or set(raw) != {"format_version", "name", "cases"}:
            raise ValueError("STPLAN study JSON has unknown or missing top-level fields")
        if not isinstance(raw["cases"], list) or not 1 <= len(raw["cases"]) <= _MAX_CASES:
            raise ValueError(f"study JSON must contain 1..{_MAX_CASES} cases")
        cases: list[STPLANCase] = []
        for item in raw["cases"]:
            if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
                raise ValueError("each case must be an object with a kind")
            kind = item["kind"]
            common = {"kind", "name", "method", "parameters"}
            if kind == "forward":
                if set(item) != common:
                    raise ValueError("invalid forward case fields")
                cases.append(STPLANForwardCase(item["name"], item["method"], item["parameters"]))
            elif kind == "inverse":
                expected = common | {
                    "compute",
                    "target_power",
                    "bounds",
                    "index",
                    "allocation_weights",
                    "integer",
                    "integer_goal",
                    "power_tolerance",
                    "max_evaluations",
                }
                if set(item) != expected:
                    raise ValueError("invalid inverse case fields")
                cases.append(
                    STPLANInverseCase(
                        **{key: value for key, value in item.items() if key != "kind"}
                    )
                )
            else:
                raise ValueError("unknown case kind")
        study = cls(raw["name"], tuple(cases), raw["format_version"])
        study._preflight()
        return study

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        return _atomic_write(path, self.to_json())


@dataclass(frozen=True)
class STPLANCaseResult:
    name: str
    method: str
    kind: str
    inputs: Mapping[str, object]
    achieved_power: float
    target_power: float | None
    computed: str | tuple[str, ...] | None
    value: object
    bounds: tuple[float, float] | None
    evaluations: int | None = None
    integer_search: bool | None = None
    integer_goal: str | None = None
    previous_value: float | None = None
    previous_power: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


@dataclass(frozen=True)
class STPLANStudy:
    name: str
    results: tuple[STPLANCaseResult, ...]
    specification: tuple[STPLANCase, ...]

    def to_json(self) -> str:
        rows = []
        for result in self.results:
            rows.append(
                {
                    "name": result.name,
                    "method": result.method,
                    "kind": result.kind,
                    "inputs": _json_safe(result.inputs),
                    "achieved_power": result.achieved_power,
                    "target_power": result.target_power,
                    "computed": result.computed,
                    "value": _json_safe(result.value),
                    "bounds": result.bounds,
                    "evaluations": result.evaluations,
                    "integer_search": result.integer_search,
                    "integer_goal": result.integer_goal,
                    "previous_value": result.previous_value,
                    "previous_power": result.previous_power,
                }
            )
        text = (
            json.dumps(
                {
                    "format_version": 1,
                    "study": self.name,
                    "limitations": _LIMITATIONS,
                    "specification": [_case_json(case) for case in self.specification],
                    "results": rows,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if len(text.encode("utf-8")) > _MAX_STUDY_BYTES:
            raise ValueError("STPLAN result JSON exceeds the 1 MiB limit")
        return text

    def to_html(self) -> str:
        sections = []
        specifications = {case.name: case for case in self.specification}
        for result in self.results:
            rows = "".join(
                _html_row(str(key), value, code=True) for key, value in result.inputs.items()
            )
            computed = (
                ""
                if result.computed is None
                else (
                    f"<p>Computed {escape(str(result.computed))}: "
                    f"{escape(json.dumps(_json_safe(result.value)))}</p>"
                )
            )
            bounds = (
                ""
                if result.bounds is None
                else f"<p>Explicit search bounds: {escape(json.dumps(result.bounds))}</p>"
            )
            target = (
                ""
                if result.target_power is None
                else f"<p>Requested power: {result.target_power:.17g}</p>"
            )
            settings = ""
            case_spec = specifications[result.name]
            if isinstance(case_spec, STPLANInverseCase):
                controls = (
                    ("Computed parameter", case_spec.compute),
                    ("Integer search", result.integer_search),
                    ("Integer goal", result.integer_goal),
                    ("Index", case_spec.index),
                    ("Allocation weights", case_spec.allocation_weights),
                    ("Power tolerance", case_spec.power_tolerance),
                    ("Maximum evaluations", case_spec.max_evaluations),
                )
                settings_rows = "".join(_html_row(label, value) for label, value in controls)
                settings_rows += "".join(
                    _html_row(label, value)
                    for label, value in (
                        ("Evaluations used", result.evaluations),
                        ("Previous candidate", result.previous_value),
                        ("Previous candidate power", result.previous_power),
                    )
                )
                settings = (
                    "<table><caption>Captured inverse settings</caption>"
                    f"<tbody>{settings_rows}</tbody></table>"
                )
            sections.append(
                f"<section><h2>{escape(result.name)}</h2>"
                f"<p>Method: <code>{escape(result.method)}</code></p>"
                "<table><caption>Captured effective forward inputs</caption>"
                f"<tbody>{rows}</tbody></table>"
                f"{target}<p>Achieved power: {result.achieved_power:.17g}</p>"
                f"{computed}{bounds}{settings}</section>"
            )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>STPLAN study: {escape(self.name)}</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1000px;"
            "margin:2em auto;padding:0 1em}table{border-collapse:collapse;"
            "margin:1em 0 2em}th,td{border:1px solid #bbb;padding:.4em;"
            "text-align:left}</style></head><body>"
            f"<h1>STPLAN study: {escape(self.name)}</h1>"
            f"<p>{escape(_LIMITATIONS)}</p>" + "".join(sections) + "</body></html>\n"
        )

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        return _atomic_write(path, self.to_json())

    def write_html(self, path: str | os.PathLike[str]) -> Path:
        return _atomic_write(path, self.to_html())


def _solution_result(case: STPLANInverseCase, solution: STPLANSolution) -> STPLANCaseResult:
    return STPLANCaseResult(
        case.name,
        case.method,
        "inverse",
        solution.inputs,
        solution.achieved_power,
        solution.target_power,
        solution.compute,
        solution.value,
        solution.bounds,
        solution.evaluations,
        solution.integer_search,
        solution.integer_goal,
        solution.previous_value,
        solution.previous_power,
    )


def _html_row(label: str, value: object, *, code: bool = False) -> str:
    serialized = escape(json.dumps(_json_safe(value), allow_nan=False))
    cell = f"<code>{serialized}</code>" if code else serialized
    return f"<tr><th>{escape(label)}</th><td>{cell}</td></tr>"


def _case_json(case: STPLANCase) -> dict[str, object]:
    common: dict[str, object] = {
        "name": case.name,
        "method": case.method,
        "parameters": _json_safe(case.parameters),
    }
    if isinstance(case, STPLANForwardCase):
        return {"kind": "forward", **common}
    return {
        "kind": "inverse",
        **common,
        "compute": case.compute,
        "target_power": case.target_power,
        "bounds": case.bounds,
        "index": case.index,
        "allocation_weights": None
        if case.allocation_weights is None
        else _vector(case.allocation_weights, "allocation_weights"),
        "integer": case.integer,
        "integer_goal": case.integer_goal,
        "power_tolerance": case.power_tolerance,
        "max_evaluations": case.max_evaluations,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write(path: str | os.PathLike[str], content: str) -> Path:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(content)
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
