"""Bounded immutable calculation logs for the CONFINT planning procedures."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from inspect import signature
from pathlib import Path
from pprint import pformat

import numpy as np

from .confint_binomial import (
    confint_binomial_event_limit,
    confint_binomial_length,
    confint_binomial_probability,
    confint_binomial_sample_size,
)
from .confint_binomial_difference import (
    confint_binomial_difference_event_limit,
    confint_binomial_difference_probability,
    confint_binomial_difference_sample_size,
)
from .confint_normal import (
    confint_normal_probability,
    confint_normal_sample_size,
    confint_normal_sd_limit,
)
from .confint_poisson import (
    confint_poisson_exposure,
    confint_poisson_length,
    confint_poisson_probability,
    confint_poisson_rate_limit,
)
from .confint_survival import confint_survival_fixed_events, confint_survival_probability
from .confint_survival_inverse import confint_survival_solve
from .confint_survival_range import confint_survival_hazard_range

_PROCEDURES: dict[str, Callable[..., object]] = {
    function.__name__: function
    for function in (
        confint_normal_probability,
        confint_normal_sd_limit,
        confint_normal_sample_size,
        confint_binomial_probability,
        confint_binomial_length,
        confint_binomial_event_limit,
        confint_binomial_sample_size,
        confint_poisson_probability,
        confint_poisson_length,
        confint_poisson_rate_limit,
        confint_poisson_exposure,
        confint_binomial_difference_probability,
        confint_binomial_difference_event_limit,
        confint_binomial_difference_sample_size,
        confint_survival_fixed_events,
        confint_survival_probability,
        confint_survival_solve,
        confint_survival_hazard_range,
    )
}
_METHOD_LABELS = {
    "confint_normal_probability": "Normal-theory confidence interval total width",
    "confint_normal_sd_limit": "Population-SD limit for normal-theory total width",
    "confint_normal_sample_size": "Normal-theory sample size for total-width assurance",
    "confint_binomial_probability": "Clopper-Pearson total-width assurance",
    "confint_binomial_length": "Clopper-Pearson total width at requested assurance",
    "confint_binomial_event_limit": "Binomial probability range attaining total-width assurance",
    "confint_binomial_sample_size": "Binomial sample size attaining total-width assurance",
    "confint_poisson_probability": "Garwood rate-interval total-width assurance",
    "confint_poisson_length": "Garwood total width at requested assurance",
    "confint_poisson_rate_limit": "Poisson rate limit attaining total-width assurance",
    "confint_poisson_exposure": "Poisson exposure attaining total-width assurance",
    "confint_binomial_difference_probability": "Two-proportion Wald total-width assurance",
    "confint_binomial_difference_event_limit": "Two-proportion Wald event-probability limit",
    "confint_binomial_difference_sample_size": "Two-proportion Wald sample size for assurance",
    "confint_survival_fixed_events": "Exponential-survival total-width assurance by event count",
    "confint_survival_probability": "Exponential-survival total-width assurance",
    "confint_survival_solve": "Inverse exponential-survival total-width calculation",
    "confint_survival_hazard_range": "Exponential-survival hazard range at requested assurance",
}
_MAX_CALCULATIONS = 100
_MAX_FIXED_EVENT_ROWS = 129
_MAX_REPORT_BYTES = 2_000_000


def _input_snapshot(value: object, name: str) -> object:
    """Accept scalar settings, explicit bounds, and a short fixed-event table."""
    values: tuple[object, ...]
    if name in {"bounds", "hazard_bounds"}:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} must be an explicit two-value tuple")
        values = tuple(value)
    elif name == "events":
        if isinstance(value, np.integer):
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if not isinstance(value, tuple) or not 1 <= len(value) <= _MAX_FIXED_EVENT_ROWS:
            raise TypeError(
                f"events must be an integer or a tuple of 1 to {_MAX_FIXED_EVENT_ROWS} counts"
            )
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
            raise TypeError("events must contain integer event counts")
        values = tuple(value)
    else:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            values = (value,)
        else:
            raise TypeError(f"{name} must be scalar; report sessions do not accept array inputs")
    snapshot: list[object] = []
    for item in values:
        if isinstance(item, np.generic):
            item = item.item()
        if not (item is None or isinstance(item, (bool, int, float, str))):
            raise TypeError(f"{name} must contain scalar values")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} must contain finite values")
        if isinstance(item, str) and len(item) > 128:
            raise ValueError(f"{name} strings may contain at most 128 characters")
        snapshot.append(item)
    return tuple(snapshot) if name in {"bounds", "hazard_bounds", "events"} else snapshot[0]


def _result_snapshot(value: object, *, depth: int = 0) -> object:
    if depth > 8:
        raise ValueError("calculation result is nested too deeply to report")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else f"{number}"
    if isinstance(value, np.generic):
        return _result_snapshot(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        if value.size > _MAX_FIXED_EVENT_ROWS:
            raise ValueError("result array exceeds the report row limit")
        return {
            "shape": tuple(value.shape),
            "values": _result_snapshot(value.tolist(), depth=depth + 1),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _result_snapshot(getattr(value, field.name), depth=depth + 1)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if len(value) > 256 or not all(isinstance(key, str) for key in value):
            raise ValueError("result mappings exceed the report limits")
        return {key: _result_snapshot(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_FIXED_EVENT_ROWS:
            raise ValueError("result sequence exceeds the report row limit")
        return tuple(_result_snapshot(item, depth=depth + 1) for item in value)
    raise TypeError(f"Unsupported result type for report: {type(value).__name__}")


@dataclass(frozen=True)
class _Calculation:
    procedure: str
    label: str
    inputs: str
    result: str


def _method_label(procedure: str, inputs: Mapping[str, object]) -> str:
    target = inputs.get("target")
    if procedure.startswith("confint_normal_") and target == "mean_difference":
        return "Pooled equal-variance two-sample mean-difference interval: total CI width"
    if procedure.startswith("confint_normal_") and target == "sd":
        return "Normal-theory population-SD confidence interval: total CI width"
    if procedure.startswith("confint_normal_") and target == "mean":
        return "Normal-theory one-sample mean confidence interval: total CI width"
    if procedure.startswith("confint_survival_"):
        interval_target = "mean" if target == "mean" else "hazard"
        return f"Exponential-survival {interval_target} interval: total CI width and assurance"
    return _METHOD_LABELS[procedure]


def _none_label(procedure: str) -> str:
    if procedure == "confint_binomial_event_limit":
        return "No event-probability range attains the requested assurance at this design."
    if procedure == "confint_poisson_rate_limit":
        return "No Poisson rate attains the requested assurance at this design."
    return "This calculation returned no value."


def _atomic_write(path: str | Path, content: str) -> Path:
    destination = Path(path)
    temporary: Path | None = None
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
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class ConfintSession:
    """Immutable log of repeated calls to named CONFINT procedures."""

    _calculations: tuple[_Calculation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self._calculations, tuple):
            raise TypeError("_calculations is private; create entries with calculate()")
        if len(self._calculations) > _MAX_CALCULATIONS or any(
            not isinstance(item, _Calculation) for item in self._calculations
        ):
            raise ValueError("session calculation log is invalid or exceeds its limit")

    @property
    def calculation_count(self) -> int:
        return len(self._calculations)

    def calculate(self, procedure: str, /, **settings: object) -> tuple[ConfintSession, object]:
        """Run one named calculation and return the updated log and ordinary result."""
        if self.calculation_count >= _MAX_CALCULATIONS:
            raise ValueError(f"a session may contain at most {_MAX_CALCULATIONS} calculations")
        if not isinstance(procedure, str) or procedure not in _PROCEDURES:
            raise ValueError(f"unknown CONFINT procedure: {procedure!r}")
        function = _PROCEDURES[procedure]
        bound = signature(function).bind(**settings)
        bound.apply_defaults()
        effective = {name: _input_snapshot(value, name) for name, value in bound.arguments.items()}
        result = function(**effective)
        result_text = pformat(_result_snapshot(result), sort_dicts=False, width=88)
        item = _Calculation(
            procedure,
            _method_label(procedure, effective),
            pformat(effective, sort_dicts=False, width=88),
            result_text,
        )
        updated = ConfintSession((*self._calculations, item))
        if len(updated.report().encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ValueError("calculation log exceeds the two-megabyte report limit")
        return updated, result

    def report(self) -> str:
        """Return a readable calculation log with effective inputs and full results."""
        lines = ["CONFINT calculation report", "==========================", ""]
        for index, item in enumerate(self._calculations, start=1):
            lines.extend(
                [
                    f"Calculation {index}: {item.label}",
                    f"Procedure: {item.procedure}",
                    "Effective inputs (defaults included):",
                    item.inputs,
                    "Result (all returned fields retained):",
                    _none_label(item.procedure) if item.result == "None" else item.result,
                    "",
                ]
            )
        if not self._calculations:
            lines.append("No calculations recorded.")
        return "\n".join(lines)

    def write_report(self, path: str | Path) -> Path:
        """Render and atomically write the readable UTF-8 calculation log."""
        content = self.report()
        if len(content.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ValueError("calculation log exceeds the two-megabyte report limit")
        return _atomic_write(path, content)
