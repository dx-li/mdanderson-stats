"""Reconstructible ASYPOW model calculations and readable result reports."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import signature
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .asypow import (
    AsymptoticPower,
    _preflight_target,
    _probability,
    asypow_information,
)
from .asypow_design import asypow_design_information
from .asypow_generic import asypow_smo_generic
from .asypow_groups import asypow_group_information
from .asypow_multinomial import asypow_multinomial_information
from .asypow_ordinal import asypow_ordinal_information, asypow_ordinal_regression_information
from .asypow_regression import asypow_regression_information
from .asypow_smo import SMOPower, asypow_smo_binomial, asypow_smo_poisson
from .asypow_smo_categorical import asypow_smo_multinomial, asypow_smo_ordinal
from .asypow_smo_design import asypow_smo_design
from .asypow_smo_exponential import asypow_smo_exponential
from .asypow_smo_ordinal_regression import asypow_smo_ordinal_regression
from .asypow_smo_regression import asypow_smo_regression
from .boin import _owned

_MAX_SETTINGS = 64
_MAX_SETTING_ELEMENTS = 1_000_000
_MAX_REPORT_CHARS = 2_000_000
_MAX_SETTING_DEPTH = 16
_MAX_SETTING_STRING_CHARS = 4096

_Procedure = Callable[..., AsymptoticPower | SMOPower]


def _lr_groups(
    parameters: ArrayLike,
    contrasts: ArrayLike,
    *,
    model: str = "binomial",
    null_values: ArrayLike = 0,
    group_size: ArrayLike = 1,
    duration: ArrayLike | None = None,
) -> AsymptoticPower:
    information = asypow_group_information(
        parameters, model=model, group_size=group_size, duration=duration
    )
    return asypow_information(parameters, information, contrasts, null_values)


def _lr_regression(
    parameters: ArrayLike,
    covariates: ArrayLike,
    contrasts: ArrayLike,
    *,
    family: str = "logistic",
    null_values: ArrayLike = 0,
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
    duration: ArrayLike | None = None,
) -> AsymptoticPower:
    information = asypow_regression_information(
        parameters,
        covariates,
        family=family,
        observations=observations,
        group_size=group_size,
        duration=duration,
    )
    theta = finite(parameters, "parameters").ravel()
    return asypow_information(theta, information, contrasts, null_values)


def _lr_ordinal(
    cumulative: ArrayLike,
    contrasts: ArrayLike,
    *,
    null_values: ArrayLike = 0,
    group_size: ArrayLike = 1,
) -> AsymptoticPower:
    information = asypow_ordinal_information(cumulative, group_size=group_size)
    theta = finite(cumulative, "cumulative").ravel()
    return asypow_information(theta, information, contrasts, null_values)


def _lr_ordinal_regression(
    parameters: ArrayLike,
    covariates: ArrayLike,
    contrasts: ArrayLike,
    *,
    quadratic: bool = False,
    link: str = "logistic",
    null_values: ArrayLike = 0,
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
) -> AsymptoticPower:
    information = asypow_ordinal_regression_information(
        parameters,
        covariates,
        quadratic=quadratic,
        link=link,
        observations=observations,
        group_size=group_size,
    )
    theta = finite(parameters, "parameters").ravel()
    return asypow_information(theta, information, contrasts, null_values)


def _lr_multinomial(
    probabilities: ArrayLike,
    contrasts: ArrayLike,
    *,
    null_values: ArrayLike = 0,
    group_size: ArrayLike = 1,
) -> AsymptoticPower:
    information = asypow_multinomial_information(probabilities, group_size=group_size)
    theta = finite(probabilities, "probabilities").ravel()
    return asypow_information(theta, information, contrasts, null_values)


def _lr_design(
    coefficients: ArrayLike,
    design: ArrayLike,
    contrasts: ArrayLike,
    *,
    model: str = "logistic",
    null_values: ArrayLike = 0,
    observations: ArrayLike = 1,
) -> AsymptoticPower:
    information = asypow_design_information(
        coefficients, design, model=model, observations=observations
    )
    return asypow_information(coefficients, information, contrasts, null_values)


_PROCEDURES: dict[str, tuple[_Procedure, Literal["LR", "SMO"], str]] = {
    "lr_information": (asypow_information, "LR", "Generic information-matrix LR"),
    "lr_groups": (_lr_groups, "LR", "Independent-group LR"),
    "lr_regression": (_lr_regression, "LR", "Regression-information LR"),
    "lr_ordinal": (_lr_ordinal, "LR", "Ordinal-information LR"),
    "lr_ordinal_regression": (_lr_ordinal_regression, "LR", "Ordinal-regression LR"),
    "lr_multinomial": (_lr_multinomial, "LR", "Multinomial-information LR"),
    "lr_design": (_lr_design, "LR", "General-design LR"),
    "asypow_smo_binomial": (asypow_smo_binomial, "SMO", "Binomial SMO"),
    "asypow_smo_poisson": (asypow_smo_poisson, "SMO", "Poisson SMO"),
    "asypow_smo_exponential": (asypow_smo_exponential, "SMO", "Exponential-survival SMO"),
    "asypow_smo_multinomial": (asypow_smo_multinomial, "SMO", "Multinomial SMO"),
    "asypow_smo_ordinal": (asypow_smo_ordinal, "SMO", "Ordinal SMO"),
    "asypow_smo_regression": (asypow_smo_regression, "SMO", "Regression SMO"),
    "asypow_smo_ordinal_regression": (
        asypow_smo_ordinal_regression,
        "SMO",
        "Ordinal-regression SMO",
    ),
    "asypow_smo_design": (asypow_smo_design, "SMO", "General-design SMO"),
    "asypow_smo_generic": (asypow_smo_generic, "SMO", "Generic callback SMO"),
}


@dataclass(frozen=True, slots=True)
class AsyPowCalculationRequest:
    """A named model route, effective settings, and exactly two target values."""

    procedure: str
    settings: Mapping[str, object]
    significance: ArrayLike | None = None
    power: ArrayLike | None = None
    sample_size: ArrayLike | None = None


@dataclass(frozen=True, slots=True)
class AsyPowCalculation:
    """One constructed ASYPOW model and its complete target table."""

    procedure: str
    method: str
    label: str
    effective_settings: tuple[tuple[str, object], ...]
    calculated: Literal["significance", "power", "sample_size"]
    significance: float | FloatArray
    power: float | FloatArray
    sample_size: float | FloatArray
    model_result: AsymptoticPower | SMOPower

    def report(self, *, digits: int = 8) -> str:
        """Return a deterministic bounded report of inputs, fit, and outputs."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        settings_size = sum(_render_size(value) for _, value in self.effective_settings)
        if settings_size > _MAX_REPORT_CHARS:
            raise ValueError("effective settings exceed the two-million-character report limit")
        fit = self.model_result
        if isinstance(fit, AsymptoticPower):
            noncentrality = _format(fit.noncentrality_per_observation, digits)
            fit_summary = (
                f"degrees_of_freedom={fit.degrees_of_freedom}; "
                f"noncentrality_per_observation={noncentrality}; "
                f"null_parameters={_render_value(fit.null_parameters)}"
            )
        else:
            fit_summary = (
                f"degrees_of_freedom={fit.degrees_of_freedom}; "
                f"divergence_per_observation={_format(fit.divergence_per_observation, digits)}; "
                f"subtract_df={fit.subtract_df}; "
                f"null_parameters={_render_value(fit.null_parameters)}"
            )
        significance = np.atleast_1d(self.significance)
        power = np.atleast_1d(self.power)
        sample_size = np.atleast_1d(self.sample_size)
        lines = [
            "ASYPOW calculation report",
            f"Procedure: {self.procedure}",
            f"Method: {self.method}",
            f"Model route: {self.label}",
            f"Calculated target: {self.calculated}",
            "Effective settings (defaults included):",
        ]
        lines.extend(
            f"  {name} = {_render_value(value)}" for name, value in self.effective_settings
        )
        lines.extend(
            [
                f"Model result: {fit_summary}",
                "\nSignificance\tPower\tSample size",
            ]
        )
        lines.extend(
            "\t".join(
                (
                    _format(float(significance[i]), digits),
                    _format(float(power[i]), digits),
                    _format(float(sample_size[i]), digits),
                )
            )
            for i in range(significance.size)
        )
        report = "\n".join(lines) + "\n"
        if len(report) > _MAX_REPORT_CHARS:
            raise ValueError("report exceeds the two-million-character limit")
        return report


def asypow_calculate(request: AsyPowCalculationRequest) -> AsyPowCalculation:
    """Construct a named LR/SMO model and calculate its missing target.

    Procedure names are listed in :mod:`mdanderson_stats.asypow_workflow`.
    Settings are bound to the selected constructor's signature, including its
    defaults, then snapshotted before the constructor is called.
    """
    if not isinstance(request, AsyPowCalculationRequest):
        raise TypeError("request must be an AsyPowCalculationRequest")
    if not isinstance(request.procedure, str) or request.procedure not in _PROCEDURES:
        raise ValueError(f"unknown ASYPOW procedure: {request.procedure!r}")
    provided = {
        "significance": request.significance,
        "power": request.power,
        "sample_size": request.sample_size,
    }
    missing = [name for name, value in provided.items() if value is None]
    if len(missing) != 1:
        raise ValueError("supply exactly two of significance, power, and sample_size")
    if not isinstance(request.settings, Mapping) or len(request.settings) > _MAX_SETTINGS:
        raise ValueError(f"settings must be a mapping with at most {_MAX_SETTINGS} entries")
    if any(
        not isinstance(name, str) or len(name) > _MAX_SETTING_STRING_CHARS
        for name in request.settings
    ):
        raise ValueError("settings keys must be strings no longer than 4096 characters")

    function, method, label = _PROCEDURES[request.procedure]
    bound = signature(function).bind(**request.settings)
    bound.apply_defaults()
    element_budget = [sum(len(name) + 1 for name in request.settings)]
    if element_budget[0] > _MAX_SETTING_ELEMENTS:
        raise ValueError("effective settings exceed the one-million-unit snapshot limit")
    execution_settings: dict[str, object] = {}
    report_settings: list[tuple[str, object]] = []
    for name, value in bound.arguments.items():
        copied, recorded = _snapshot(value, name, element_budget)
        execution_settings[name] = copied
        report_settings.append((name, recorded))

    call_value, scalar = _paired_targets(request, missing[0])
    model = function(**execution_settings)
    if not isinstance(model, (AsymptoticPower, SMOPower)):
        raise TypeError("selected constructor did not return an ASYPOW power result")
    significance, power, sample_size = _calculate_targets(model, request, missing[0], call_value)
    if scalar:
        significance_value: float | FloatArray = float(significance[0])
        power_value: float | FloatArray = float(power[0])
        sample_size_value: float | FloatArray = float(sample_size[0])
    else:
        significance_value = _owned(significance)
        power_value = _owned(power)
        sample_size_value = _owned(sample_size)
    return AsyPowCalculation(
        request.procedure,
        method,
        label,
        tuple(report_settings),
        missing[0],  # type: ignore[arg-type]
        significance_value,
        power_value,
        sample_size_value,
        model,
    )


def _paired_targets(
    request: AsyPowCalculationRequest, missing: str
) -> tuple[tuple[FloatArray, FloatArray], bool]:
    first_name, second_name = {
        "significance": ("power", "sample_size"),
        "power": ("significance", "sample_size"),
        "sample_size": ("power", "significance"),
    }[missing]
    first = getattr(request, first_name)
    second = getattr(request, second_name)
    first_size, first_scalar = _preflight_target(first, first_name)
    second_size, second_scalar = _preflight_target(second, second_name)
    if first_size != 1 and second_size != 1 and first_size != second_size:
        raise ValueError(
            f"{first_name} and {second_name} vectors must have equal lengths or be scalar"
        )
    size = max(first_size, second_size)
    if first_name in {"power", "significance"}:
        first_values = _probability(first, first_name)
    else:
        first_values = finite(first, first_name)
        if np.any(first_values < 0):
            raise ValueError("sample_size must be nonnegative")
    if second_name in {"power", "significance"}:
        second_values = _probability(second, second_name)
    else:
        second_values = finite(second, second_name)
        if np.any(second_values < 0):
            raise ValueError("sample_size must be nonnegative")
    first_values = _owned(np.broadcast_to(first_values, (size,)))
    second_values = _owned(np.broadcast_to(second_values, (size,)))
    return (first_values, second_values), first_scalar and second_scalar


def _calculate_targets(
    model: AsymptoticPower | SMOPower,
    request: AsyPowCalculationRequest,
    missing: str,
    pair: tuple[FloatArray, FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    first_name, second_name = {
        "significance": ("power", "sample_size"),
        "power": ("significance", "sample_size"),
        "sample_size": ("power", "significance"),
    }[missing]
    first_values, second_values = pair
    scalar_inputs = all(
        _preflight_target(value, name)[1]
        for name, value in (
            (first_name, getattr(request, first_name)),
            (second_name, getattr(request, second_name)),
        )
    )
    requested = {first_name: first_values, second_name: second_values}
    computed: float | FloatArray
    if missing == "power":
        computed = model.power(requested["sample_size"], requested["significance"])
    elif missing == "significance":
        computed = model.significance(requested["sample_size"], requested["power"])
    else:
        power_value = requested["power"]
        significance_value = requested["significance"]
        computed = model.sample_size(
            float(power_value[0]) if scalar_inputs else power_value,
            float(significance_value[0]) if scalar_inputs else significance_value,
        )
    requested[missing] = finite(computed, missing)
    values = []
    for name in ("significance", "power", "sample_size"):
        array = finite(requested[name], name)
        if array.size not in (1, first_values.size):
            raise ArithmeticError(f"computed {name} does not align with the input target vectors")
        values.append(np.broadcast_to(array, (first_values.size,)).copy())
    return values[0], values[1], values[2]


def _snapshot(
    value: object, name: str, budget: list[int], *, depth: int = 0
) -> tuple[object, object]:
    """Copy bounded settings once for execution and immutable report capture."""
    if depth > _MAX_SETTING_DEPTH:
        raise ValueError(f"setting {name} is nested too deeply")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, str) and len(value) > _MAX_SETTING_STRING_CHARS:
            raise ValueError(f"setting {name} exceeds the string-length limit")
        if isinstance(value, str):
            budget[0] += len(value)
            if budget[0] > _MAX_SETTING_ELEMENTS:
                raise ValueError("effective settings exceed the one-million-unit snapshot limit")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"setting {name} must be finite")
        return value, value
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, complex):
            raise ValueError(f"setting {name} must be real")
        return _snapshot(native, name, budget, depth=depth + 1)
    if callable(value):
        module = getattr(value, "__module__", "?")
        name = getattr(value, "__qualname__", type(value).__name__)
        label = f"<callable {module}.{name}; closure state not captured>"
        return value, label
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or value.dtype.kind not in "biuf":
            raise ValueError(f"setting {name} must be a real numeric array")
        if value.ndim > 2:
            raise ValueError(f"setting {name} arrays may have at most two dimensions")
        budget[0] += int(value.size)
        if budget[0] > _MAX_SETTING_ELEMENTS:
            raise ValueError("effective settings exceed the one-million-unit snapshot limit")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(value, dtype=np.float64)
        if np.any(~np.isfinite(converted)):
            raise ValueError(f"setting {name} must remain finite when represented as float64")
        copied = _owned(converted)
        return copied, copied
    if isinstance(value, Mapping):
        if len(value) > _MAX_SETTINGS:
            raise ValueError(f"nested setting {name} exceeds the mapping limit")
        budget[0] += len(value)
        if budget[0] > _MAX_SETTING_ELEMENTS:
            raise ValueError("effective settings exceed the one-million-unit snapshot limit")
        execution: dict[str, object] = {}
        recorded: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > _MAX_SETTING_STRING_CHARS:
                raise ValueError(f"nested setting {name} keys must be strings")
            budget[0] += len(key)
            if budget[0] > _MAX_SETTING_ELEMENTS:
                raise ValueError("effective settings exceed the one-million-unit snapshot limit")
            execution[key], recorded[key] = _snapshot(
                item, f"{name}.{key}", budget, depth=depth + 1
            )
        return MappingProxyType(execution), MappingProxyType(recorded)
    if isinstance(value, (list, tuple)):
        budget[0] += len(value)
        if budget[0] > _MAX_SETTING_ELEMENTS:
            raise ValueError("effective settings exceed the one-million-unit snapshot limit")
        items = [_snapshot(item, name, budget, depth=depth + 1) for item in value]
        execution_items = tuple(item[0] for item in items)
        recorded_items = tuple(item[1] for item in items)
        return execution_items, recorded_items
    raise ValueError(f"setting {name} has an unsupported value type")


def _render_size(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size) * 32 + 64
    if isinstance(value, Mapping):
        return sum(len(str(key)) + _render_size(item) + 4 for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return sum(_render_size(item) + 2 for item in value)
    return len(repr(value))


def _render_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(
            value,
            threshold=value.size,
            separator=", ",
            max_line_width=88,
            precision=17,
            floatmode="unique",
            formatter={"float_kind": lambda number: repr(float(number))},
        )
    if isinstance(value, Mapping):
        return (
            "{" + ", ".join(f"{key!r}: {_render_value(item)}" for key, item in value.items()) + "}"
        )
    if isinstance(value, tuple):
        rendered = ", ".join(_render_value(item) for item in value)
        return f"({rendered}{',' if len(value) == 1 else ''})"
    return repr(value)


def _format(value: float, digits: int) -> str:
    return format(float(value), f".{digits}g")
