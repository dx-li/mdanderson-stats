"""Complete numerical summaries for combined CTA studies."""

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cta_study import CTAStudy


def _value(value: object, digits: int) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _value(value.item(), digits)
        separator = "\t" if value.ndim == 1 else "\n"
        return separator.join(_value(item, digits) for item in value)
    if isinstance(value, (float, np.floating)):
        return "NA" if np.isnan(value) else f"{value:.{digits}g}"
    return str(value)


def format_cta(study: "CTAStudy", *, digits: int = 6) -> str:
    """Format all result fields, with zero-based indices and explicit omissions."""
    if (
        isinstance(digits, (bool, np.bool_))
        or not isinstance(digits, (int, np.integer))
        or not 1 <= digits <= 17
    ):
        raise ValueError("digits must be an integer from one through 17")
    lines = [
        "CTA contingency-table study",
        "Indices are zero-based; NA denotes an undefined quantity.",
        "\nSettings",
    ]
    for field in fields(study.specification):
        lines.append(f"{field.name}: {_value(getattr(study.specification, field.name), digits)}")
    lines.extend(
        [
            "\nObserved table",
            _value(study.observed, digits),
            "Row totals: " + _value(study.observed.sum(axis=1), digits),
            "Column totals: " + _value(study.observed.sum(axis=0), digits),
            "Total: " + _value(study.observed.sum(), digits),
            "Fisher selection: " + study.fisher_note,
        ]
    )
    for name in ("chi_square", "fisher", "kappa", "mcnemar", "diagnostic", "odds", "binomial"):
        result = getattr(study, name)
        lines.append(f"\n{name}")
        if result is None:
            lines.append("Not calculated")
            continue
        if name == "diagnostic":
            lines.append("Error/denominator order: sensitivity, specificity, PPV, NPV")
        elif name == "odds":
            lines.append("standard_error is on the log-odds-ratio scale")
        elif name == "binomial":
            lines.append("Conditional Poisson model; p_less/p_greater refer to group zero")
        for field in fields(result):
            if field.name == "observed":  # Already printed without integer rounding.
                continue
            value = getattr(result, field.name)
            if value is None:
                lines.append(f"{field.name}: Not applicable")
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                lines.extend([field.name + ":", _value(value, digits)])
            else:
                lines.append(f"{field.name}: {_value(value, digits)}")
    return "\n".join(lines) + "\n"
