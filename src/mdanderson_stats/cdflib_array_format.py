"""Checked CDFLIB numeric array formatting without a fixed output buffer."""

import math
import re

from numpy.typing import ArrayLike

from ._validation import finite

# The original unkinded REAL literal is single precision, promoted for comparison.
_SMALL = 0.0010000000474974513
_FIELD = re.compile(r"([0-9]*)([FE])([0-9]+)\.([0-9]+)\Z")
_SPACE = re.compile(r"([0-9]*)X\Z")


def _fit(body: str, width: int) -> str:
    if len(body) > width:
        if body.startswith("0."):
            body = body[1:]
        elif body.startswith("-0."):
            body = "-" + body[2:]
    return body.rjust(width) if len(body) <= width else "*" * width


def _scientific(value: float, width: int, precision: int) -> str:
    if precision == 0:
        raise ValueError("Scientific fields require at least one decimal place")
    if precision + 5 > width:
        return "*" * width
    negative = math.copysign(1.0, value) < 0
    if value == 0:
        digits, exponent = "0" * precision, 0
    else:
        mantissa, power = format(abs(value), f".{precision - 1}e").split("e")
        digits = mantissa.replace(".", "")
        exponent = int(power) + 1
    sign = "+" if exponent >= 0 else "-"
    # Fortran's default E descriptor omits E for a three-digit exponent.
    suffix = ("E" if abs(exponent) < 100 else "") + sign + str(abs(exponent)).zfill(2)
    return _fit(("-" if negative else "") + "0." + digits + suffix, width)


def _fixed(value: float, width: int, precision: int) -> str:
    if precision + 1 > width:
        return "*" * width
    body = format(value, f".{precision}f")
    if precision == 0:
        body += "."
    return _fit(body, width)


def format_cdflib_array(values: ArrayLike, format_spec: str, *, max_output: int = 1000000) -> str:
    """Render a finite vector using repeated Fw.d, Ew.d and nX fields.

    F fields switch to scientific notation below the archived single-precision
    1e-3 threshold or above 1000; scientific precision is reduced by three when
    F precision exceeds four. E fields always use their specified precision.
    Insufficient field width produces asterisks, as in Fortran. The complete
    field count must match the input length. Return exactly the requested field
    widths, with no implicit list-directed leading blank, padding or newline.
    """
    array = finite(values, "values")
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not isinstance(format_spec, str):
        raise ValueError("format_spec must be a string")
    if isinstance(max_output, bool) or not isinstance(max_output, int) or max_output < 1:
        raise ValueError("max_output must be a positive integer")
    spec = format_spec.strip().upper()
    if spec.startswith("(") and spec.endswith(")"):
        spec = spec[1:-1].strip()
    fields = []
    count = total = 0
    for token in re.split(r"[\s,]+", spec) if spec else []:
        if not token:
            continue
        spaces = _SPACE.fullmatch(token)
        numeric = _FIELD.fullmatch(token)
        if spaces:
            repeat = int(spaces[1] or "1")
            if repeat < 1:
                raise ValueError("Space counts must be positive")
            fields.append(("X", repeat, 0, 0))
            total += repeat
        elif numeric:
            repeat = int(numeric[1] or "1")
            width, precision = int(numeric[3]), int(numeric[4])
            if repeat < 1 or width < 1:
                raise ValueError("Field counts and widths must be positive")
            if numeric[2] == "E" and precision == 0:
                raise ValueError("Scientific fields require at least one decimal place")
            fields.append((numeric[2], repeat, width, precision))
            count += repeat
            total += repeat * width
        else:
            raise ValueError(f"Unsupported array format field: {token}")
        if total > max_output:
            raise ValueError("Array format exceeds max_output")
    if count != array.size:
        raise ValueError("Numeric field count must equal the input length")
    parts = []
    index = 0
    for kind, repeat, width, precision in fields:
        if kind == "X":
            parts.append(" " * repeat)
            continue
        for value in array[index : index + repeat]:
            number = float(value)
            scientific = kind == "E" or abs(number) < _SMALL or abs(number) > 1000
            if scientific:
                digits = precision - 3 if kind == "F" and precision > 4 else precision
                parts.append(_scientific(number, width, digits))
            else:
                parts.append(_fixed(number, width, precision))
        index += repeat
    return "".join(parts)
