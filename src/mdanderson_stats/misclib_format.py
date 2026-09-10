"""Checked MISCLIB integer/real number formatting without native buffers."""

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from numbers import Integral, Real

import numpy as np

from ._validation import scalar


@dataclass(frozen=True)
class FormattedNumber:
    """Text and returned field width; text is None when fit is false."""

    text: str | None
    width: int
    fit: bool


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return int(value)


def _decimal_text(value: Decimal, digits: int) -> str:
    text = format(value, f".{digits}f")
    return text if digits else text + "."


def format_number(
    x: int | float,
    *,
    justi: int = 1,
    width: int = 20,
    maxf: float = 1e6,
    minf: float = 1e-4,
    ndecf: int = 4,
    npe: int = 1,
    ndece: int = 4,
    qpad: bool = True,
) -> FormattedNumber:
    """Format one number, using source-compatible strict fixed-format thresholds.

    Integer inputs ignore floating-format options. NumPy float32 uses E notation;
    Python floats and float64 use D. Left justification returns the used width.
    """
    width = _integer(width, "width", 0, 10000)
    justi = _integer(justi, "justi", -1, 1)
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Real):
        raise ValueError("x must be a real scalar or integer")
    if isinstance(x, Integral):
        # Avoid conversion through float, including for integers beyond 2**53.
        body = str(int(x))
    else:
        value = scalar(x, "x")
        minimum, maximum = scalar(minf, "minf"), scalar(maxf, "maxf")
        if not 0 <= minimum < maximum:
            raise ValueError("Require 0 <= minf < maxf")
        single = isinstance(x, np.float32)
        if single:
            with np.errstate(over="ignore"):
                minimum, maximum = float(np.float32(minimum)), float(np.float32(maximum))
        fixed_digits = _integer(ndecf, "ndecf", 0, 100)
        exp_digits = _integer(ndece, "ndece", 0, 100)
        scale = _integer(npe, "npe", 1 - exp_digits, 100)
        if not isinstance(qpad, bool):
            raise ValueError("qpad must be boolean")
        fixed = value == 0 or minimum < abs(value) < maximum
        digits = fixed_digits if fixed else exp_digits
        with localcontext(Context(prec=800, rounding=ROUND_HALF_EVEN)):
            number = Decimal.from_float(value if value else 0.0)
            quantum = Decimal(1).scaleb(-digits)
            if fixed:
                body = _decimal_text(number.quantize(quantum), digits)
            else:
                exponent = number.copy_abs().adjusted() + 1 - scale
                mantissa = number.scaleb(-exponent).quantize(quantum)
                if abs(mantissa) >= Decimal(1).scaleb(scale):
                    exponent += 1
                    mantissa = number.scaleb(-exponent).quantize(quantum)
                body = _decimal_text(mantissa, digits)
                letter = "E" if single else "D"
                body += f"{letter}{exponent:+03d}"
        # Source checks the padded field before removing zeros.
        if len(body) > width:
            return FormattedNumber(None, width, False)
        if not qpad and digits:
            if fixed:
                mantissa_text, suffix = body, ""
            else:
                mantissa_text, exponent_text = body.split(letter)
                suffix = letter + exponent_text
            mantissa_text = mantissa_text.rstrip("0")
            if mantissa_text.endswith("."):
                mantissa_text += "0"
            body = mantissa_text + suffix
    if len(body) > width:
        return FormattedNumber(None, width, False)
    if justi == -1:
        return FormattedNumber(body, len(body), True)
    if justi == 1:
        return FormattedNumber(body.rjust(width), width, True)
    left = (width - len(body)) // 2
    return FormattedNumber(" " * left + body + " " * (width - len(body) - left), width, True)
