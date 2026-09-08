"""MULTI p-value file/terminal parsing with explicit ignored-token diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._multi_lexer import tokens
from ._validation import FloatArray


@dataclass(frozen=True)
class MultiInputWarning:
    line: int
    column: int
    token: str
    reason: str


@dataclass(frozen=True)
class MultiData:
    pvalues: FloatArray
    order: NDArray[np.intp]
    warnings: tuple[MultiInputWarning, ...]

    @property
    def entered_values(self) -> FloatArray:
        result = np.empty_like(self.pvalues)
        result[self.order] = self.pvalues
        return result


def parse_multi_data(text: str, *, terminal: bool = False) -> MultiData:
    """Parse MULTI QLEX syntax and return sorted p-values and ignored tokens.

    At least 4 and at most 10,000 accepted values are required, as in RDDATA.
    order contains zero-based accepted-input indices; equal values sort stably.
    In terminal mode any nonnumeric token starting q/Q ends input. File mode
    diagnoses it normally. Negative integer signs and long-line loss are corrected.
    """
    if not isinstance(text, str) or not isinstance(terminal, bool):
        raise ValueError("text must be a string and terminal must be boolean")
    values, diagnostics = [], []
    stopped = False
    for line_number, line in enumerate(text.split("\n"), 1):
        for token in tokens(line.removesuffix("\r")):
            if token.value is not None:
                if np.isfinite(token.value) and 0 <= token.value <= 1:
                    values.append(token.value)
                    if len(values) > 10000:
                        raise ValueError("Too many p-values: MULTI supports at most 10000")
                    continue
                reason = "P-value not in range [0,1]"
            elif terminal and token.text[:1] in ("q", "Q"):
                stopped = True
                break
            else:
                reason = "Not a numeric value"
            diagnostics.append(MultiInputWarning(line_number, token.column, token.text, reason))
        if stopped:
            break
    if len(values) < 4:
        raise ValueError(f"Too few p-values: MULTI requires at least 4; accepted {len(values)}")
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    sorted_values.flags.writeable = False
    order.flags.writeable = False
    return MultiData(sorted_values, order, tuple(diagnostics))


def read_multi_data(path: str | Path) -> MultiData:
    """Read a UTF-8 text p-value file; propagate I/O errors without retrying."""
    return parse_multi_data(Path(path).read_text(encoding="utf-8"))
