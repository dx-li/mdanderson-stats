"""Reentrant CDFLIB console input with explicit streams and bounded retries."""

import re
import sys
from typing import NoReturn, TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite
from .cdflib_strings import _MANTISSA, _NUMBER, lower_case_char


class CDFConsoleError(RuntimeError):
    """A console operation failed or exhausted its input allowance."""


class _InvalidEntry(ValueError):
    """An input value that may be retried, distinct from stream failures."""


def _positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _cast(token: str, dtype: np.dtype) -> int | float:
    if dtype.kind == "i":
        if re.fullmatch(r"[+-]?[0-9]+", token) is None:
            raise _InvalidEntry("Expected an integer")
        digits = token.lstrip("+-").lstrip("0") or "0"
        if len(digits) > 10:
            raise _InvalidEntry("Integer is outside the selected dtype")
        value = int(("-" if token.startswith("-") else "") + digits)
        info = np.iinfo(dtype)
        if not info.min <= value <= info.max:
            raise _InvalidEntry("Integer is outside the selected dtype")
        return value
    if _NUMBER.fullmatch(token) is None:
        implicit = re.fullmatch(rf"([+-]?{_MANTISSA})([+-][0-9]+)", token)
        if implicit is None:
            raise _InvalidEntry("Expected a finite number")
        token = implicit[1] + "e" + implicit[2]
    with np.errstate(over="ignore", invalid="ignore"):
        number = dtype.type(token.replace("D", "e").replace("d", "e"))
    if not np.isfinite(number):
        raise _InvalidEntry("Number is outside the finite range of the selected dtype")
    return float(number)


def _bounds(value: ArrayLike | None, size: int, dtype: np.dtype, name: str) -> NDArray | None:
    if value is None:
        return None
    array = finite(value, name)
    if array.ndim > 1 or (array.ndim == 1 and array.size not in (1, size)):
        raise ValueError(f"{name} must be scalar, length one, or length size")
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        if np.any((array != np.floor(array)) | (array < info.min) | (array > info.max)):
            raise ValueError(f"{name} must contain integers in the selected dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite in the selected dtype")
    return np.broadcast_to(array, (size,))


class CDFConsole:
    """One console's streams; caller retains ownership of all supplied streams.

    Messages are plain Python strings, not Fortran FORMAT expressions. Numeric
    input supports decimal/D exponents, commas, repeats and continuation records.
    EOF raises EOFError; exhausted retries or record/line limits raise
    CDFConsoleError. No partially initialized numeric output is returned.
    """

    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        *,
        report_stream: TextIO | None = None,
        max_attempts: int = 3,
        max_records: int = 10000,
        max_line_length: int = 10000,
    ) -> None:
        self.input = sys.stdin if input_stream is None else input_stream
        self.output = sys.stdout if output_stream is None else output_stream
        self.report = report_stream
        self.max_attempts = _positive(max_attempts, "max_attempts")
        self.max_records = _positive(max_records, "max_records")
        self.max_line_length = _positive(max_line_length, "max_line_length")

    def _read(self) -> str:
        line = self.input.readline(self.max_line_length + 2)
        if not line:
            raise EOFError("End of console input")
        line = line.removesuffix("\n").removesuffix("\r")
        if len(line) > self.max_line_length:
            raise CDFConsoleError("Console line exceeds max_line_length")
        return line

    def _write_output(self, message: str) -> str:
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        text = message.rstrip(" ") + "\n"
        self.output.write(text)
        return text

    def write_message(self, message: str) -> None:
        """Write a plain message to output and an optional report stream."""
        text = self._write_output(message)
        if self.report is not None and self.report is not self.output:
            self.report.write(text)

    def write_error(self, message: str) -> NoReturn:
        """Display the error and raise instead of terminating the process."""
        self.write_message(message)
        raise CDFConsoleError(message)

    def prompt(self) -> None:
        """Write and flush the source's nonadvancing prompt."""
        self.output.write(" > ")
        self.output.flush()

    def clear_screen(self, lines: int = 24) -> None:
        """Write blank lines, as in CDFLIB; no terminal escape sequences."""
        self.output.write("\n" * _positive(lines, "lines", allow_zero=True))

    def hold(self) -> None:
        """Display the hold message and consume one input record."""
        self.output.write("\nPress the Return or Enter key to continue ...\n")
        self.output.flush()
        self._read()

    def get_character(self, chars: str, message: str = "") -> int:
        """Return the one-based index of the first nonblank ASCII-lowered character."""
        if (
            not isinstance(chars, str)
            or not chars
            or any(not 33 <= ord(c) <= 126 or c != lower_case_char(c) for c in chars)
        ):
            raise ValueError("chars must contain nonblank lowercase ASCII choices")
        for _ in range(self.max_attempts):
            if message:
                self._write_output(message)
            self.output.write(f"Please enter one of [{chars}]:")
            self.prompt()
            line = self._read().lstrip(" ")
            if line:
                index = chars.find(lower_case_char(line[0]))
                if index >= 0:
                    return index + 1
            self.output.write("Invalid choice. Please try again.\n")
        raise CDFConsoleError("Too many invalid character responses")

    def get_yn(self, message: str = "") -> bool:
        """Read y/n using the character input contract."""
        return self.get_character("yn", message) == 1

    def get_string(self, message: str = "", *, allow_blank: bool = False) -> str:
        """Read text, skipping leading-# records and stripping inline comments.

        Leading spaces are retained. Trailing Fortran padding is removed. Unlike
        the source's fixed buffer, valid lines are not silently truncated.
        """
        if not isinstance(allow_blank, bool):
            raise ValueError("allow_blank must be boolean")
        attempts = 0
        if message:
            self._write_output(message)
        self.prompt()
        for _ in range(self.max_records):
            line = self._read()
            if line.startswith("#"):
                continue
            line = line.partition("#")[0].rstrip(" ")
            if line or allow_blank:
                return line
            attempts += 1
            if attempts >= self.max_attempts:
                raise CDFConsoleError("Too many blank string responses")
            self.output.write("Blank line not allowed. Please try again.\n")
            if message:
                self._write_output(message)
            self.prompt()
        raise CDFConsoleError("Console input exhausted max_records")

    def get_numbers(
        self,
        size: int | None = None,
        *,
        dtype: str = "float64",
        message: str = "",
        lo: ArrayLike | None = None,
        hi: ArrayLike | None = None,
        lo_eq_ok: bool = True,
        hi_eq_ok: bool = True,
    ) -> NDArray:
        """Read scalar or vector float64/float32/int32, returning an immutable array.

        size=None selects a zero-dimensional result; explicit size selects a
        vector. Bounds broadcast from scalars or length-one arrays, or match size.
        Invalid numbers or bounds restart the entire requested vector. Excess
        fields on the final record are discarded, as in list-directed input.
        Null fields and slash termination are rejected when a value is required.
        """
        count = 1 if size is None else _positive(size, "size", allow_zero=True)
        if dtype not in ("float64", "float32", "int32"):
            raise ValueError("dtype must be float64, float32 or int32")
        if not isinstance(lo_eq_ok, bool) or not isinstance(hi_eq_ok, bool):
            raise ValueError("Bound inclusion flags must be boolean")
        kind = np.dtype(dtype)
        low, high = _bounds(lo, count, kind, "lo"), _bounds(hi, count, kind, "hi")
        if low is not None and high is not None:
            if np.any(low > high) or (not (lo_eq_ok and hi_eq_ok) and np.any(low == high)):
                raise ValueError("Bounds have no admissible interval")
        if not count:
            return np.frombuffer(b"", dtype=kind)
        records = 0
        for _ in range(self.max_attempts):
            if message:
                self._write_output(message)
            self.prompt()
            values: list[int | float] = []
            try:
                while len(values) < count:
                    if records >= self.max_records:
                        raise CDFConsoleError("Console input exhausted max_records")
                    records += 1
                    line = self._read()
                    seen = False
                    for part in re.findall(r"[^,\s]+|[,\s]+", line):
                        if part[0].isspace() or part[0] == ",":
                            if part.count(",") > 1 or (not seen and "," in part):
                                raise _InvalidEntry("Null numeric fields are not supported")
                            continue
                        seen = True
                        repeat, token = 1, part
                        if "*" in part:
                            copies, token = part.split("*", 1)
                            if re.fullmatch(r"[0-9]+", copies) is None:
                                raise _InvalidEntry("Repeat count must be positive")
                            copies = copies.lstrip("0")
                            if not copies:
                                raise _InvalidEntry("Repeat count must be positive")
                            remaining = count - len(values)
                            repeat = (
                                remaining
                                if len(copies) > len(str(remaining))
                                else min(int(copies), remaining)
                            )
                        values.extend([_cast(token, kind)] * repeat)
                        if len(values) == count:
                            break
                array = np.asarray(values, dtype=kind)
                if low is not None and np.any(array < low if lo_eq_ok else array <= low):
                    raise _InvalidEntry("Input violates the lower bound")
                if high is not None and np.any(array > high if hi_eq_ok else array >= high):
                    raise _InvalidEntry("Input violates the upper bound")
            except _InvalidEntry as error:
                self.output.write(f"{error}. Please try again from the beginning.\n")
                continue
            return np.frombuffer(array.tobytes(), dtype=kind).reshape(
                () if size is None else (count,)
            )
        raise CDFConsoleError("Too many invalid numeric responses")
