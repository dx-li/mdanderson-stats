"""ACCFLF named-table input and explicit covariate selection."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .cdflib_strings import qlex


def _column(names: tuple[str, ...], value: str | int) -> int:
    if isinstance(value, str):
        try:
            return names.index(value.upper())
        except ValueError as exc:
            raise ValueError(f"unknown column {value!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(names):
        raise ValueError("column must be a name or zero-based integer index")
    return value


@dataclass(frozen=True)
class AccflfData:
    names: tuple[str, ...]
    values: FloatArray
    time_column: int
    event_column: int
    multiplicity_column: int | None
    selected_columns: tuple[int, ...]

    @property
    def time(self) -> FloatArray:
        return self.values[:, self.time_column]

    @property
    def event(self) -> FloatArray:
        return self.values[:, self.event_column]

    @property
    def weights(self) -> FloatArray:
        return (
            _freeze(np.ones(self.values.shape[0]))
            if self.multiplicity_column is None
            else self.values[:, self.multiplicity_column]
        )

    @property
    def covariates(self) -> FloatArray:
        return _freeze(self.values[:, self.selected_columns])

    @property
    def covariate_names(self) -> tuple[str, ...]:
        return tuple(self.names[i] for i in self.selected_columns)

    @property
    def available_covariates(self) -> tuple[str, ...]:
        occupied = (
            self.time_column,
            self.event_column,
            self.multiplicity_column,
            *self.selected_columns,
        )
        return tuple(name for i, name in enumerate(self.names) if i not in occupied)

    def select(self, columns: tuple[str | int, ...]) -> "AccflfData":
        """Replace the selected covariates; an empty tuple gives an intercept-only model."""
        indices = tuple(_column(self.names, value) for value in columns)
        if (
            len(indices) > 16
            or len(set(indices)) != len(indices)
            or any(
                i in (self.time_column, self.event_column, self.multiplicity_column)
                for i in indices
            )
        ):
            raise ValueError("select <=16 distinct covariates, excluding time/status/multiplicity")
        return replace(self, selected_columns=indices)

    def add(self, columns: tuple[str | int, ...]) -> "AccflfData":
        """Add unused covariates without mutating the original dataset or fit."""
        return self.select((*self.selected_columns, *columns))


def read_accflf_data(
    path: str | Path,
    *,
    time: str | int = "TIME",
    event: str | int = "STATUS",
    multiplicity: str | int | None = "auto",
    covariates: tuple[str | int, ...] = (),
    comment: str = "#",
) -> AccflfData:
    """Read original numeric tables with an optional identifier header.

    Uses the existing native-compatible QLEX lexer. Blank/full-line comments
    are ignored; mixed text/numeric records and ragged rows are errors. Header
    names are uppercase and truncated to eight characters as in read_table_mod.
    Headerless columns are X1, X2, ...; specify their roles explicitly.
    """
    if not isinstance(comment, str) or len(comment) != 1 or comment.isspace():
        raise ValueError("comment must be one non-whitespace character")
    names: tuple[str, ...] | None = None
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            tokens = list(qlex(line))
            numeric = all(t.kind in ("IN", "RL") for t in tokens)
            if not 1 <= len(tokens) <= 200:
                raise ValueError(f"line {number}: require 1..200 fields")
            if names is None:
                if all(t.kind == "ID" for t in tokens):
                    names = tuple(t.text.upper()[:8] for t in tokens)
                    if len(set(names)) != len(names):
                        raise ValueError("duplicate column names after eight-character truncation")
                    continue
                names = tuple(f"X{i + 1}" for i in range(len(tokens)))
            if not numeric or len(tokens) != len(names):
                raise ValueError(f"line {number}: require {len(names)} numeric fields")
            row = [
                float(t.integer) if t.kind == "IN" and t.integer is not None else t.real
                for t in tokens
            ]
            if any(v is None or not np.isfinite(v) for v in row) or any(
                t.overflow or t.underflow for t in tokens
            ):
                raise ValueError(f"line {number}: numeric value exceeds supported range")
            rows.append(row)
            if len(rows) > 20000:
                raise ValueError("at most 20000 data rows are supported")
    if names is None or not rows:
        raise ValueError("data file must contain numeric rows")
    ti, ei = _column(names, time), _column(names, event)
    if multiplicity == "auto":
        mi = names.index("MULTI") if "MULTI" in names else None
    else:
        mi = None if multiplicity is None else _column(names, multiplicity)
    occupied = [ti, ei] + ([] if mi is None else [mi])
    if len(set(occupied)) != len(occupied):
        raise ValueError("time, status and multiplicity columns must be distinct")
    values = np.asarray(rows, dtype=float)
    if np.any(values[:, ti] <= 0) or np.any((values[:, ei] != 0) & (values[:, ei] != 1)):
        raise ValueError("times must be positive and status must be zero or one")
    if mi is not None and (
        np.any(values[:, mi] < 1) or np.any(values[:, mi] != np.floor(values[:, mi]))
    ):
        raise ValueError("multiplicity must contain positive integers")
    return AccflfData(names, _freeze(values), ti, ei, mi, ()).select(covariates)
