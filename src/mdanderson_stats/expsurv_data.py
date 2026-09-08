"""Named numerical tables replacing EXPSURV's file dialogs and global variables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


@dataclass(frozen=True, init=False)
class ExploratoryTable:
    names: tuple[str, ...]
    values: FloatArray

    def __init__(self, names: tuple[str, ...] | list[str], values: ArrayLike) -> None:
        names = tuple(names)
        if not names or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("names must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        matrix = finite(values, "values").copy()
        if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] != len(names):
            raise ValueError("values must be nonempty rows with one column per name")
        matrix.flags.writeable = False
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", matrix)

    def column(self, name: str) -> FloatArray:
        """Return a read-only named column; unknown names raise KeyError."""
        try:
            index = self.names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.values[:, index]

    def cosort(self, name: str) -> Self:
        """Stably sort all columns by one named column, preserving ties and alignment."""
        order = np.argsort(self.column(name), kind="stable")
        return type(self)(self.names, self.values[order])

    def write(self, path: str | Path) -> None:
        """Write headerless whitespace-separated rows with round-trip float precision."""
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            for row in self.values:
                stream.write(" ".join(format(value, ".17g") for value in row) + "\n")

    @classmethod
    def read(cls, path: str | Path, names: tuple[str, ...] | list[str]) -> Self:
        """Read EXPSURV ASCII rows; reject missing, ragged or nonfinite data."""
        rows = []
        with Path(path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != len(names):
                    raise ValueError(
                        f"line {number}: expected {len(names)} columns, got {len(fields)}"
                    )
                try:
                    row = [float(value) for value in fields]
                except ValueError as error:
                    raise ValueError(f"line {number}: expected numeric values") from error
                if not np.all(np.isfinite(row)):
                    raise ValueError(
                        f"line {number}: nonfinite or missing values are not supported"
                    )
                rows.append(row)
        return cls(names, rows)
