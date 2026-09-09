"""Bounded, transactional numeric-list state for CDFLIB console editing."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .cdflib_sort import sort_list


class _ListEditError(ValueError):
    """A rejected list action, distinct from stream and system errors."""


def _index(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _ListEditError(f"{name} must be an integer >= {minimum}")
    return value


class CDFNumberList:
    """One list's capacity, bounds and last successfully applied edits.

    Each action validates before mutation. The caller can inspect values after
    EOF or another console exception without losing prior successful edits.
    Values snapshots are immutable owned arrays; supplied inputs are never mutated.
    """

    def __init__(
        self,
        values: ArrayLike = (),
        *,
        max_size: int = 1000,
        lo: float | None = None,
        hi: float | None = None,
        lo_eq_ok: bool = True,
        hi_eq_ok: bool = True,
    ) -> None:
        self._max_size = _index(max_size, "max_size")
        if self._max_size > np.iinfo(np.int32).max:
            raise ValueError("max_size exceeds the console's int32 index range")
        self._lo = None if lo is None else scalar(lo, "lo")
        self._hi = None if hi is None else scalar(hi, "hi")
        if not isinstance(lo_eq_ok, bool) or not isinstance(hi_eq_ok, bool):
            raise ValueError("Bound inclusion flags must be boolean")
        self._lo_eq_ok, self._hi_eq_ok = lo_eq_ok, hi_eq_ok
        if self._lo is not None and self._hi is not None:
            if self._lo > self._hi or (self._lo == self._hi and not (lo_eq_ok and hi_eq_ok)):
                raise ValueError("Bounds have no admissible interval")
        initial = self._validate(values)
        if initial.size > self._max_size:
            raise ValueError("Initial list exceeds max_size")
        self._storage = np.empty(self._max_size, dtype=float)
        self._size = initial.size
        self._storage[: self._size] = initial

    @property
    def values(self) -> FloatArray:
        """Immutable snapshot of all currently retained values."""
        return _freeze(self._storage[: self._size])

    @property
    def size(self) -> int:
        return self._size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def lo(self) -> float | None:
        return self._lo

    @property
    def hi(self) -> float | None:
        return self._hi

    @property
    def lo_eq_ok(self) -> bool:
        return self._lo_eq_ok

    @property
    def hi_eq_ok(self) -> bool:
        return self._hi_eq_ok

    def _validate(self, values: ArrayLike) -> FloatArray:
        array = finite(values, "values")
        if array.ndim != 1:
            raise _ListEditError("values must be one-dimensional")
        if self._lo is not None and np.any(
            array < self._lo if self._lo_eq_ok else array <= self._lo
        ):
            raise _ListEditError("List values violate the lower bound")
        if self._hi is not None and np.any(
            array > self._hi if self._hi_eq_ok else array >= self._hi
        ):
            raise _ListEditError("List values violate the upper bound")
        return array

    def append(self, values: ArrayLike) -> None:
        """Append a finite one-dimensional batch atomically."""
        array = self._validate(values)
        if array.size > self.max_size - self.size:
            raise _ListEditError("List capacity exceeded")
        end = self.size + array.size
        self._storage[self.size : end] = array
        self._size = end

    def append_spaced(
        self, start: float, stop: float, intervals: int, *, log: bool = False
    ) -> None:
        """Append intervals+1 points, including exact endpoints, in either direction."""
        intervals = _index(intervals, "intervals", 1)
        if intervals >= self.max_size - self.size:
            raise _ListEditError("List capacity exceeded")
        if not isinstance(log, bool):
            raise ValueError("log must be boolean")
        start, stop = scalar(start, "start"), scalar(stop, "stop")
        self._validate([start, stop])
        if log and min(start, stop) <= 0:
            raise _ListEditError("Logarithmic spacing requires positive endpoints")
        weights = np.arange(intervals + 1, dtype=float) / intervals
        if start == stop:
            result = np.full(intervals + 1, start)
        elif log and max(start, stop) / 2 <= min(start, stop):
            # Close endpoints need a relative log; subtracting their large
            # absolute logarithms would discard their small separation.
            delta = np.log1p((stop - start) / start)
            with np.errstate(over="ignore"):
                result = start * np.exp(weights * delta)
            result = np.clip(result, min(start, stop), max(start, stop))
        elif log:
            # Logs avoid overflow/underflow in stop/start. Work from the nearer
            # endpoint to retain precision on short intervals at large scales.
            first, last = np.log(start), np.log(stop)
            near_start = weights <= 0.5
            exponents = np.where(
                near_start, first + weights * (last - first), last - (1 - weights) * (last - first)
            )
            with np.errstate(over="ignore"):
                result = np.exp(exponents)
            result = np.clip(result, min(start, stop), max(start, stop))
        else:
            # Power-of-two normalization prevents both overflow for opposite
            # extremes and premature underflow of subnormal weighted terms.
            exponent = int(np.frexp(max(abs(start), abs(stop)))[1])
            a, b = np.ldexp(start, -exponent), np.ldexp(stop, -exponent)
            with np.errstate(over="ignore"):
                result = np.ldexp((1 - weights) * a + weights * b, exponent)
            result = np.clip(result, min(start, stop), max(start, stop))
        result[0], result[-1] = start, stop
        self.append(result)

    def delete(self, first: int, last: int | None = None) -> None:
        """Delete an inclusive one-based range; reversed bounds are accepted."""
        first = _index(first, "first", 1)
        last = first if last is None else _index(last, "last", 1)
        low, high = min(first, last), max(first, last)
        if high > self.size:
            raise _ListEditError("Deletion index exceeds list size")
        count = high - low + 1
        self._storage[low - 1 : self.size - count] = self._storage[high : self.size]
        self._size -= count

    def sort_unique(self) -> None:
        """Sort and drop near-duplicates against the last retained representative.

        Source criterion: abs(x-y)/max(abs(x)+abs(y),1e-100) < 1e-14.
        Scaling avoids overflow; comparing against retained representatives
        preserves the source's nontransitive grouping rule.
        """
        if not self.size:
            raise _ListEditError("Cannot sort an empty list")
        ordered = sort_list(self._storage[: self.size])
        assert isinstance(ordered, np.ndarray)
        kept = [float(ordered[0])]
        for candidate in ordered[1:]:
            value, previous = float(candidate), kept[-1]
            scale = max(abs(value), abs(previous), 1e-100)
            u, v = value / scale, previous / scale
            if abs(u - v) >= 1e-14 * max(abs(u) + abs(v), 1e-100 / scale):
                kept.append(value)
        self._storage[: len(kept)] = kept
        self._size = len(kept)
