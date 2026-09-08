"""Explicit RANDLIB generator banks with original stream/block controls."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._randlib_sampling import bounded, raw_batch
from ._validation import scalar
from .ranlist_random import _DEFAULT, _M1, _M2, _stream_seeds


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


class RandlibGenerator:
    """A private bank of 32 generators; no module-global state.

    State-changing methods affect only the selected stream unless named
    set_all_seeds. Draw arrays are read-only. A bank is mutable and should not
    be shared by concurrent callers without external coordination.
    """

    def __init__(
        self, seed: tuple[int, int] = _DEFAULT, *, stream: int = 1, max_draws: int = 1_000_000
    ) -> None:
        self._stream = _integer(stream, "stream", 1, 32)
        self._max_draws = _integer(max_draws, "max_draws", 1, 2**53 - 1)
        self._antithetic = [False] * 32
        self.set_all_seeds(seed)

    @property
    def stream(self) -> int:
        return self._stream

    def select(self, stream: int) -> None:
        """Select an existing stream without resetting it (SETCGN)."""
        self._stream = _integer(stream, "stream", 1, 32)

    def get_seeds(self) -> tuple[int, int]:
        """Return the current component states, not the original seed pair (GETSD)."""
        return self._current[self._stream - 1]

    def set_seeds(self, seed: tuple[int, int]) -> None:
        """Replace selected stream's initial, block and current states (SETSD)."""
        state = _stream_seeds(seed, 1)
        i = self._stream - 1
        self._initial[i] = self._block[i] = self._current[i] = state

    def set_all_seeds(self, seed: tuple[int, int]) -> None:
        """Initialize all streams, retaining selected stream and antithetic flags."""
        initial = [_stream_seeds(seed, stream) for stream in range(1, 33)]
        self._initial = initial
        self._block = initial.copy()
        self._current = initial.copy()

    def set_antithetic(self, enabled: bool) -> None:
        """Complement selected stream's future raw draws as M1-z (SETANT)."""
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be boolean")
        self._antithetic[self._stream - 1] = bool(enabled)

    def reinitialize(self, mode: int = -1) -> None:
        """INITGN: -1 original state, 0 current block start, 1 next block start.

        The next block is 2**30 draws from the current block's start, regardless
        of the number of draws already consumed within that block.
        """
        mode = _integer(mode, "mode", -1, 1)
        i = self._stream - 1
        if mode == -1:
            self._block[i] = self._initial[i]
        elif mode == 1:
            a, b = self._block[i]
            self._block[i] = (a * 1033780774 % _M1, b * 1494757890 % _M2)
        self._current[i] = self._block[i]

    def advance_state(self, exponent: int) -> None:
        """ADVNST: advance by 2**exponent draws and adopt that as the initial state.

        This also resets the selected block start. Modular exponent reduction
        uses the prime component moduli; it does not materialize 2**exponent.
        """
        exponent = _integer(exponent, "exponent", 0, 2**53 - 1)
        a, b = self.get_seeds()
        self.set_seeds(
            (
                a * pow(40014, pow(2, exponent, _M1 - 1), _M1) % _M1,
                b * pow(40692, pow(2, exponent, _M2 - 1), _M2) % _M2,
            )
        )

    def integers(self, size: int = 1) -> NDArray[np.int64]:
        """Consume IGNLGI draws in 1..2147483562; zero size leaves state unchanged.

        Vectorized modular powers compute a batch and update its final state
        once. max_draws bounds each batch before allocating memory.
        """
        size = _integer(size, "size", 0, self._max_draws)
        result, final = raw_batch(self.get_seeds(), size, self._antithetic[self._stream - 1])
        self._current[self._stream - 1] = final
        result.flags.writeable = False
        return result

    def uniform(
        self,
        size: int = 1,
        *,
        low: float = 0.0,
        high: float = 1.0,
        legacy: bool = False,
        source: Literal["fortran", "c"] = "fortran",
    ) -> NDArray[np.float64]:
        """GENUNF bounded uniforms; equal bounds still consume draws.

        Modern uses overflow-resistant interpolation. Legacy applies float32 bound,
        difference, product and sum rounding. Select Fortran or C raw-uniform
        scaling with source; C scaling can round to one. Floating-point rounding can reach
        a bound even though the underlying raw uniforms are strictly interior.
        """
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("legacy must be boolean")
        if not isinstance(source, str) or source not in ("fortran", "c"):
            raise ValueError("source must be fortran or c")
        if not legacy and source != "fortran":
            raise ValueError("source selection requires legacy=True")
        low, high = scalar(low, "low"), scalar(high, "high")
        if low > high:
            raise ValueError("low must not exceed high")
        if legacy:
            with np.errstate(over="ignore", invalid="ignore"):
                lower, upper = np.float32(low), np.float32(high)
                width = upper - lower
            if not np.isfinite(lower) or not np.isfinite(upper) or not np.isfinite(width):
                raise ValueError("legacy bounds and their difference must fit float32")
        raw = self.integers(size)
        if legacy:
            u = (
                (raw.astype(np.float64) * 4.656613057e-10).astype(np.float32)
                if source == "c"
                else raw.astype(np.float32) * np.float32(4.656613057e-10)
            )
            result = (lower + width * u).astype(np.float64)
        else:
            u = raw / _M1
            if low == high:
                result = np.full(size, low, dtype=np.float64)
            elif low >= 0 or high <= 0:
                result = low + (high - low) * u
            else:
                result = (1.0 - u) * low + u * high
        result.flags.writeable = False
        return result

    def integer_uniform(
        self,
        low: int,
        high: int,
        size: int = 1,
        *,
        legacy: bool = False,
        max_attempts: int | None = None,
    ) -> NDArray[np.int64]:
        """Inclusive bounded integers, with unbiased sampling by default.

        Legacy preserves IGNUIN's biased inclusive rejection boundary. Failed
        rejection budgets leave state unchanged. Equal bounds consume no draws.
        Bounds are signed 32-bit integers with width at most 2147483562.
        """
        low = _integer(low, "low", -(2**31), 2**31 - 1)
        high = _integer(high, "high", low, 2**31 - 1)
        if high - low + 1 > _M1 - 1:
            raise ValueError("integer interval width cannot exceed 2147483562")
        size = _integer(size, "size", 0, self._max_draws)
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("legacy must be boolean")
        budget = _integer(
            max(100_000, 4 * size) if max_attempts is None else max_attempts,
            "max_attempts",
            1,
            2**53 - 1,
        )
        result, state, _ = bounded(
            self.get_seeds(),
            size,
            low,
            high,
            self._antithetic[self._stream - 1],
            bool(legacy),
            budget,
        )
        self._current[self._stream - 1] = state
        result.flags.writeable = False
        return result

    def permutation(
        self,
        values: ArrayLike,
        *,
        legacy: bool = False,
        max_attempts: int | None = None,
    ) -> NDArray[np.int64]:
        """Forward GENPRM shuffle of a one-dimensional signed-32-bit integer array.

        Input is copied. Empty/singleton inputs consume no draws. Rejection
        budgets cover the complete permutation; failure leaves state unchanged.
        """
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise ValueError("values must be a one-dimensional integer array")
        size = _integer(len(array), "size", 0, min(self._max_draws, _M1 - 1))
        result = np.array(
            [_integer(value, "values", -(2**31), 2**31 - 1) for value in array], dtype=np.int64
        )
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("legacy must be boolean")
        budget = _integer(
            max(100_000, 4 * size) if max_attempts is None else max_attempts,
            "max_attempts",
            1,
            2**53 - 1,
        )
        state = self.get_seeds()
        used = 0
        for i in range(size - 1):
            draw, state, attempts = bounded(
                state,
                1,
                i,
                size - 1,
                self._antithetic[self._stream - 1],
                bool(legacy),
                budget - used,
            )
            j = int(draw[0])
            result[i], result[j] = result[j], result[i]
            used += attempts
        self._current[self._stream - 1] = state
        result.flags.writeable = False
        return result
