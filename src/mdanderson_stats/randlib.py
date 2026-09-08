"""Explicit RANDLIB generator banks with original stream/block controls."""

import numpy as np
from numpy.typing import NDArray

from .ranlist_random import _DEFAULT, _M1, _M2, _mod_power, _stream_seeds


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
        positions = np.arange(1, size + 1, dtype=np.int64)
        a, b = self.get_seeds()
        first = a * _mod_power(40014, positions, _M1) % _M1
        second = b * _mod_power(40692, positions, _M2) % _M2
        difference = first - second
        result = np.where(difference < 1, difference + _M1 - 1, difference)
        if self._antithetic[self._stream - 1]:
            result = _M1 - result
        if size:
            self._current[self._stream - 1] = (int(first[-1]), int(second[-1]))
        result.flags.writeable = False
        return result

    def uniform(self, size: int = 1, *, legacy: bool = False) -> NDArray[np.float64]:
        """Consume uniforms; legacy reproduces original RANF float32 scaling."""
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("legacy must be boolean")
        raw = self.integers(size)
        result = (
            (raw.astype(np.float32) * np.float32(4.656613057e-10)).astype(np.float64)
            if legacy
            else raw / _M1
        )
        result.flags.writeable = False
        return result
