"""Indexed reproduction of the integer streams embedded in RANLIST."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count

_M1, _M2 = 2147483563, 2147483399
_DEFAULT = (1234567890, 123456789)


def ranlist_seeds(phrase: str) -> tuple[int, int]:
    """Apply PHRTSD to ASCII text, ignoring trailing spaces (not other whitespace).

    Empty/all-space phrases retain the original default pair. Unknown ASCII
    characters use the source's fallback character code; Unicode is rejected
    because the original operates on single-byte characters.
    """
    if not isinstance(phrase, str) or not phrase.isascii():
        raise ValueError("phrase must be an ASCII string")
    table = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+[];:'\"<>?,./"
    )
    first, second = _DEFAULT
    for character in phrase.rstrip(" "):
        code = (table.find(character) + 1) % 64 or 63
        values = [(code - j - 1) % 63 + 1 for j in range(1, 6)]
        for j in range(5):
            first = (first + 64**j * values[j]) % 2**30
            second = (second + 64**j * values[4 - j]) % 2**30
    return first, second


def _mod_power(base: int, exponents: NDArray[np.int64], modulus: int) -> NDArray[np.int64]:
    power = exponents.copy()
    result = np.ones(power.shape, dtype=np.int64)
    while np.any(power):
        result = np.where(power & 1, result * base % modulus, result)
        power //= 2
        base = base * base % modulus
    return result


def _stream_seeds(seed: tuple[int, int], stream: int) -> tuple[int, int]:
    if (
        isinstance(stream, (bool, np.bool_))
        or not isinstance(stream, (int, np.integer))
        or not 1 <= stream <= 32
    ):
        raise ValueError("stream must be an integer from one through 32")
    pair = np.asarray(seed, dtype=object)
    if pair.shape != (2,) or any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pair
    ):
        raise ValueError("seed must contain two integers")
    first, second = map(int, pair)
    if not 1 <= first < _M1 or not 1 <= second < _M2:
        raise ValueError("seed values must be in 1..2147483562 and 1..2147483398")
    first = first * pow(2082007225, int(stream) - 1, _M1) % _M1
    second = second * pow(784306273, int(stream) - 1, _M2) % _M2
    return first, second


def ranlist_integers(
    positions: ArrayLike, *, seed: tuple[int, int] = _DEFAULT, stream: int = 1
) -> NDArray[np.int64]:
    """Return original IGNLGI draws 1..2147483562 at one-based positions.

    Stream numbers are 1..32, spaced by 2**50 draws as in SETALL. Positions
    retain input shape and must be positive integers below 2**53. There is no
    shared mutable state. Vectorized modular exponentiation costs logarithmic
    work in the largest position; products fit exactly in signed int64.
    """
    positions_array = count(positions, "positions")
    if np.any(positions_array < 1):
        raise ValueError("positions must be positive one-based draw numbers")
    first, second = _stream_seeds(seed, stream)
    exponent = positions_array.astype(np.int64)
    s1 = first * _mod_power(40014, exponent, _M1) % _M1
    s2 = second * _mod_power(40692, exponent, _M2) % _M2
    difference = s1 - s2
    result = np.where(difference < 1, difference + _M1 - 1, difference)
    result.flags.writeable = False
    return result


def ranlist_uniform(
    positions: ArrayLike,
    *,
    seed: tuple[int, int] = _DEFAULT,
    stream: int = 1,
    legacy: bool = False,
) -> FloatArray:
    """Indexed uniforms; legacy reproduces RANF's single-precision rounding.

    Default divides integer draws by 2147483563 in double precision, keeping
    values strictly inside (0,1). Legacy rounds both the integer conversion
    and scale constant to float32, as in the archived RANF.
    """
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    integers = ranlist_integers(positions, seed=seed, stream=stream)
    result = np.asarray(
        (integers.astype(np.float32) * np.float32(4.656613057e-10)).astype(np.float64)
        if legacy
        else integers / _M1
    )
    result.flags.writeable = False
    return result


def ranlist_starting_seeds(phrase: str) -> tuple[int, int]:
    """Apply GTSEED's phrase entry behavior: 31 ASCII characters and zero repair.

    The original input field truncates longer phrases before PHRTSD. A zero
    first seed is replaced by 1, and a zero second seed by 12. Use ranlist_seeds
    for the unmodified hash of an arbitrary-length ASCII phrase instead.
    """
    if not isinstance(phrase, str) or not phrase.isascii():
        raise ValueError("phrase must be an ASCII string")
    first, second = ranlist_seeds(phrase[:31])
    return first or 1, second or 12
