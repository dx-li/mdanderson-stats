"""Balanced RANLIST allocation, with explicit source compatibility."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count
from .ranlist_random import _DEFAULT, _M1, _M2, _stream_seeds


class _AllocationStream:
    """Local integer state; no process-wide generator or mutable default."""

    def __init__(self, state: tuple[int, int]) -> None:
        self.first, self.second = state

    def advance(self, draws: int) -> None:
        self.first = self.first * pow(40014, draws, _M1) % _M1
        self.second = self.second * pow(40692, draws, _M2) % _M2

    def integer(self, low: int, high: int, legacy: bool) -> int:
        if low == high:
            return low
        width = high - low + 1
        # Callers bound width by 500. Source accepts one extra residue zero.
        limit = ((_M1 - 2 if legacy else _M1 - 1) // width) * width
        while True:
            self.advance(1)
            value = self.first - self.second
            if value < 1:
                value += _M1 - 1
            value -= 1
            if value <= limit if legacy else value < limit:
                return low + value % width

    def permute(
        self, natural: NDArray[np.int64], multiplier: int, legacy: bool
    ) -> NDArray[np.int64]:
        block = np.tile(natural, multiplier)
        for i in range(len(block) - 1):
            j = self.integer(i, len(block) - 1, legacy)
            block[i], block[j] = block[j], block[i]
        return block


@dataclass(frozen=True)
class RestrictedAllocation:
    patients: NDArray[np.int64]
    treatments: NDArray[np.int64]
    counts: NDArray[np.int64]
    block_start: NDArray[np.int64]
    block_end: NDArray[np.int64]
    multiplier: NDArray[np.int64]
    balance: tuple[int, int]
    seed: tuple[int, int]
    stream: int
    legacy: bool


def ranlist_restricted(
    patients: ArrayLike,
    counts: ArrayLike,
    *,
    balance: tuple[int, int] = (1, 1),
    seed: tuple[int, int] = _DEFAULT,
    stream: int = 1,
    legacy: bool = False,
    max_blocks: int = 10_000,
) -> RestrictedAllocation:
    """Assign treatments by permuting balanced blocks of integer counts.

    Patients and treatments are one-based; input shape/order are preserved.
    Each block repeats the natural treatment block K times, where K is drawn
    uniformly from inclusive ``balance`` bounds on each refill. A block may
    contain at most 500 assignments, as in RANLIST. Equal bounds fix K.

    Legacy reproduces IGTRT: K is selected once per stream, integer sampling
    has the source's endpoint bias, and each preceding block skips exactly
    length-minus-one draws even if permutation sampling would reject a draw.
    Default follows the manual's refill rule and unbiased integer sampling.
    ``max_blocks`` bounds generated permutations (including preceding blocks
    in default mode); increase deliberately for longer lists.
    """
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    if (
        isinstance(max_blocks, (bool, np.bool_))
        or not isinstance(max_blocks, (int, np.integer))
        or max_blocks < 1
    ):
        raise ValueError("max_blocks must be a positive integer")
    state = _stream_seeds(seed, stream)
    patient = count(patients, "patients").astype(np.int64)
    if np.any(patient < 1):
        raise ValueError("patients must be positive one-based numbers")
    repetitions = count(counts, "counts").astype(np.int64)
    if repetitions.ndim != 1 or not 1 <= repetitions.size <= 20 or np.any(repetitions < 1):
        raise ValueError("counts must be a vector of 1..20 positive integers")
    bounds = count(balance, "balance").astype(np.int64)
    if bounds.shape != (2,) or not 1 <= bounds[0] <= bounds[1]:
        raise ValueError("balance must contain ordered positive integer bounds")
    low, high = map(int, bounds)
    size = sum(map(int, repetitions))
    if size * high > 500:
        raise ValueError("the largest balance block must contain at most 500 assignments")
    natural = np.repeat(np.arange(1, len(repetitions) + 1, dtype=np.int64), repetitions)
    unique, inverse = np.unique(patient, return_inverse=True)
    assigned = np.empty(unique.size, dtype=np.int64)
    starts, ends, multipliers = (np.empty_like(assigned) for _ in range(3))
    generator = _AllocationStream(state)
    if legacy:
        multiplier = generator.integer(low, high, True)
        length = size * multiplier
        block_indices = (unique - 1) // length
        distinct = np.unique(block_indices)
        if distinct.size > max_blocks:
            raise ValueError("requested blocks exceed max_blocks")
        initial = (generator.first, generator.second)
        for index in distinct:
            generator = _AllocationStream(initial)
            generator.advance(int(index) * (length - 1))
            block = generator.permute(natural, multiplier, True)
            left, right = np.searchsorted(block_indices, [index, index + 1])
            start = int(index) * length + 1
            assigned[left:right] = block[unique[left:right] - start]
            starts[left:right], ends[left:right] = start, start + length - 1
            multipliers[left:right] = multiplier
    elif unique.size:
        if int(unique[-1]) > int(max_blocks) * size * high:
            raise ValueError("preceding blocks exceed max_blocks")
        start, left = 1, 0
        for _ in range(int(max_blocks)):
            multiplier = generator.integer(low, high, False)
            block = generator.permute(natural, multiplier, False)
            end = start + len(block) - 1
            right = int(np.searchsorted(unique, end, side="right"))
            assigned[left:right] = block[unique[left:right] - start]
            starts[left:right], ends[left:right] = start, end
            multipliers[left:right] = multiplier
            if right == unique.size:
                break
            start, left = end + 1, right
        else:
            raise ValueError("preceding blocks exceed max_blocks")
    outputs = [
        np.asarray(value[inverse]).reshape(patient.shape)
        for value in (assigned, starts, ends, multipliers)
    ]
    for value in (patient, repetitions, *outputs):
        value.flags.writeable = False
    return RestrictedAllocation(
        patient,
        outputs[0],
        repetitions,
        outputs[1],
        outputs[2],
        outputs[3],
        (low, high),
        (int(seed[0]), int(seed[1])),
        int(stream),
        bool(legacy),
    )
