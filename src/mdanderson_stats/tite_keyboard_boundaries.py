"""Precomputed effective-follow-up transition boundaries for TITE-Keyboard."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .keyboard import KeyboardDesign


@dataclass(frozen=True)
class TITEKeyboardBoundaries:
    """Transition brackets indexed by observed DLT count y, including zero.

    Columns are [last value before transition, first value after transition],
    in effective NON-DLT count m. Adjacent floating-point endpoints bracket each
    transition. [0,0] means the new decision already applies at zero; [NaN,NaN]
    means it cannot occur within effective n <= max_patients. Safety, suspension,
    dose-range and precision rules are separate from these posterior transitions.
    """

    toxicities: NDArray[np.int64]
    stay_or_escalate: FloatArray
    escalate: FloatArray
    enrolled_patients: NDArray[np.int64]
    eliminate_min: NDArray[np.int64]
    lowest_stop_min: NDArray[np.int64]
    max_patients: int

    def moves(self, toxicities: ArrayLike, effective_nontoxic: ArrayLike) -> NDArray[np.int64]:
        """Batched ordinary moves from the table: +1/escalate, 0/stay, -1/de-escalate."""
        y, m = np.broadcast_arrays(
            count(toxicities, "toxicities"), finite(effective_nontoxic, "effective_nontoxic")
        )
        if np.any((m < 0) | (y + m > self.max_patients)):
            raise ValueError("require effective_nontoxic >= 0 and effective total <= max_patients")
        indices = y.astype(np.int64)
        move = np.where(
            m >= self.escalate[indices, 1],
            1,
            np.where(m >= self.stay_or_escalate[indices, 1], 0, -1),
        )
        return _owned(move)


def _transitions(
    design: KeyboardDesign, y: FloatArray, maximum: FloatArray, minimum_move: int
) -> FloatArray:
    low = np.zeros(y.shape)
    high = maximum.copy()
    at_zero = design._posterior_effective(y, y).move >= minimum_move
    reachable = design._posterior_effective(y + high, y).move >= minimum_move
    search = reachable & ~at_zero
    for _ in range(128):
        active = search & (np.nextafter(low, np.inf) < high)
        if not np.any(active):
            break
        middle = low + (high - low) / 2
        moved = design._posterior_effective(y + middle, y).move >= minimum_move
        high = np.where(active & moved, middle, high)
        low = np.where(active & ~moved, middle, low)
    else:
        raise ArithmeticError("effective-follow-up transition did not converge")
    low = np.where(at_zero, 0, low)
    high = np.where(at_zero, 0, high)
    bracket = np.column_stack((low, high))
    bracket[~reachable] = np.nan
    return _owned(bracket)


def tite_keyboard_boundaries(
    design: KeyboardDesign, max_patients: int = 30
) -> TITEKeyboardBoundaries:
    """Find follow-up thresholds without rounding the observed DLT count or ESS.

    The ordinary decision is monotone as effective non-DLT follow-up increases.
    Vectorized bisection locates both transitions to adjacent floating values,
    using the same posterior-key kernel and conservative tie rule as conduct.
    The thresholds are independent of window length and follow-up weighting.
    """
    if not isinstance(design, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
    maximum = scalar(max_patients, "max_patients")
    if maximum != int(maximum) or not 1 <= maximum <= 200:
        raise ValueError("max_patients must be an integer in [1,200]")
    y = np.arange(int(maximum) + 1, dtype=float)
    available = maximum - y
    safety = design._shared.boundary_table(int(maximum))
    return TITEKeyboardBoundaries(
        _owned(y.astype(np.int64)),
        _transitions(design, y, available, 0),
        _transitions(design, y, available, 1),
        safety.patients,
        safety.eliminate_min,
        safety.lowest_stop_min,
        int(maximum),
    )
