"""Calibrate SEQBIN posterior cutoffs to discrete null rejection probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .seqbin import SeqBinDesign, SeqBinProperties


@dataclass(frozen=True)
class SeqBinCalibrationPoint:
    design: SeqBinDesign
    significance: float


@dataclass(frozen=True)
class SeqBinCalibration:
    chosen: SeqBinCalibrationPoint
    lower: SeqBinCalibrationPoint | None
    upper: SeqBinCalibrationPoint | None
    requested_significance: float
    selection: str
    evaluations: int


def seqbin_calibrate(
    max_subjects: int,
    significance: float,
    *,
    prior: ArrayLike = (1, 1),
    null_probability: float = 0.2,
    alternative: str = "greater",
    looks: ArrayLike | None = None,
    selection: str = "conservative",
    tail_bounds: ArrayLike = (1e-6, 1 - 1e-6),
    legacy_bounds: bool = False,
) -> SeqBinCalibration:
    """Find the largest attainable level <= target, or the nearest level.

    A common posterior tail cutoff is varied within tail_bounds. For two-sided
    designs this calibrates total rejection probability, not equal tail errors.
    lower/upper bracket the target with attainable designs; an absent endpoint
    means the entire allowed range lies on the other side. Nearest ties select
    the lower level. Conservative selection raises if no allowed design is feasible.

    Search terminates at adjacent binary64 cutoffs rather than treating a jump
    as a continuous root. Comparisons use the computed beta probabilities and
    forward-recursion significance, without claiming exact real arithmetic.
    """
    alpha = scalar(significance, "significance")
    bounds = finite(tail_bounds, "tail_bounds")
    if not 0 <= alpha <= 1:
        raise ValueError("significance must lie in [0,1]")
    if bounds.shape != (2,) or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError("tail_bounds must be two increasing probabilities in (0,1)")
    if selection not in ("conservative", "nearest"):
        raise ValueError("selection must be conservative or nearest")
    cache: dict[tuple[bytes, bytes], float] = {}

    def evaluate(cutoff: float) -> SeqBinCalibrationPoint:
        design = SeqBinDesign(
            max_subjects,
            prior=prior,
            null_probability=null_probability,
            alternative=alternative,
            looks=looks,
            tail_probability=cutoff,
            legacy_bounds=legacy_bounds,
        )
        key = (design.continue_low.tobytes(), design.continue_high.tobytes())
        if key not in cache:
            value = float(design.operating_characteristics(null_probability).rejection_probability)
            if not np.isfinite(value):
                raise RuntimeError("Null rejection probability is nonfinite")
            cache[key] = value
        return SeqBinCalibrationPoint(design, cache[key])

    low, high = evaluate(float(bounds[0])), evaluate(float(bounds[1]))
    lower: SeqBinCalibrationPoint | None
    upper: SeqBinCalibrationPoint | None
    if low.significance > alpha:
        lower, upper = None, low
    elif high.significance <= alpha:
        lower, upper = high, None
    else:
        # Positive IEEE-754 float bit patterns have the same order as their
        # values. Integer bisection reaches adjacent cutoffs in at most 63 steps,
        # including subnormal tail_bounds; arithmetic midpoint searches may stall.
        left = int(np.float64(bounds[0]).view(np.uint64))
        right = int(np.float64(bounds[1]).view(np.uint64))
        while right - left > 1:
            middle = (left + right) // 2
            point = evaluate(float(np.uint64(middle).view(np.float64)))
            if point.significance <= alpha:
                left, low = middle, point
            else:
                right, high = middle, point
        lower, upper = low, high
    if lower is None:
        if selection == "conservative":
            raise ValueError("No design within tail_bounds has significance <= the requested level")
        assert upper is not None
        chosen = upper
    elif upper is None or selection == "conservative":
        chosen = lower
    else:
        chosen = lower if alpha - lower.significance <= upper.significance - alpha else upper
    return SeqBinCalibration(chosen, lower, upper, alpha, selection, len(cache))


@dataclass(frozen=True)
class SeqBinTailCalibration:
    low: SeqBinCalibration
    high: SeqBinCalibration
    design: SeqBinDesign
    null_properties: SeqBinProperties


def seqbin_calibrate_tails(
    max_subjects: int,
    significance: ArrayLike,
    *,
    prior: ArrayLike = (1, 1),
    null_probability: float = 0.2,
    looks: ArrayLike | None = None,
    selection: str = "conservative",
    tail_bounds: ArrayLike = (1e-6, 1 - 1e-6),
    legacy_bounds: bool = False,
) -> SeqBinTailCalibration:
    """Calibrate separate one-sided levels, combine and evaluate the two-sided design.

    significance is [low, high]. Opposite-side stopping can reduce each actual
    joint-design error below its separately calibrated value. null_properties
    reports the actual combined design. Nearest selection is honored on each
    side; the original dialog erroneously always used the conservative cutoffs.
    """
    levels = finite(significance, "significance")
    if levels.shape != (2,):
        raise ValueError("significance must contain [low, high] target levels")
    low, high = (
        seqbin_calibrate(
            max_subjects,
            float(level),
            prior=prior,
            null_probability=null_probability,
            alternative=alternative,
            looks=looks,
            selection=selection,
            tail_bounds=tail_bounds,
            legacy_bounds=legacy_bounds,
        )
        for level, alternative in zip(levels, ("less", "greater"), strict=True)
    )
    design = SeqBinDesign(
        max_subjects,
        prior=prior,
        null_probability=null_probability,
        alternative="two-sided",
        looks=looks,
        tail_probability=[low.chosen.design.tail_probability, high.chosen.design.tail_probability],
        legacy_bounds=legacy_bounds,
    )
    return SeqBinTailCalibration(
        low, high, design, design.operating_characteristics(null_probability)
    )
