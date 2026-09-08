"""First qualifying integer TDTASP designs, retaining discrete power oscillations."""

from dataclasses import dataclass
from operator import index

import numpy as np

from ._validation import scalar
from .tdtasp_ascertainment import TDTASPAscertainment
from .tdtasp_power import (
    TDTASPFixedPower,
    TDTASPPower,
    _family_count_distribution,
    _rounded_observations,
    tdtasp_fixed_power,
    tdtasp_power,
)


def _bounds(minimum: int, maximum: int, batch_size: int) -> tuple[int, int, int]:
    values = []
    for value in (minimum, maximum, batch_size):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("search bounds and batch_size must be positive integers")
        try:
            values.append(index(value))
        except TypeError as exc:
            raise ValueError("search bounds and batch_size must be positive integers") from exc
    lo, hi, batch = values
    if not 1 <= lo <= hi < 2**53 or batch < 1:
        raise ValueError("require 1 <= minimum <= maximum < 2**53 and batch_size >= 1")
    return lo, hi, batch


def _target(value: float) -> float:
    target = scalar(value, "target_power")
    if not 0 < target < 1:
        raise ValueError("target_power must lie in (0, 1)")
    return target


def tdtasp_fixed_sample_size(
    answer_probability: float,
    target_power: float = 0.8,
    alpha: float = 0.05,
    *,
    sides: int = 1,
    legacy_two_sided: bool = False,
    min_observations: int = 1,
    max_observations: int = 100_000,
    batch_size: int = 512,
) -> TDTASPFixedPower:
    """First fixed-observation design meeting target power within inclusive bounds.

    Ordered vectorized batches retain downward jumps in discrete power.
    No qualifying design raises ValueError, including unattainable null-power
    requests. The returned fixed-power result has scalar observation count.
    """
    lo, hi, batch = _bounds(min_observations, max_observations, batch_size)
    target = _target(target_power)
    checked = tdtasp_fixed_power(
        0, answer_probability, alpha, sides=sides, legacy_two_sided=legacy_two_sided
    )
    effective_level = checked.alpha / (checked.sides if checked.legacy_two_sided else 1)
    if checked.answer_probability == 0.5 and target > effective_level:
        raise ValueError("no qualifying design: null power cannot exceed actual significance")
    for start in range(lo, hi + 1, batch):
        counts = np.arange(start, min(start + batch, hi + 1), dtype=float)
        candidates = tdtasp_fixed_power(
            counts, answer_probability, alpha, sides=sides, legacy_two_sided=legacy_two_sided
        )
        passing = np.flatnonzero(candidates.power >= target)
        if passing.size:
            return tdtasp_fixed_power(
                int(counts[passing[0]]),
                answer_probability,
                alpha,
                sides=sides,
                legacy_two_sided=legacy_two_sided,
            )
    raise ValueError("no qualifying design within the requested observation bounds")


@dataclass(frozen=True)
class TDTASPSampleSize:
    """First passing screened-family design and search diagnostics.

    All smaller candidates in range are ruled out by the monotone upper bound
    or evaluated directly. search_start is the first directly scanned count;
    evaluations counts mixture evaluations, including upper-bound evaluations.
    """

    design: TDTASPPower
    target_power: float
    min_families: int
    max_families: int
    search_start: int
    evaluations: int


def tdtasp_sample_size(
    ascertainment: TDTASPAscertainment,
    target_power: float = 0.8,
    alpha: float = 0.05,
    *,
    sides: int = 1,
    legacy_scale: bool = False,
    legacy_two_sided: bool = False,
    min_families: int = 1,
    max_families: int = 100_000,
    max_terms: int = 100_001,
) -> TDTASPSampleSize:
    """Find the first qualifying integer screened-family count within the bounds.

    Cache the conditional power curve f(k). Its prefix maximum g(k) is a
    monotone upper bound. Because Binomial(n,q) increases stochastically in n,
    E[g(K)] is monotone and bounds E[f(K)]. Bisection locates a conservative
    starting count, followed by an ordered scan of actual power. A 1e-12 upward
    numerical cushion on the bound avoids excluding borderline designs.

    Conditional curves grow geometrically, constrained by max_terms before
    allocation. This shares the forward power model and compatibility options;
    it does not assume actual power is monotone or claim all larger designs pass.
    """
    lower, maximum, _ = _bounds(min_families, max_families, 1)
    target = _target(target_power)
    probe = tdtasp_power(
        ascertainment,
        0,
        alpha,
        sides=sides,
        legacy_scale=legacy_scale,
        legacy_two_sided=legacy_two_sided,
        max_terms=max_terms,
    )
    q, p, scale = (
        ascertainment.selection_probability,
        ascertainment.answer_probability,
        probe.contribution_per_family,
    )
    effective_level = probe.conditional.alpha / (sides if legacy_two_sided else 1)
    if p == 0.5 and target > effective_level:
        raise ValueError("no qualifying design: null power cannot exceed actual significance")
    curve = np.empty(0)
    envelope = np.empty(0)
    evaluations = 0

    def ensure(n: int) -> None:
        nonlocal curve, envelope
        if n < len(curve):
            return
        end = min(maximum, max_terms - 1, max(n, 2 * len(curve) - 1))
        if n + 1 > max_terms:
            raise ValueError(
                f"conditional curve requires {n + 1} terms, exceeding max_terms={max_terms}"
            )
        observations = _rounded_observations(np.arange(len(curve), end + 1, dtype=float), scale)
        extra = tdtasp_fixed_power(
            observations, p, alpha, sides=sides, legacy_two_sided=legacy_two_sided
        ).power
        curve = np.concatenate((curve, extra))
        envelope = np.maximum.accumulate(curve)

    def evaluate(n: int, upper: bool) -> float:
        nonlocal evaluations
        ensure(n)
        eligible, weights = _family_count_distribution(n, q, max_terms)
        values = envelope if upper else curve
        result = float(weights @ values[eligible.astype(np.int64)])
        evaluations += 1
        return min(1.0, result + 1e-12) if upper else result

    lo, hi = lower - 1, lower
    while evaluate(hi, True) < target:
        if hi == maximum:
            raise ValueError("no qualifying design within the requested family bounds")
        lo, hi = hi, min(maximum, max(hi + 1, 2 * hi))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if evaluate(mid, True) < target:
            lo = mid
        else:
            hi = mid
    start = max(lower, hi - 1)
    for n in range(start, maximum + 1):
        if evaluate(n, False) >= target:
            design = tdtasp_power(
                ascertainment,
                n,
                alpha,
                sides=sides,
                legacy_scale=legacy_scale,
                legacy_two_sided=legacy_two_sided,
                max_terms=max_terms,
            )
            if design.power < target:
                raise ArithmeticError("sample-size candidate changed during forward verification")
            return TDTASPSampleSize(design, target, lower, maximum, start, evaluations)
    raise ValueError("no qualifying design within the requested family bounds")
