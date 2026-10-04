"""Exact finite-state cutoff calibration for single-arm binary outcomes."""

import numpy as np
from scipy.special import betainc, betaincc
from scipy.stats import betabinom

from ._validation import finite, scalar
from .success_calibration import (
    SuccessCalibration,
    SuccessOperatingCharacteristics,
    _direction,
    binary_success_oc,
)


def calibrate_binary_success_cutoff(
    n: int,
    target: float,
    *,
    cutoff_range: tuple[float, float] = (0.6, 0.999),
    margin: float,
    design_prior: tuple[float, float] = (1.0, 1.0),
    analysis_prior: tuple[float, float] = (1.0, 1.0),
    direction: str = "greater",
    null_rate: float | None = None,
) -> SuccessCalibration:
    """Find the smallest cutoff in a range whose single-arm PID meets target.

    The search evaluates the lower range endpoint and posterior probability
    breakpoints for the finite set of possible response counts. Success uses
    the strict rule ``posterior_probability > cutoff``. The smallest-feasible
    tie policy is an explicit Python convention, not native search parity.
    """
    sample_size = scalar(n, "n")
    if sample_size != np.floor(sample_size) or not 1 <= sample_size <= 100000:
        raise ValueError("n must be an integer in [1,100000]")
    sample_size = int(sample_size)

    target_value = scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must be in (0,1)")
    bounds = finite(cutoff_range, "cutoff_range")
    if bounds.shape != (2,) or np.any((bounds < 0) | (bounds > 1)):
        raise ValueError("cutoff_range must contain two values in [0,1]")
    lower, upper = map(float, bounds)
    if lower > upper:
        raise ValueError("cutoff_range lower bound must not exceed upper bound")

    margin_value = scalar(margin, "margin")
    null = margin_value if null_rate is None else scalar(null_rate, "null_rate")
    if not 0 <= margin_value <= 1 or not 0 <= null <= 1:
        raise ValueError("margin and null_rate must be in [0,1]")
    sign = _direction(direction)
    design = finite(design_prior, "design_prior")
    analysis = finite(analysis_prior, "analysis_prior")
    if (
        design.shape != (2,)
        or analysis.shape != (2,)
        or np.any(design <= 0)
        or np.any(analysis <= 0)
    ):
        raise ValueError("each prior must contain two positive beta shape parameters")

    counts = np.arange(sample_size + 1)
    posterior_tail = betaincc if sign == 1 else betainc
    posterior_probability = posterior_tail(
        analysis[0] + counts, analysis[1] + sample_size - counts, margin_value
    )
    design_effective = posterior_tail(
        design[0] + counts, design[1] + sample_size - counts, margin_value
    )
    design_ineffective = (betainc if sign == 1 else betaincc)(
        design[0] + counts, design[1] + sample_size - counts, margin_value
    )
    predictive_mass = betabinom.pmf(counts, sample_size, *design)
    if (
        not np.all(np.isfinite(posterior_probability))
        or not np.all(np.isfinite(design_effective))
        or not np.all(np.isfinite(design_ineffective))
        or not np.all(np.isfinite(predictive_mass))
        or np.any((posterior_probability < 0) | (posterior_probability > 1))
        or np.any((design_effective < 0) | (design_effective > 1))
        or np.any((design_ineffective < 0) | (design_ineffective > 1))
        or np.any(predictive_mass < 0)
        or abs(float(np.sum(predictive_mass)) - 1.0) > 1e-8
    ):
        raise ArithmeticError("binary posterior or predictive probabilities are not representable")
    if (
        sign == 1
        and (np.any(np.diff(posterior_probability) < 0) or np.any(np.diff(design_effective) < 0))
    ) or (
        sign == -1
        and (np.any(np.diff(posterior_probability) > 0) or np.any(np.diff(design_effective) > 0))
    ):
        raise ArithmeticError("binary posterior tails lost monotone response ordering")

    candidates = np.unique(
        np.concatenate(
            (
                np.array([lower]),
                posterior_probability[
                    (posterior_probability > lower) & (posterior_probability <= upper)
                ],
            )
        )
    )
    if margin_value not in (0, 1) and lower == 0 and np.any(posterior_probability == 0):
        raise ArithmeticError("a positive posterior success probability underflowed at cutoff zero")

    largest_posterior = float(np.max(posterior_probability))
    last_positive_index = int(np.searchsorted(candidates, largest_posterior, side="left")) - 1
    if last_positive_index < 0:
        raise ValueError(
            "no cutoff in cutoff_range achieves the PID target with positive success probability"
        )

    # The beta-binomial has a monotone likelihood ratio in the response count.
    # Both the analysis success probability and design-prior probability of the
    # favorable state have the same count ordering. Raising the cutoff therefore
    # removes the least favorable successful states first, so PID is
    # nonincreasing as long as success probability remains positive.
    def evaluate(index: int) -> SuccessOperatingCharacteristics:
        cutoff = float(candidates[index])
        result = binary_success_oc(
            sample_size,
            cutoff,
            margin=margin_value,
            design_prior=(float(design[0]), float(design[1])),
            analysis_prior=(float(analysis[0]), float(analysis[1])),
            direction=direction,
            null_rate=null,
        )
        if result.incorrect_decision_probability is None and np.any(posterior_probability > cutoff):
            raise ArithmeticError(
                "success probability is structurally positive but not representable"
            )
        return result

    lower_oc = evaluate(0)
    candidates_evaluated = 1
    lower_pid = lower_oc.incorrect_decision_probability
    if lower_pid is None:
        raise ValueError(
            "no cutoff in cutoff_range achieves the PID target with positive success probability"
        )
    if lower_pid <= target_value:
        return SuccessCalibration(lower, target_value, lower_oc, candidates_evaluated)

    lower_index, upper_index = 1, last_positive_index
    selected_index: int | None = None
    operating_characteristics = None
    while lower_index <= upper_index:
        middle = (lower_index + upper_index) // 2
        candidate_oc = evaluate(middle)
        candidates_evaluated += 1
        pid = candidate_oc.incorrect_decision_probability
        if pid is not None and pid <= target_value:
            selected_index = middle
            operating_characteristics = candidate_oc
            upper_index = middle - 1
        else:
            lower_index = middle + 1

    if selected_index is None or operating_characteristics is None:
        raise ValueError(
            "no cutoff in cutoff_range achieves the PID target with positive success probability"
        )
    selected = float(candidates[selected_index])

    return SuccessCalibration(
        selected,
        target_value,
        operating_characteristics,
        candidates_evaluated,
    )
