"""Native BaCIS fixed-response-count equivalent sample size.

This adapts ``compESS`` from bacistool: it matches the posterior variance of
``Beta(y + 1, N - y + 1)`` to the sample variance of retained response-rate
draws, holding the observed response count ``y`` fixed. It does not implement
the paper's less-specific beta mean-and-variance matching description.
"""

from dataclasses import dataclass
from math import log, prod, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ._validation import FloatArray
from .bacis import _validate_data

_MAX_SAMPLE_CELLS = 500_000


def _freeze(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class BaCISEquivalentSampleSize:
    """Per-subgroup variance match and every admissible real root.

    ``candidate_roots[j]`` contains finite roots N >= observed responses for
    subgroup j. The selected root minimizes the native discrepancy
    ``abs(y / N - y / n_observed)``; exact ties use the smaller N. When y=N=0,
    the rate is defined by continuity as zero.
    """

    posterior_variance: FloatArray
    equivalent_sample_size: FloatArray
    candidate_roots: tuple[FloatArray, ...]
    candidate_rate_distance: tuple[FloatArray, ...]
    achieved_variance: FloatArray
    variance_residual: FloatArray
    relative_variance_residual: FloatArray


def _log_beta_variance(sample_size: float, responses: int) -> float:
    """Log Var(Beta(y+1, N-y+1)) for real N >= y, without large powers."""
    a = responses + 1.0
    b = sample_size - responses + 1.0
    return log(a) + log(b) - 2.0 * log(sample_size + 2.0) - log(sample_size + 3.0)


def _roots_for_variance(variance: float, responses: int) -> list[float]:
    """Find all roots on the rising and falling branches of the beta variance."""
    if variance <= 0.0:
        raise ValueError("posterior sample variance must be positive for a finite ESS")
    y = float(responses)
    a = y + 1.0
    # The unconstrained peak in b=N-y+1 follows from d log(var)/db=0.
    b_peak = (sqrt((a + 1.0) * (9.0 * a + 1.0)) - (a + 1.0)) / 4.0
    peak = max(y, b_peak + y - 1.0)
    log_target = log(variance)
    peak_log_variance = _log_beta_variance(peak, responses)
    tolerance = 64.0 * np.finfo(float).eps
    if log_target > peak_log_variance + tolerance:
        raise ValueError(
            "posterior variance has no beta-equivalent sample size for this response count"
        )
    if abs(log_target - peak_log_variance) <= tolerance:
        return [peak]

    roots: list[float] = []
    lower = y
    lower_value = _log_beta_variance(lower, responses) - log_target
    if abs(lower_value) <= tolerance:
        roots.append(lower)
    elif lower < peak and lower_value < 0.0:
        roots.append(
            float(
                brentq(
                    lambda n: _log_beta_variance(n, responses) - log_target,
                    lower,
                    peak,
                    xtol=1e-12,
                    rtol=1e-14,
                    maxiter=200,
                )
            )
        )

    # The falling branch is solved in log(N), avoiding loss of resolution when
    # very small posterior variances imply a large effective sample size. For
    # y=0 the peak is the allowed endpoint N=0, so a small root can lie in (0,1).
    if responses == 0:
        at_one = _log_beta_variance(1.0, responses) - log_target
        if at_one <= 0.0:
            roots.append(
                float(
                    brentq(
                        lambda n: _log_beta_variance(n, responses) - log_target,
                        0.0,
                        1.0,
                        xtol=1e-14,
                        rtol=1e-14,
                        maxiter=200,
                    )
                )
            )
            return sorted(roots)
        left_z = 0.0
    else:
        left_z = log(max(peak, y))
    left_n = float(np.exp(left_z))
    if not np.isfinite(left_n):
        raise ArithmeticError("variance root is outside representable sample-size range")
    left_value = _log_beta_variance(left_n, responses) - log_target
    if abs(left_value) <= tolerance:
        roots.append(left_n)
    elif left_value > 0.0:
        right_z = left_z + log(2.0)
        max_z = log(np.finfo(float).max)
        falling_root_found = False
        for _ in range(1100):
            if right_z >= max_z:
                break
            right_n = float(np.exp(right_z))
            right_value = _log_beta_variance(right_n, responses) - log_target
            if right_value <= 0.0:
                root_z = brentq(
                    lambda z: _log_beta_variance(float(np.exp(z)), responses) - log_target,
                    left_z,
                    right_z,
                    xtol=1e-12,
                    rtol=1e-14,
                    maxiter=200,
                )
                roots.append(float(np.exp(root_z)))
                falling_root_found = True
                break
            left_z = right_z
            right_z += log(2.0)
        else:
            raise ArithmeticError("variance root search exceeded its iteration bound")
        if not falling_root_found:
            raise ArithmeticError("variance root exceeds the representable sample-size range")
    roots = sorted(root for root in roots if np.isfinite(root) and root >= y)
    if not roots:
        raise ValueError("posterior variance has no admissible beta-equivalent sample size")
    # Remove a duplicate at the branch maximum / lower endpoint.
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1e-10 * max(1.0, abs(root)):
            unique.append(root)
    return unique


def bacis_equivalent_sample_size(
    probability_samples: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
) -> BaCISEquivalentSampleSize:
    """Compute the native BaCIS fixed-y posterior variance equivalent sample size.

    ``probability_samples`` has axes ``(chains, draws, subgroups)`` with at
    least two total samples. Variance is the unbiased sample variance pooled
    over chain and draw, matching R's ``var()``. For each subgroup, all valid
    roots of the native cubic are retained; the chosen root minimizes
    ``abs(y/N - y/n)``. Unlike the native
    helper, this rejects zero variance, inadmissible roots and unrepresentable
    answers instead of clamping a root to an arbitrary small positive value.
    """
    shape = np.shape(probability_samples)
    if (
        len(shape) != 3
        or shape[0] < 1
        or shape[1] < 1
        or shape[0] * shape[1] < 2
        or not 1 <= shape[2] <= 100
    ):
        raise ValueError(
            "probability_samples must have shape (chains, draws, 1..100 groups) "
            "with at least two total samples"
        )
    if prod(shape) > _MAX_SAMPLE_CELLS:
        raise ValueError("probability_samples exceeds the 500000-cell limit")
    y, n = _validate_data(successes, trials)
    if y.shape != (shape[2],):
        raise ValueError("sample subgroup count must match successes and trials")
    raw = np.asarray(probability_samples)
    if np.iscomplexobj(raw):
        raise ValueError("probability_samples must be real")
    samples = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(samples)) or np.any((samples < 0.0) | (samples > 1.0)):
        raise ValueError("probability_samples must be finite probabilities in [0,1]")
    variance = np.var(samples.reshape(-1, shape[2]), axis=0, ddof=1)
    selected = np.empty(shape[2], dtype=np.float64)
    residual = np.empty_like(selected)
    relative = np.empty_like(selected)
    achieved = np.empty_like(selected)
    roots_by_group: list[FloatArray] = []
    distances_by_group: list[FloatArray] = []
    for group, (response_count, patient_count, sample_variance) in enumerate(
        zip(y.astype(np.int64), n.astype(np.int64), variance, strict=True)
    ):
        roots = _roots_for_variance(float(sample_variance), int(response_count))
        observed_rate = int(response_count) / int(patient_count)
        distances = np.asarray(
            [
                abs((int(response_count) / root if root > 0 else 0.0) - observed_rate)
                for root in roots
            ],
            dtype=np.float64,
        )
        minimum = float(np.min(distances))
        tied = [i for i, distance in enumerate(distances) if distance == minimum]
        chosen_index = min(tied, key=lambda i: roots[i])
        estimate = roots[chosen_index]
        achieved_log_variance = _log_beta_variance(estimate, int(response_count))
        achieved_variance = float(np.exp(achieved_log_variance))
        selected[group] = estimate
        achieved[group] = achieved_variance
        residual[group] = abs(achieved_variance - sample_variance)
        relative[group] = abs(np.expm1(achieved_log_variance - log(float(sample_variance))))
        roots_by_group.append(_freeze(roots))
        distances_by_group.append(_freeze(distances))
    return BaCISEquivalentSampleSize(
        _freeze(variance),
        _freeze(selected),
        tuple(roots_by_group),
        tuple(distances_by_group),
        _freeze(achieved),
        _freeze(residual),
        _freeze(relative),
    )
