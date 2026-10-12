"""Explicit WFMM 3.1 empirical-Bayes compatibility, including signed stopping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, scalar
from .wfmm_empirical_bayes import _coefficient_labels
from .wfmm_model import _finite_real
from .wfmm_variance_init import _frozen, _frozen_int


@dataclass(frozen=True)
class WFMMNativeEmpiricalBayes:
    """Native group parameters and the last E-step's coefficient probabilities.

    ``inclusion_probability`` is the returned native gamma/alpha array, evaluated
    using the *preceding* group's parameters. It is not recomputed at returned
    pi/T. ``converged`` additionally requires absolute changes within 1e-4;
    native signed stopping alone does not establish convergence.
    """

    group_inclusion_probability: FloatArray
    group_slab_to_sampling_variance: FloatArray
    inclusion_probability: FloatArray
    partition_labels: np.ndarray
    iterations: int
    converged: bool
    legacy_stopping_satisfied: bool
    maximum_change: float
    maximum_signed_change: float
    profile: str


def wfmm_native_empirical_bayes(
    standardized_estimates: ArrayLike,
    *,
    profile: str,
    coefficient_partition: ArrayLike | None = None,
    initial_inclusion_probability: float = 0.5,
    initial_slab_to_sampling_variance: float = 1.0,
    min_inclusion: float = 1e-14,
    min_ratio: float = 10.0,
    vague_ratio: float = 1000.0,
    max_odds: float = 1e20,
    unsmoothed_partitions: int = 0,
    max_iterations: int = 1000,
) -> WFMMNativeEmpiricalBayes:
    """Replay native bounded EM with an explicit ``legacy_3_1`` profile.

    Input is P-by-K joint-GLS estimates divided by their conditional sampling
    standard deviations. Partition labels identify the native level groups; the
    first ``unsmoothed_partitions`` sorted groups receive pi=1-minp and T=bigT.
    T is a slab-to-sampling variance ratio, not a raw slab variance. All groups
    share the native signed stopping test at 1e-4. The mathematically converged
    unbounded EM alternative remains ``calibrate_wfmm_shrinkage``.
    """
    z = _finite_real(standardized_estimates, "standardized_estimates")
    if z.ndim != 2 or not 1 <= z.shape[0] <= 100 or not 1 <= z.shape[1] <= 512:
        raise ValueError("standardized_estimates must be P-by-K with P<=100 and K<=512")
    if np.max(np.abs(z)) > 1e100:
        raise ValueError("standardized estimates exceed bounded numerical range")
    if profile != "legacy_3_1":
        raise ValueError("explicit legacy_3_1 profile is required")
    labels = _coefficient_labels(coefficient_partition, z.shape[1])
    partitions, indices = np.unique(labels, return_inverse=True)
    settings = (
        (initial_inclusion_probability, "initial_inclusion_probability"),
        (initial_slab_to_sampling_variance, "initial_slab_to_sampling_variance"),
        (min_inclusion, "min_inclusion"),
        (min_ratio, "min_ratio"),
        (vague_ratio, "vague_ratio"),
        (max_odds, "max_odds"),
    )
    if any(isinstance(v, (bool, np.bool_)) for v, _ in settings):
        raise ValueError("EM settings must be real scalars")
    pi0, u0, minp, mint, bigt, cap = (scalar(v, name) for v, name in settings)
    if not (
        0 < pi0 < 1
        and 0 < u0 <= 1e100
        and np.finfo(float).eps <= minp < 0.5
        and 0 < mint <= 1e100
        and 0 < bigt <= 1e100
        and 0 < cap <= 1e100
    ):
        raise ValueError("invalid bounded native EM settings")
    for value, name, lower, upper in (
        (unsmoothed_partitions, "unsmoothed_partitions", 0, partitions.size),
        (max_iterations, "max_iterations", 1, 10000),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    if z.size * max_iterations * 8 > 250_000_000:
        raise ValueError("native EM exceeds bounded work budget")
    pi = np.full((z.shape[0], partitions.size), pi0)
    ratio = np.full_like(pi, u0)
    squared = z * z
    group_columns = [indices == j for j in range(partitions.size)]
    for iteration in range(1, max_iterations + 1):
        p, u = pi[:, indices], ratio[:, indices]
        # Native exponent ceiling 250 then odds ceiling maxO. Log arithmetic
        # avoids overflow of the intermediate product without changing its law.
        log_odds = np.log(p) - np.log1p(-p) - 0.5 * np.log1p(u)
        log_odds += np.minimum(0.5 * squared * (u / (1 + u)), 250.0)
        odds = np.exp(np.minimum(log_odds, np.log(cap)))
        gamma = odds / (1 + odds)
        new_pi = np.clip(
            np.column_stack([gamma[:, g].mean(axis=1) for g in group_columns]), minp, 1 - minp
        )
        new_pi[:, :unsmoothed_partitions] = 1 - minp
        weighted = np.column_stack([(gamma * squared)[:, g].mean(axis=1) for g in group_columns])
        new_ratio = np.maximum(weighted / new_pi - 1, mint)
        new_ratio[:, :unsmoothed_partitions] = bigt
        if not np.all(np.isfinite(new_ratio)):
            raise ArithmeticError("native EM ratio is not representable")
        signed = float(max(np.max(new_pi - pi), np.max(new_ratio - ratio)))
        change = float(max(np.max(np.abs(new_pi - pi)), np.max(np.abs(new_ratio - ratio))))
        pi, ratio = new_pi, new_ratio
        if signed <= 1e-4:
            break
    return WFMMNativeEmpiricalBayes(
        _frozen(pi),
        _frozen(ratio),
        _frozen(gamma),
        _frozen_int(partitions),
        iteration,
        change <= 1e-4,
        signed <= 1e-4,
        change,
        signed,
        profile,
    )
