"""Approximate matched-pairs binomial power from archived STPLAN 4.5.

This procedure is an older, inactive STPLAN menu option. Its normal
approximation uses the Miettinen matched-pair variance expression; it is not
the package's exact matched case-control procedure.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ._validation import FloatArray, scalar
from .stplan_continuous import _alpha, _inputs, _power


@dataclass(frozen=True)
class STPLANMatchedPairsInitialSize:
    """No-pilot size estimate and source-recommended pilot size."""

    sample_size: float
    achieved_power: float
    psi: float
    input_mode: Literal["estimated_group_proportions", "source_default"]
    recommended_preliminary_size: float


def _table_psi(
    difference: FloatArray,
    z11: FloatArray,
    z10: FloatArray,
    z01: FloatArray,
    z00: FloatArray,
) -> FloatArray:
    """Evaluate PSIMPD's pilot-table variance parameter stably."""
    cells = (z11, z10, z01, z00)
    if np.any(difference < 0) or np.any(difference >= 1):
        raise ValueError("difference must lie in [0, 1)")
    if any(np.any(cell < 0) for cell in cells):
        raise ValueError("pilot table counts must be nonnegative")

    scale = np.maximum.reduce(cells)
    if np.any(scale == 0):
        raise ValueError("pilot table total must be at least 1")
    scaled = tuple(cell / scale for cell in cells)
    scaled_total = scaled[0] + scaled[1] + scaled[2] + scaled[3]
    if np.any(np.minimum(scale, 1.0) * scaled_total < 1):
        raise ValueError("pilot table total must be at least 1")

    # Normalize before addition so large finite counts cannot overflow. The
    # algebraic radicand below is identical to PSIMPD's expression but avoids
    # cancellation when the discordant-cell proportions are small.
    u = scaled[1] / scaled_total
    v = scaled[2] / scaled_total
    d = difference
    r = (u + v + d * (u - v)) / 2
    root = np.hypot(r - d, np.sqrt(2) * np.sqrt(d) * np.sqrt(1 - d) * np.sqrt(v))
    psi = r + root
    if np.any(~np.isfinite(psi) | (psi <= 0) | (psi > 1 + 16 * np.finfo(float).eps)):
        raise ArithmeticError("matched-pair pilot variance parameter is not representable")
    return np.minimum(psi, 1.0)


def _q_factor(difference: FloatArray, psi: FloatArray) -> FloatArray:
    """Stable PMPD denominator factor for pilot and no-pilot inputs."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = difference / psi
        pilot_q = (1 - t) * (1 + t) + (t * t) * (1 - psi) / 4
        delta_critical = 2 * psi / np.sqrt(3 + psi)
        ratio = difference / delta_critical
        no_pilot_q = (1 - ratio) * (1 + ratio)
    return np.where(difference <= psi, pilot_q, no_pilot_q)


def _critical_alpha(alpha: float, sides: int) -> float:
    result = _alpha(alpha, sides)
    if result <= 0:
        raise ValueError("alpha/sides is too small to represent a positive normal tail")
    return result


def _power_from_psi(
    difference: FloatArray,
    sample_size: FloatArray,
    psi: FloatArray,
    alpha: float,
    sides: int,
) -> FloatArray:
    if np.any(sample_size < 1):
        raise ValueError("sample_size must be at least 1")
    if np.any(difference < 0) or np.any(difference >= 1):
        raise ValueError("difference must lie in [0, 1)")
    critical_alpha = _critical_alpha(alpha, sides)

    # D/psi from PMPD's denominator is sqrt(1 - t^2(3+psi)/4). The shared
    # factorization supports both pilot (delta<=psi) and no-pilot inputs.
    q = _q_factor(difference, psi)
    if np.any(~np.isfinite(q) | (q <= 0)):
        raise ArithmeticError("matched-pair normal-approximation denominator is not positive")
    with np.errstate(over="ignore", invalid="ignore"):
        effect = (difference / np.sqrt(psi)) * np.sqrt(sample_size)
        z = (ndtri(critical_alpha) + effect) / np.sqrt(q)
    return _power(ndtr(z))


def stplan_matched_pairs_power(
    difference: ArrayLike,
    sample_size: ArrayLike,
    z11: ArrayLike,
    z10: ArrayLike,
    z01: ArrayLike,
    z00: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Approximate power for STPLAN's archived matched-pairs procedure.

    ``z11, z10, z01, z00`` are pilot paired-table cell counts in the order
    used by the original Fortran routine. Counts may be noninteger weights,
    as accepted by that routine, but must be nonnegative and have total >= 1.
    For ``sides=2``, alpha is halved and power is reported only in the
    specified direction of truth, matching native STPLAN's convention.
    """
    d, n, a11, a10, a01, a00 = _inputs(
        (difference, "difference"),
        (sample_size, "sample_size"),
        (z11, "z11"),
        (z10, "z10"),
        (z01, "z01"),
        (z00, "z00"),
    )
    psi = _table_psi(d, a11, a10, a01, a00)
    return _power_from_psi(d, n, psi, alpha, sides)


def _initial_size_from_psi(
    difference: float,
    target_power: float,
    psi: float,
    alpha: float,
    sides: int,
) -> float:
    if not 0 < difference < 1:
        raise ValueError("difference must lie strictly between 0 and 1 for size planning")
    if not 0 < target_power < 1:
        raise ValueError("target_power must lie strictly between 0 and 1")
    critical_alpha = _critical_alpha(alpha, sides)
    q = float(_q_factor(np.asarray(difference), np.asarray(psi)))
    if not isfinite(q) or q <= 0:
        raise ArithmeticError("matched-pair normal-approximation denominator is not positive")
    root_branch = -ndtri(critical_alpha) + ndtri(target_power) * np.sqrt(q)
    if not isfinite(float(root_branch)) or root_branch <= 0:
        raise ValueError(
            "requested power has no positive sample-size solution on the source branch"
        )
    # This is QMPD's closed-form size equation, rearranged to avoid an
    # underflowing psi**2 or a premature difference*sqrt(psi) product.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        root_n = (np.sqrt(np.longdouble(psi)) / np.longdouble(difference)) * np.longdouble(
            root_branch
        )
        size = root_n * root_n
    result = float(size)
    if not isfinite(result) or result < 1:
        raise ArithmeticError("matched-pair initial sample size is not representable at or above 1")
    return result


def stplan_matched_pairs_initial_size(
    difference: float,
    target_power: float,
    *,
    theta1: float | None = None,
    theta2: float | None = None,
    alpha: float = 0.05,
    sides: int = 1,
) -> STPLANMatchedPairsInitialSize:
    """Use archived STPLAN no-pilot modes to estimate matched-pair size.

    Supply both ``theta1`` and ``theta2`` to use the source's group-proportion
    approximation ``psi=theta1+theta2-2*theta1*theta2``; STPLAN recommends a
    preliminary sample of ``n/4`` pairs in that mode. If both are omitted, the
    source heuristic uses theta1=.9 and theta2=.1 and recommends ``n/6``.
    The .9/.1 values are attributed to the archived program's “ultraconservative”
    default, but are not a guarantee over arbitrary within-pair dependence.
    The returned pilot size is fractional as in the source; it is a
    recommendation, not a total-size guarantee. This API is a Python wrapper
    around the inactive legacy equations, not a reactivation of the menu.
    """
    d = scalar(difference, "difference")
    target = scalar(target_power, "target_power")
    if theta1 is None and theta2 is None:
        first, second = 0.9, 0.1
        mode: Literal["estimated_group_proportions", "source_default"] = "source_default"
        pilot_divisor = 6.0
    elif theta1 is not None and theta2 is not None:
        first, second = scalar(theta1, "theta1"), scalar(theta2, "theta2")
        if not (0 < first < 1 and 0 < second < 1):
            raise ValueError("theta1 and theta2 must lie strictly between 0 and 1")
        mode = "estimated_group_proportions"
        pilot_divisor = 4.0
    else:
        raise ValueError("theta1 and theta2 must be supplied together")
    if not (0 < first < 1 and 0 < second < 1):
        raise ValueError("theta1 and theta2 must lie strictly between 0 and 1")
    psi = first * (1 - second) + second * (1 - first)
    sample_size = _initial_size_from_psi(d, target, psi, alpha, sides)
    achieved = float(
        _power_from_psi(np.asarray(d), np.asarray(sample_size), np.asarray(psi), alpha, sides)
    )
    if abs(achieved - target) > 1e-10:
        raise ArithmeticError("closed-form sample size did not recover the requested power")
    pilot = sample_size / pilot_divisor
    if not isfinite(pilot):
        raise ArithmeticError("recommended preliminary sample size is not representable")
    return STPLANMatchedPairsInitialSize(sample_size, achieved, psi, mode, pilot)
