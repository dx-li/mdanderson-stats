"""Proper partition-level spike/slab hyperpriors and conjugate conditionals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from ._cdflib import _freeze
from ._validation import FloatArray


def _positive(value, name):
    raw = np.asarray(value)
    if raw.ndim > 2 or raw.size == 0 or raw.size > 51_200 or raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be a bounded real scalar/vector/matrix")
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1e100):
        raise ValueError(name + " must be positive, finite and at most 1e100")
    return _freeze(values)


@dataclass(frozen=True)
class WFMMShrinkageHyperprior:
    """Independent Beta(a,b) pi and inverse-gamma(shape,scale) slab priors.

    Parameters broadcast to fixed effects × retained partitions, sorted by
    partition label. A vector addresses partitions. ``update_mask`` selects
    pairs to update; False conditions on the supplied initial pi/tau instead.
    Shapes/scales are explicit proper prior parameters, not native conversion
    outputs or empirical-Bayes re-estimates. Infinite prior moments are possible
    for small inverse-gamma shapes, despite a proper probability distribution.
    """

    probability_shape_a: ArrayLike
    probability_shape_b: ArrayLike
    slab_shape: ArrayLike
    slab_scale: ArrayLike
    update_mask: ArrayLike = True

    def __post_init__(self):
        for name in ("probability_shape_a", "probability_shape_b", "slab_shape", "slab_scale"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        raw = np.asarray(self.update_mask)
        if raw.ndim > 2 or raw.size == 0 or raw.size > 51_200 or raw.dtype.kind != "b":
            raise ValueError("update_mask must be a bounded boolean scalar/vector/matrix")
        mask = np.array(raw, dtype=bool, copy=True)
        mask.flags.writeable = False
        object.__setattr__(self, "update_mask", mask)


@dataclass(frozen=True)
class WFMMShrinkageConditional:
    partition_labels: NDArray[np.int64]
    probability_shape_a: FloatArray
    probability_shape_b: FloatArray
    slab_shape: FloatArray
    slab_scale: FloatArray


def _hyperprior_arrays(prior, shape):
    if not isinstance(prior, WFMMShrinkageHyperprior):
        raise ValueError("shrinkage_hyperprior must be a WFMMShrinkageHyperprior")
    try:
        return tuple(
            np.broadcast_to(np.asarray(getattr(prior, name)), shape)
            for name in (
                "probability_shape_a",
                "probability_shape_b",
                "slab_shape",
                "slab_scale",
                "update_mask",
            )
        )
    except ValueError as exc:
        raise ValueError(
            "hyperprior arrays must broadcast to fixed effects by retained partitions"
        ) from exc


def _posterior_parameters(beta, gamma, group_index, arrays):
    p = beta.shape[0]
    a, b, shape, scale, _ = arrays
    counts = np.bincount(group_index, minlength=a.shape[1])
    included = np.zeros_like(a)
    squares = np.zeros_like(a)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(p):
            included[i] = np.bincount(group_index, weights=gamma[i], minlength=a.shape[1])
            squares[i] = np.bincount(
                group_index, weights=np.where(gamma[i], beta[i] ** 2, 0), minlength=a.shape[1]
            )
        result = (a + included, b + (counts - included), shape + included / 2, scale + squares / 2)
    if any(not np.all(np.isfinite(x)) or np.any(x <= 0) for x in result):
        raise ArithmeticError("shrinkage conditional parameters are not representable")
    return result


def wfmm_shrinkage_conditional(
    coefficients: ArrayLike,
    inclusion_indicators: ArrayLike,
    *,
    hyperprior: WFMMShrinkageHyperprior,
    coefficient_partition: ArrayLike | None = None,
) -> WFMMShrinkageConditional:
    """Return exact partition-wise conditional beta/inverse-gamma parameters.

    An included coefficient equal to zero still counts toward the slab shape.
    Excluded coefficients must be exactly zero. Update masks do not change the
    algebraic conditional; they select which parameters the sampler updates.
    """
    raw = np.asarray(coefficients)
    if (
        raw.ndim != 2
        or not 1 <= raw.shape[0] <= 100
        or not 1 <= raw.shape[1] <= 512
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError("coefficients must be a real 1..100 by 1..512 matrix")
    beta = np.asarray(raw, dtype=float)
    gamma = np.asarray(inclusion_indicators)
    if gamma.shape != beta.shape or gamma.dtype.kind != "b":
        raise ValueError("inclusion_indicators must be a matching boolean matrix")
    if not np.all(np.isfinite(beta)) or np.any(beta[~gamma] != 0):
        raise ValueError("coefficients must be finite with exact zeros at excluded indicators")
    labels = (
        np.zeros(beta.shape[1], dtype=np.int64)
        if coefficient_partition is None
        else np.asarray(coefficient_partition)
    )
    if (
        labels.shape != (beta.shape[1],)
        or labels.dtype.kind not in "iuf"
        or not np.all(np.isfinite(labels))
        or np.any(labels < 0)
        or np.any(labels > 2**31 - 1)
        or np.any(labels != np.floor(labels))
    ):
        raise ValueError("partitions must be bounded nonnegative integer labels")
    unique, groups = np.unique(labels.astype(np.int64), return_inverse=True)
    unique.flags.writeable = False
    arrays = _hyperprior_arrays(hyperprior, (beta.shape[0], len(unique)))
    parameters = _posterior_parameters(beta, gamma, groups, arrays)
    return WFMMShrinkageConditional(unique, *(_freeze(x) for x in parameters))


def _draw_hyperparameters(beta, gamma, groups, arrays, probability, slab, rng):
    mask = arrays[4]
    if not np.any(mask):
        return probability, slab
    a, b, shape, scale = _posterior_parameters(beta, gamma, groups, arrays)
    pi = rng.beta(a[mask], b[mask])
    precision_unit = rng.gamma(shape[mask])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tau = scale[mask] / precision_unit
    if np.any((pi <= 0) | (pi >= 1)) or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        raise ArithmeticError("hyperparameter draw is outside representable open support")
    probability[mask], slab[mask] = pi, tau
    return probability, slab


def wfmm_mode_shrinkage_hyperprior(
    probability_mode: ArrayLike,
    slab_mode: ArrayLike,
    *,
    probability_variance: ArrayLike = 0.06,
    slab_variance: ArrayLike = 1000.0,
    update_mask: ArrayLike = True,
) -> WFMMShrinkageHyperprior:
    """Calibrate proper priors to explicit modes and finite variances.

    This independently defined mode-centering policy is NOT the diagnosed
    native conversion. Beta variances must be in [1e-8,1/12); slab modes in
    [1e-6,1e6] and variances in [1e-12,1e12]. Extreme ratios can exceed the
    bounded root solver. Inverse-gamma shape is strictly above two.
    """
    raw = [
        np.asarray(x) for x in (probability_mode, slab_mode, probability_variance, slab_variance)
    ]
    if any(x.ndim > 2 or x.size == 0 or x.size > 51_200 or x.dtype.kind not in "iuf" for x in raw):
        raise ValueError("modes and variances must be bounded real arrays")
    try:
        pm, tm, pv, tv = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in raw))
    except ValueError as exc:
        raise ValueError("mode/variance shapes must broadcast together") from exc
    if (
        pm.size > 51_200
        or any(not np.all(np.isfinite(x)) for x in (pm, tm, pv, tv))
        or np.any((pm < 0) | (pm > 1))
        or np.any((pv < 1e-8) | (pv >= 1 / 12))
        or np.any((tm < 1e-6) | (tm > 1e6))
        or np.any((tv < 1e-12) | (tv > 1e12))
    ):
        raise ValueError("modes/variances exceed bounded calibration domain")
    aa, bb, shapes, scales = (np.empty(pm.shape) for _ in range(4))
    for index in np.ndindex(pm.shape):
        m, v, t, w = float(pm[index]), float(pv[index]), float(tm[index]), float(tv[index])
        try:
            c = brentq(
                lambda c: (1 + m * c) * (1 + (1 - m) * c) / (2 + c) ** 2 / (3 + c) - v,
                0,
                1e12,
                xtol=5e-15,
                rtol=1e-15,
            )
            a = brentq(
                lambda a: t * t * (a + 1) ** 2 / (a - 1) ** 2 / (a - 2) - w,
                np.nextafter(2.0, 3.0),
                1e12,
                xtol=5e-15,
                rtol=1e-15,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError("mode/variance combination exceeds bounded root solver") from exc
        b = t * (a + 1)
        actual = b * b / (a - 1) ** 2 / (a - 2)
        if not np.isclose(actual, w, rtol=1e-7, atol=0):
            raise ArithmeticError("requested slab variance is not accurately representable")
        aa[index], bb[index], shapes[index], scales[index] = 1 + m * c, 1 + (1 - m) * c, a, b
    return WFMMShrinkageHyperprior(aa, bb, shapes, scales, update_mask)
