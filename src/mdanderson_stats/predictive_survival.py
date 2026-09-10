"""Final two-arm exponential/inverse-gamma comparisons for predictive analysis."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc, ndtri

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import _owned
from .beta_comparison import _cdf_from_logs


@dataclass(frozen=True)
class SurvivalPredictiveComparison:
    posterior_shape: FloatArray
    posterior_scale: FloatArray
    arm_a_probability: FloatArray
    arm_b_probability: FloatArray
    z_statistic: FloatArray
    decision: NDArray[np.str_]


def _survival_prior(prior: ArrayLike) -> FloatArray:
    value = finite(prior, "prior")
    if value.shape != (2, 2) or np.any(value <= 0):
        raise ValueError("prior must be positive [[shape_A,scale_A],[shape_B,scale_B]]")
    return value


def compare_predictive_survival(
    events: ArrayLike,
    exposure: ArrayLike,
    *,
    prior: ArrayLike,
    method: str = "bayesian",
    posterior_cutoff: float = 0.95,
    significance_level: float = 0.025,
) -> SurvivalPredictiveComparison:
    """Compare mean survival on A versus B; final axis indexes the two arms.

    Prior is inverse-gamma shape and scale on mean survival (not median).
    Frequentist significance_level is one-sided, as in the guide's survival equations.
    A zero-exposure arm makes the frequentist comparison not_evaluable.
    """
    d, e = np.broadcast_arrays(count(events, "events"), finite(exposure, "exposure"))
    if d.ndim == 0 or d.shape[-1] != 2 or np.any(e < 0) or np.any((d > 0) & (e == 0)):
        raise ValueError(
            "require an arm axis of length two, nonnegative exposure, positive exposure for events"
        )
    pair = _survival_prior(prior)
    shape, scale = pair[:, 0] + d, pair[:, 1] + e
    if np.any(~np.isfinite(shape)) or np.any(~np.isfinite(scale)):
        raise ArithmeticError("posterior shape or scale overflows")
    cutoff, alpha = (
        scalar(posterior_cutoff, "posterior_cutoff"),
        scalar(significance_level, "significance_level"),
    )
    if not 0.5 <= cutoff < 1 or not 0 < alpha < 0.5:
        raise ValueError(
            "require posterior_cutoff in [0.5,1), one-sided significance_level in (0,0.5)"
        )
    if method not in ("bayesian", "frequentist"):
        raise ValueError("method must be bayesian or frequentist")
    a, b = scale[..., 0], scale[..., 1]
    ratio = np.minimum(a, b) / np.maximum(a, b)
    small, large = ratio / (1 + ratio), 1 / (1 + ratio)
    q, qc = np.where(a <= b, small, large), np.where(a <= b, large, small)
    aa, bb = shape[..., 0], shape[..., 1]
    pa = np.where(q <= 0.5, betainc(aa, bb, q), betaincc(bb, aa, qc))
    pb = np.where(qc <= 0.5, betainc(bb, aa, qc), betaincc(aa, bb, q))
    # Recover small beta arguments in log space if the ratio itself underflows.
    log_ratio = np.log(b) - np.log(a)
    logq, logqc = -np.logaddexp(0, log_ratio), -np.logaddexp(0, -log_ratio)
    for index in np.ndindex(pa.shape):
        if q[index] == 0 or qc[index] == 0:
            pa[index] = _cdf_from_logs(
                float(aa[index]), float(bb[index]), float(logq[index]), float(logqc[index])
            )
            pb[index] = _cdf_from_logs(
                float(bb[index]), float(aa[index]), float(logqc[index]), float(logq[index])
            )
    if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pb)):
        raise ArithmeticError("inverse-gamma ordering probability failed")
    decision = np.full(d.shape[:-1], "inconclusive", dtype="U16")
    z = np.full(d.shape[:-1], np.nan)
    if method == "bayesian":
        decision[pa > cutoff] = "arm_a_superior"
        decision[pb > cutoff] = "arm_b_superior"
    else:
        available = np.all(e > 0, axis=-1)
        maximum = np.max(e, axis=-1, keepdims=True)
        scaled = np.divide(e, maximum, out=np.zeros_like(e), where=maximum > 0)
        ea, eb = scaled[..., 0], scaled[..., 1]
        da, db = d[..., 0], d[..., 1]
        numerator = da * eb - db * ea
        denominator = np.hypot(np.sqrt(da) * eb, np.sqrt(db) * ea)
        # Both event counts zero: no evidence of a hazard difference.
        z = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        z = np.where(available, z, np.nan)
        critical = -ndtri(alpha)
        decision[z < -critical] = "arm_a_superior"
        decision[z > critical] = "arm_b_superior"
        decision[~available] = "not_evaluable"
    decision.flags.writeable = False
    return SurvivalPredictiveComparison(
        _owned(shape), _owned(scale), _owned(pa), _owned(pb), _owned(z), decision
    )
