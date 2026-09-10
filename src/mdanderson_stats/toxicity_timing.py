"""Calibrated, window-censored toxicity event-time quantiles."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .boin import _owned


def _hazard_factor(p: FloatArray) -> FloatArray:
    """-log(1-p)/p, with its continuous value at zero."""
    return np.divide(-np.log1p(-p), p, out=np.ones_like(p), where=p != 0)


def toxicity_time_quantile(
    quantile: ArrayLike,
    toxicity_probability: ArrayLike,
    window: float,
    *,
    distribution: str = "weibull",
    late_probability: ArrayLike | None = None,
) -> FloatArray:
    """Broadcast unconditional event quantiles, censored to +inf beyond window.

    For Weibull/log-logistic, calibrate F(window)=p and
    F(window/2)=(1-late_probability)*p. The latter probability defaults to .5.
    These continuous families require p<1 and 0<late_probability<1; p=0
    denotes no toxicity. Uniform timing is conditional on toxicity and does
    not take a late_probability parameter. Quantiles must lie in [0,1].
    """
    u, p = np.broadcast_arrays(
        finite(quantile, "quantile"), finite(toxicity_probability, "toxicity_probability")
    )
    duration = scalar(window, "window")
    if duration <= 0 or np.any((u < 0) | (u > 1) | (p < 0) | (p > 1)):
        raise ValueError("require positive window and quantiles/probabilities in [0,1]")
    if distribution not in ("uniform", "weibull", "log-logistic"):
        raise ValueError("distribution must be uniform, weibull or log-logistic")
    if distribution == "uniform":
        if late_probability is not None:
            raise ValueError("uniform timing does not accept late_probability")
        fraction = np.divide(u, p, out=np.zeros_like(u), where=(p > 0) & (u <= p))
    else:
        a = finite(0.5 if late_probability is None else late_probability, "late_probability")
        u, p, a = np.broadcast_arrays(u, p, a)
        if np.any(p == 1) or np.any((a <= 0) | (a >= 1)):
            raise ValueError("calibrated timing requires toxicity probability <1 and 0<late<1")
        # Only event quantiles need evaluation. Avoid singular log expressions
        # at zero or outside the assessment window.
        event = (p > 0) & (u > 0) & (u <= p)
        pe, ue, ae = p[event], u[event], a[event]
        z = pe * ae / (1 - pe)
        log_factor = np.divide(np.log1p(z), z, out=np.ones_like(z), where=z != 0)
        if distribution == "weibull":
            # H(p)/H((1-a)p)=1+[H(p)-H((1-a)p)]/H((1-a)p).
            # Scaled factors retain small p and small late probabilities.
            ratio = ae / ((1 - pe) * (1 - ae) * _hazard_factor(pe * (1 - ae)))
            shape = np.log1p(ratio * log_factor) / np.log(2)
            numerator = np.log(ue / pe) + np.log(_hazard_factor(ue) / _hazard_factor(pe))
        else:
            shape = (-np.log1p(-ae) + np.log1p(z)) / np.log(2)
            numerator = np.log(ue / pe) + np.log1p(-pe) - np.log1p(-ue)
        if np.any(~np.isfinite(shape) | (shape <= 0)):
            raise ArithmeticError("timing shape is not representable")
        fraction = np.zeros_like(u)
        with np.errstate(under="ignore", over="ignore"):
            fraction[event] = np.exp(np.minimum(numerator / shape, 0))
        fraction = np.where(u == p, 1, fraction)
    return _owned(np.where((p > 0) & (u <= p), duration * fraction, np.inf))
