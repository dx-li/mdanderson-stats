"""Three-point trinary Lp EffTox desirability contour."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import logsumexp

from ._validation import FloatArray, finite

_MAX_CONTOUR_CELLS = 200_000
_LOG_MAX = float(np.log(np.finfo(np.float64).max))
_LOG_TINY = float(np.log(np.nextafter(0.0, 1.0)))


def _frozen(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    shape = array.shape
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64).reshape(shape)


def _log_ratio_increment(
    difference: float, denominator: float, log_numerator: float, log_denominator: float
) -> float:
    if difference < 0.5 * denominator:
        return float(np.log1p(difference / denominator))
    return float(log_numerator - log_denominator)


def _log_value_ratio(numerator: float, denominator: float) -> float:
    difference = numerator - denominator
    if abs(difference) < 0.5 * denominator:
        return float(np.log1p(difference / denominator))
    return float(np.log(numerator) - np.log(denominator))


def _small_log_factor(log_x: float) -> float:
    """log((1-exp(-x))/x), where x is available as log(x)."""
    if log_x < -20.0:
        x = float(np.exp(log_x))
        x2 = x * x
        return -0.5 * x + x2 / 24.0 - x2 * x2 / 2880.0
    if log_x > 5.0:
        return -log_x
    x = float(np.exp(log_x))
    return float(np.log(-np.expm1(-x)) - log_x)


def _log_one_minus_exp_negative(log_x: float) -> float:
    if log_x < -20.0:
        return log_x + _small_log_factor(log_x)
    if log_x > 5.0:
        return 0.0
    x = float(np.exp(log_x))
    return float(np.log(-np.expm1(-x)))


@dataclass(frozen=True)
class EffToxTrinaryContour:
    """Trinary Lp contour through ``(e0,0)``, ``(em,tm)``, ``(eh,th)``.

    The high-efficacy point must lie on the hypotenuse, ``eh + th = 1``.
    The analytical toxicity intercept may exceed one; the probability domain
    for utility evaluation is the entire unit square.
    """

    efficacy_points: ArrayLike
    toxicity_points: ArrayLike
    shape: float = field(init=False)
    toxicity_intercept: float = field(init=False)
    _log_a: float = field(init=False, repr=False, compare=False)
    _log_toxicity_intercept: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_e, raw_t = np.asarray(self.efficacy_points), np.asarray(self.toxicity_points)
        if raw_e.shape != (3,) or raw_t.shape != (3,):
            raise ValueError("trinary contour requires three efficacy and toxicity points")
        efficacy = finite(raw_e, "efficacy_points")
        toxicity = finite(raw_t, "toxicity_points").copy()
        e0, em, eh = map(float, efficacy)
        t0, tm, th = map(float, toxicity)
        hypotenuse_toxicity = 1.0 - eh
        if not (
            0.0 < e0 < em < eh < 1.0
            and t0 == 0.0
            and 0.0 < tm < th < 1.0
            and abs(th - hypotenuse_toxicity)
            <= 32.0 * np.finfo(float).eps * max(th, hypotenuse_toxicity)
        ):
            raise ValueError("points must satisfy 0<e0<em<eh<1, t0=0<tm<th<1, and eh+th=1")

        # Canonicalize the hypotenuse coordinate after a relative, local-scale
        # check; an O(1) absolute tolerance would accept a materially different
        # high point when its toxicity coordinate is tiny.
        th = hypotenuse_toxicity
        toxicity[2] = th
        b = 1.0 - em
        c = th
        d = tm
        aa = _log_ratio_increment(em - e0, b, np.log1p(-e0), np.log1p(-em))
        cc = _log_ratio_increment(eh - e0, c, np.log1p(-e0), np.log(c))
        dd = _log_ratio_increment(th - d, d, np.log(th), np.log(d))
        if not (0.0 < aa < cc and dd > 0.0 and np.all(np.isfinite([aa, cc, dd]))):
            raise ValueError("elicited points do not define a unique positive trinary contour")
        log_a_over_c = _log_value_ratio(aa, cc)

        def equation(log_p: float) -> float:
            if log_p < -300.0:
                return log_a_over_c + dd * float(np.exp(log_p))
            if log_p > _LOG_MAX:
                return np.inf
            p = float(np.exp(log_p))
            log_ap = float(np.log(aa) + log_p)
            log_cp = float(np.log(cc) + log_p)
            if log_p > _LOG_MAX - np.log(dd):
                linear_term = np.inf
            else:
                linear_term = dd * p
            return (
                log_a_over_c + _small_log_factor(log_ap) - _small_log_factor(log_cp) + linear_term
            )

        lo, hi = -1.0, 1.0
        if equation(0.0) < 0.0:
            lo = 0.0
            hi = 1.0
            while equation(hi) < 0.0 and hi < 700.0:
                hi = min(2.0 * hi, 700.0)
            if equation(hi) < 0.0:
                raise ArithmeticError("could not bracket positive trinary contour shape")
        else:
            hi = 0.0
            lo = -1.0
            while equation(lo) > 0.0 and lo > -700.0:
                lo = max(2.0 * lo, -700.0)
            if equation(lo) > 0.0:
                raise ArithmeticError("positive trinary contour shape is below float range")
        try:
            log_shape = float(brentq(equation, lo, hi, xtol=2e-13, rtol=8 * np.finfo(float).eps))
        except (ValueError, RuntimeError) as exc:
            raise ArithmeticError("could not solve positive trinary contour shape") from exc
        if not _LOG_TINY <= log_shape <= _LOG_MAX:
            raise ArithmeticError("trinary contour shape is outside floating-point range")
        shape = float(np.exp(log_shape))
        log_tstar = float(np.log(c) - _log_one_minus_exp_negative(np.log(cc) + log_shape) / shape)
        if not _LOG_TINY <= log_tstar <= _LOG_MAX:
            raise ArithmeticError("trinary toxicity intercept is outside floating-point range")
        tstar = float(np.exp(log_tstar))
        if not np.isfinite(shape) or shape <= 0 or not np.isfinite(tstar) or tstar <= 0:
            raise ArithmeticError("trinary contour parameters are not representable")

        object.__setattr__(self, "efficacy_points", _frozen(efficacy))
        object.__setattr__(self, "toxicity_points", _frozen(toxicity))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "toxicity_intercept", tstar)
        object.__setattr__(self, "_log_a", float(np.log1p(-e0)))
        object.__setattr__(self, "_log_toxicity_intercept", log_tstar)

    @classmethod
    def from_points(cls, efficacy: ArrayLike, toxicity: ArrayLike) -> EffToxTrinaryContour:
        """Construct from the low, middle and hypotenuse target points."""
        return cls(efficacy, toxicity)

    def utility(self, efficacy: ArrayLike, toxicity: ArrayLike) -> FloatArray:
        """Evaluate trinary desirability ``1 - Lp_norm`` on the unit square."""
        raw_e, raw_t = np.asarray(efficacy), np.asarray(toxicity)
        try:
            output_shape = np.broadcast_shapes(raw_e.shape, raw_t.shape)
        except ValueError as exc:
            raise ValueError("efficacy and toxicity are not broadcast-compatible") from exc
        if max(raw_e.size, raw_t.size, prod(output_shape)) > _MAX_CONTOUR_CELLS:
            raise ValueError("trinary contour utility exceeds 200000 cells")
        e, t = finite(raw_e, "efficacy"), finite(raw_t, "toxicity")
        if np.any((e < 0.0) | (e > 1.0)) or np.any((t < 0.0) | (t > 1.0)):
            raise ValueError("efficacy and toxicity must lie in [0,1]")
        with np.errstate(divide="ignore", invalid="ignore"):
            log_e_term = self.shape * (np.log1p(-e) - self._log_a)
            log_t_term = self.shape * (np.log(t) - self._log_toxicity_intercept)
        log_norm = logsumexp(np.stack(np.broadcast_arrays(log_e_term, log_t_term)), axis=0)
        log_norm = log_norm / self.shape
        utility = -np.expm1(log_norm)
        if np.any(np.isnan(utility)) or np.any(np.isneginf(utility)):
            raise ArithmeticError("trinary desirability calculation is invalid")
        return _frozen(utility)
