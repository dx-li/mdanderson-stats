"""Original inverse-quadratic EffTox desirability contour.

Thall and Cook (2004) define the target curve as ``t = a + b/e + c/e**2``
and score a point by the radial distance ratio from the ideal point ``(1,0)``.
This exact-interpolation implementation is an explicit extension: the 2006
report says the former native fit minimized an error criterion, but does not
provide its objective or constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite

_MAX_CONTOUR_CELLS = 200_000


def _frozen(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    shape = array.shape
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class EffToxLegacyContour:
    """Monotone inverse-quadratic EffTox target with exact three-point fit.

    ``efficacy_points`` and ``toxicity_points`` describe the elicited target
    points in increasing efficacy order. The first must be ``(e0, 0)`` and
    the last must have efficacy one. Exact interpolation and monotonicity on
    ``[e0, 1]`` are required; a nonmonotone or ill-conditioned fit is rejected.
    """

    efficacy_points: ArrayLike
    toxicity_points: ArrayLike
    a: float = field(init=False)
    b: float = field(init=False)
    c: float = field(init=False)
    _scaled_coefficients: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_e, raw_t = np.asarray(self.efficacy_points), np.asarray(self.toxicity_points)
        if raw_e.shape != (3,) or raw_t.shape != (3,):
            raise ValueError("legacy contour requires three efficacy and three toxicity points")
        e = finite(raw_e, "efficacy_points")
        t = finite(raw_t, "toxicity_points")
        if not (0.0 < e[0] < e[1] < e[2] == 1.0 and t[0] == 0.0 and 0.0 < t[1] < t[2] < 1.0):
            raise ValueError("points must satisfy 0<e0<e1<1, e2=1 and 0=t0<t1<t2<1")
        # Fit a + b/e + c/e^2 using r=e0/e, which keeps the interpolation
        # matrix bounded even when the lowest elicited efficacy is small.
        r = e[0] / e
        design = np.column_stack((np.ones(3), r, r * r))
        condition = float(np.linalg.cond(design))
        if not np.isfinite(condition) or condition > 1e12:
            raise ValueError("legacy contour elicitation points are numerically ill-conditioned")
        scaled = np.linalg.solve(design, t)
        if not np.isfinite(scaled).all():
            raise ValueError("legacy contour coefficients are nonfinite")
        # f'(e) has the sign of -(B + 2*C*r), for r=e0/e in [e0,1].
        derivative_numerator = np.asarray(
            [scaled[1] + 2.0 * scaled[2] * e[0], scaled[1] + 2.0 * scaled[2]]
        )
        derivative_scale = max(float(np.max(np.abs(scaled[1:]))), float(np.finfo(float).tiny))
        if float(np.max(derivative_numerator)) > 128 * np.finfo(float).eps * derivative_scale:
            raise ValueError("inverse-quadratic target must be nondecreasing on [e0,1]")
        coefficients = np.asarray(
            [
                scaled[0],
                scaled[1] * e[0],
                scaled[2] * e[0] * e[0],
            ],
            dtype=np.float64,
        )
        reconstructed = design @ scaled
        if not np.allclose(reconstructed, t, rtol=2e-13, atol=2e-14):
            raise ArithmeticError("inverse-quadratic interpolation did not reproduce its points")
        object.__setattr__(self, "efficacy_points", _frozen(e))
        object.__setattr__(self, "toxicity_points", _frozen(t))
        object.__setattr__(self, "a", float(coefficients[0]))
        object.__setattr__(self, "b", float(coefficients[1]))
        object.__setattr__(self, "c", float(coefficients[2]))
        object.__setattr__(self, "_scaled_coefficients", _frozen(scaled))

    @classmethod
    def from_points(cls, efficacy: ArrayLike, toxicity: ArrayLike) -> EffToxLegacyContour:
        """Construct from three increasing-efficacy ``(E,T)`` target points."""
        return cls(efficacy, toxicity)

    @property
    def coefficients(self) -> FloatArray:
        """Return the coefficients ``(a,b,c)`` of ``a + b/E + c/E**2``."""
        return _frozen([self.a, self.b, self.c])

    def target_toxicity(self, efficacy: ArrayLike) -> FloatArray:
        """Evaluate the elicited curve on its defined efficacy interval [e0,1]."""
        raw = np.asarray(efficacy)
        if raw.size > _MAX_CONTOUR_CELLS:
            raise ValueError("legacy contour evaluation exceeds 200000 cells")
        e = finite(raw, "efficacy")
        points = cast(FloatArray, self.efficacy_points)
        if np.any((e < points[0]) | (e > 1.0)):
            raise ValueError("curve efficacy must lie on the elicited domain [e0,1]")
        r = points[0] / e
        a, _, c = self._scaled_coefficients
        # Since the elicited lower endpoint has f(e0)=0, this factored form
        # preserves that endpoint exactly and avoids cancellation near e0.
        return _frozen((1.0 - r) * (a - c * r))

    def utility(self, efficacy: ArrayLike, toxicity: ArrayLike) -> FloatArray:
        """Return radial desirability ``distance(p,I)/distance(q,I) - 1``.

        ``p`` is the intersection of the ray from the ideal point (1,0)
        through ``q=(efficacy,toxicity)`` with the inverse-quadratic target.
        The ideal point has positive-infinite desirability. Inputs broadcast
        to at most 200000 output cells.
        """
        raw_e, raw_t = np.asarray(efficacy), np.asarray(toxicity)
        try:
            shape = np.broadcast_shapes(raw_e.shape, raw_t.shape)
        except ValueError as exc:
            raise ValueError("efficacy and toxicity are not broadcast-compatible") from exc
        if max(raw_e.size, raw_t.size, prod(shape)) > _MAX_CONTOUR_CELLS:
            raise ValueError("legacy contour broadcast exceeds 200000 cells")
        e, t = finite(raw_e, "efficacy"), finite(raw_t, "toxicity")
        if np.any((e < 0.0) | (e > 1.0)) or np.any((t < 0.0) | (t > 1.0)):
            raise ValueError("efficacy and toxicity must lie in [0,1]")
        e, t = np.broadcast_arrays(e, t)
        result = np.empty(shape, dtype=np.float64)
        flat_e, flat_t, flat_result = e.reshape(-1), t.reshape(-1), result.reshape(-1)
        efficacy_points = cast(FloatArray, self.efficacy_points)
        toxicity_points = cast(FloatArray, self.toxicity_points)
        e0 = float(efficacy_points[0])
        t1 = float(toxicity_points[2])
        ideal = (flat_e == 1.0) & (flat_t == 0.0)
        flat_result[ideal] = np.inf

        at_efficacy_boundary = (flat_e == 1.0) & ~ideal
        if np.any(at_efficacy_boundary):
            radial = t1 / flat_t[at_efficacy_boundary]
            if not np.isfinite(radial).all():
                raise ArithmeticError("finite nonideal desirability exceeds floating-point range")
            flat_result[at_efficacy_boundary] = radial - 1.0

        at_zero_toxicity = (flat_t == 0.0) & (flat_e < 1.0)
        if np.any(at_zero_toxicity):
            radial = (1.0 - e0) / (1.0 - flat_e[at_zero_toxicity])
            if not np.isfinite(radial).all():
                raise ArithmeticError("finite nonideal desirability exceeds floating-point range")
            flat_result[at_zero_toxicity] = radial - 1.0

        solve = (flat_e < 1.0) & (flat_t > 0.0)
        if np.any(solve):
            eff = flat_e[solve]
            tox = flat_t[solve]
            efficacy_limit = (1.0 - e0) / (1.0 - eff)
            toxicity_limit = t1 / tox
            efficacy_active = efficacy_limit <= toxicity_limit
            upper = np.minimum(efficacy_limit, toxicity_limit)
            if not np.isfinite(upper).all() or np.any(upper <= 0.0):
                raise ArithmeticError("radial intersection is outside floating-point range")
            a, _, c = self._scaled_coefficients

            def equation(radial: np.ndarray) -> np.ndarray:
                p_e = 1.0 - radial * (1.0 - eff)
                ratio = e0 / p_e
                target = (1.0 - ratio) * (a - c * ratio)
                return radial * tox - target

            # Evaluate the geometric endpoint from its active boundary. This
            # avoids recomputing p_e=e0 after rounding it slightly upward,
            # which can make a tiny positive-toxicity bracket appear negative.
            endpoint_value = np.empty_like(upper)
            endpoint_value[efficacy_active] = upper[efficacy_active] * tox[efficacy_active]
            toxicity_indices = ~efficacy_active
            if np.any(toxicity_indices):
                p_e = 1.0 - upper[toxicity_indices] * (1.0 - eff[toxicity_indices])
                ratio = np.clip(e0 / p_e, e0, 1.0)
                # f(e0)-f(r) factors exactly for f(r)=(1-r)(a-c*r).
                endpoint_value[toxicity_indices] = (ratio - e0) * (a + c * (1.0 - ratio - e0))
            if np.any(endpoint_value < 0.0):
                raise ArithmeticError("could not bracket the legacy contour intersection")
            lower_fraction = np.zeros_like(upper)
            upper_fraction = np.ones_like(upper)
            for _ in range(80):
                midpoint = lower_fraction + 0.5 * (upper_fraction - lower_fraction)
                radial_midpoint = upper * midpoint
                midpoint_value = equation(radial_midpoint)
                # The endpoints are defined geometrically: at fraction zero
                # the point is ideal, and at one an efficacy/toxicity limit
                # is active. Use their exact endpoint equations when floating
                # point rounding stalls the normalized interval.
                midpoint_value = np.where(midpoint == 0.0, -t1, midpoint_value)
                midpoint_value = np.where(midpoint == 1.0, endpoint_value, midpoint_value)
                below = midpoint_value < 0.0
                lower_fraction = np.where(below, midpoint, lower_fraction)
                upper_fraction = np.where(below, upper_fraction, midpoint)
            radial = upper * (lower_fraction + 0.5 * (upper_fraction - lower_fraction))
            if not np.isfinite(radial).all():
                raise ArithmeticError("finite nonideal desirability exceeds floating-point range")
            flat_result[solve] = radial - 1.0
        if np.any(np.isnan(result)) or np.any(np.isneginf(result)):
            raise ArithmeticError("legacy contour desirability is invalid")
        return _frozen(result)
