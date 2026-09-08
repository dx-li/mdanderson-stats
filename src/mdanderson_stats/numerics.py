"""Numerical functionality of CUMNOR and INVMF using compiled SciPy kernels."""

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from ._validation import FloatArray, scalar


def normal_tails(x: ArrayLike, *, log: bool = False) -> tuple[FloatArray, FloatArray]:
    """Return standard-normal lower and upper tails, optionally as natural logs.

    The log representation preserves the extreme tails supported by CUMNOR,
    including x = +/-67,861,400. Plain probabilities may underflow to zero;
    the complementary tail is evaluated directly, never by subtracting from 1.
    Infinite arguments are valid distribution endpoints; NaNs are rejected.
    """
    values = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(values)):
        raise ValueError("x must not contain NaN")
    evaluate = log_ndtr if log else ndtr
    return np.asarray(evaluate(values)), np.asarray(evaluate(-values))


def invert_monotone(
    function: Callable[..., float],
    initial: float,
    target: float,
    *,
    lower: float = -1e300,
    upper: float = 1e300,
    parameter: str | None = None,
    function_kwargs: Mapping[str, object] | None = None,
    absolute_step: float = 1e-5,
    relative_step: float = 1e-3,
    step_multiplier: float = 2,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> float:
    """Solve function(x) = target for a continuous monotone scalar function.

    As in INVMF, check the endpoints, expand a bracket geometrically from the
    initial guess, and refine it. Refinement uses Brent's method rather than the
    legacy Algorithm R. Tolerance is absolute_tolerance + relative_tolerance*abs(x).
    The relative tolerance must be at least four double-precision epsilons.

    ``parameter`` selects a named argument to invert; otherwise x is positional.
    Other named arguments come from ``function_kwargs``. Calls are reentrant,
    including nested inversion, unlike the original S/Fortran implementation.
    Invalid evaluations, unbracketed targets, and nonconvergence raise exceptions.
    Monotonicity is checked at sampled points, not proved over the entire domain.
    """
    initial = scalar(initial, "initial")
    target = scalar(target, "target")
    lower, upper = scalar(lower, "lower"), scalar(upper, "upper")
    absolute_step = scalar(absolute_step, "absolute_step")
    relative_step = scalar(relative_step, "relative_step")
    step_multiplier = scalar(step_multiplier, "step_multiplier")
    absolute_tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = scalar(relative_tolerance, "relative_tolerance")
    if not lower < upper or not lower <= initial <= upper:
        raise ValueError("Require lower < upper and lower <= initial <= upper")
    if absolute_step < 0 or relative_step < 0 or step_multiplier <= 1:
        raise ValueError("Steps must be nonnegative and step_multiplier must exceed one")
    step = max(absolute_step, relative_step * abs(initial))
    if step <= 0:
        raise ValueError("The initial step must be positive")
    if absolute_tolerance <= 0 or relative_tolerance < 4 * np.finfo(float).eps:
        raise ValueError("Require positive absolute_tolerance and relative_tolerance >= 4*eps")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if parameter is not None and (not isinstance(parameter, str) or not parameter):
        raise ValueError("parameter must be a nonempty argument name")
    kwargs = dict(function_kwargs or {})
    if parameter in kwargs:
        raise ValueError("The inverted parameter must not also appear in function_kwargs")

    def evaluate(x: float) -> float:
        value = function(x, **kwargs) if parameter is None else function(**kwargs, **{parameter: x})
        residual = scalar(value, f"function({x})") - target
        if not np.isfinite(residual):
            raise ValueError(f"function({x}) - target must be finite")
        return residual

    low_value, high_value = evaluate(lower), evaluate(upper)
    if low_value == 0:
        return lower
    if high_value == 0:
        return upper
    if np.signbit(low_value) == np.signbit(high_value):
        raise ValueError("The endpoint function values do not bracket the target")
    value = evaluate(initial)
    if not min(low_value, high_value) <= value <= max(low_value, high_value):
        raise ValueError("Function evaluations violate monotonicity")
    if value == 0:
        return initial
    increasing = high_value > low_value
    rightward = (value < 0) == increasing
    previous, previous_value = initial, value
    left, right = lower, upper
    if absolute_step < upper - lower:
        for _ in range(max_iterations):
            candidate = min(previous + step, upper) if rightward else max(previous - step, lower)
            if candidate == previous:
                # A requested step may be below the local floating-point spacing.
                candidate = float(np.nextafter(previous, upper if rightward else lower))
            candidate_value = evaluate(candidate)
            if (rightward == increasing and candidate_value < previous_value) or (
                rightward != increasing and candidate_value > previous_value
            ):
                raise ValueError("Function evaluations violate monotonicity")
            if candidate_value == 0:
                return candidate
            if np.signbit(candidate_value) != np.signbit(previous_value):
                left, right = sorted((previous, candidate))
                break
            previous, previous_value = candidate, candidate_value
            step *= step_multiplier
        else:
            raise RuntimeError("Bracket expansion exceeded max_iterations")
    result = brentq(
        evaluate,
        left,
        right,
        xtol=absolute_tolerance,
        rtol=relative_tolerance,
        maxiter=max_iterations,
    )
    return float(result)
