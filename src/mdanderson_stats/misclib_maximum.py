"""Reentrant direct/reverse-communication MISCLIB scalar maximization."""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from math import copysign, isfinite, nextafter, sqrt, ulp

from ._validation import scalar


@dataclass(frozen=True)
class FunctionMaximum:
    """Status 1 requests f(x); 0 finishes; -2 exhausts the evaluation budget.

    Bounds are the retained search interval, not a certificate of global
    optimality. Value is defined only after completion or budget exhaustion.
    """

    status: int
    x: float
    value: float | None
    lower: float
    upper: float
    evaluations: int


class FunctionMaximizer:
    """One private continuation, constructed with set_fun_max; no global state."""

    def __init__(
        self, low: float, high: float, absolute: float, relative: float, budget: int
    ) -> None:
        self._bounds = (low, high)
        self._absolute, self._relative, self._budget = absolute, relative, budget
        self._search: Generator[FunctionMaximum, float, FunctionMaximum] | None = None
        self._result: FunctionMaximum | None = None

    @property
    def result(self) -> FunctionMaximum | None:
        return self._result

    def _advance(self, fx: float | None) -> FunctionMaximum:
        if self._search is None:
            if fx is not None:
                raise ValueError("First call must not supply fx")
            self._search = self._iterate()
            self._result = next(self._search)
        else:
            if self._result is None or self._result.status != 1:
                raise ValueError("Search has finished; construct a fresh state")
            if fx is None:
                raise ValueError("Supply the function value for the pending request")
            value = scalar(fx, "fx")
            try:
                self._result = self._search.send(value)
            except StopIteration as done:
                self._result = done.value
        assert self._result is not None
        if self._result.status == -2:
            raise ArithmeticError("Function maximization exhausted max_evaluations")
        return self._result

    def _iterate(self) -> Generator[FunctionMaximum, float, FunctionMaximum]:
        a, b = self._bounds
        x = a / 2 + b / 2
        w = v = x
        d = e = 0.0
        count = 1
        fx = -(yield FunctionMaximum(1, x, None, a, b, count))
        fw = fv = fx
        while True:
            midpoint = a / 2 + b / 2
            tolerance = max(
                max(sqrt(ulp(1.0)), self._relative) * abs(x) + self._absolute / 3,
                ulp(x),
            )
            if abs(x - midpoint) <= 2 * tolerance - (b - a) / 2 or nextafter(a, b) >= b:
                return FunctionMaximum(0, x, -fx, a, b, count)
            if count >= self._budget:
                return FunctionMaximum(-2, x, -fx, a, b, count)
            golden = abs(e) < tolerance
            if not golden:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0:
                    p = -p
                q = abs(q)
                old_e, e = e, d
                # Keep the source's signed old_e acceptance criterion. It is
                # conservative for leftward steps; golden refinement is valid.
                checks = (p, q, 0.5 * q * old_e, q * (a - x), q * (b - x))
                golden = (
                    not all(isfinite(t) for t in checks)
                    or abs(p) >= checks[2]
                    or p <= checks[3]
                    or p >= checks[4]
                )
                if not golden:
                    d = p / q
                    u = x + d
                    if u - a < 2 * tolerance or b - u < 2 * tolerance:
                        d = copysign(tolerance, midpoint - x)
            if golden:
                e = a - x if x >= midpoint else b - x
                d = 0.38196601125010515179541316563 * e
            u = x + (d if abs(d) >= tolerance else copysign(tolerance, d))
            if not a < u < b:
                # Adjacent floats or rounded steps: request a distinct interior
                # point where available, never evaluate outside the bounds.
                u = nextafter(x, a if d < 0 else b)
                if not a < u < b:
                    return FunctionMaximum(0, x, -fx, a, b, count)
            count += 1
            fu = -(yield FunctionMaximum(1, u, None, a, b, count))
            if fu > fx:
                if u < x:
                    a = u
                else:
                    b = u
                if fu < fw or w == x:
                    v, fv, w, fw = w, fw, u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
            else:
                if u >= x:
                    a = x
                else:
                    b = x
                v, fv, w, fw, x, fx = w, fw, x, fx, u, fu


def set_fun_max(
    low_limit: float = -1e35,
    hi_limit: float = 1e35,
    *,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-5,
    max_evaluations: int = 1000,
) -> FunctionMaximizer:
    """Configure a fresh bounded Brent/golden search for a unimodal maximum."""
    low, high = scalar(low_limit, "low_limit"), scalar(hi_limit, "hi_limit")
    absolute, relative = scalar(abs_tol, "abs_tol"), scalar(rel_tol, "rel_tol")
    if low >= high or not isfinite(high - low):
        raise ValueError("Bounds must increase and their difference must be finite")
    if absolute < 0 or not 0 <= relative < 1 or max(absolute, relative) == 0:
        raise ValueError("Tolerances must be nonnegative, one positive, and rel_tol < 1")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise ValueError("max_evaluations must be a positive integer")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    return FunctionMaximizer(low, high, absolute, relative, max_evaluations)


def rc_fun_max(local: FunctionMaximizer, fx: float | None = None) -> FunctionMaximum:
    """Start/resume a search, supplying f(x) after each status-1 request."""
    return local._advance(fx)


def fun_max(
    function: Callable[[float], float], *, local: FunctionMaximizer | None = None
) -> FunctionMaximum:
    """Maximize a finite scalar function on the configured interval."""
    state = local if local is not None else set_fun_max()
    if state.result is not None:
        raise ValueError("Direct search requires a fresh state")
    result = rc_fun_max(state)
    while result.status == 1:
        result = rc_fun_max(state, function(result.x))
    return result
