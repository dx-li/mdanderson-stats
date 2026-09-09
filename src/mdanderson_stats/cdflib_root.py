"""Reentrant CDFLIB root searches with direct and reverse communication.

Refinement uses safeguarded inverse quadratic/secant interpolation and bisection,
not the archived TOMS 748 implementation. See docs/cdflib-root.md for the protocol.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from math import isfinite, nextafter

from ._validation import scalar


@dataclass(frozen=True)
class ZeroFinderResult:
    """Immutable request/result; status 1 requests f(x)-y, 0 succeeds, -1 fails.

    Status -2 records evaluation-budget exhaustion, which also raises
    ArithmeticError. Bounds enclose the retained bracket on success. Failure
    flags are defined only for status -1; they describe endpoint residuals,
    not a proof that no interior root exists.
    """

    status: int
    x: float
    evaluations: int
    bound_low: float
    bound_high: float
    crash_left: bool | None = None
    crash_hi: bool | None = None


@dataclass(frozen=True)
class _Settings:
    low: float
    high: float
    absolute: float
    relative: float
    absolute_step: float
    relative_step: float
    multiplier: float
    budget: int


class ZeroFinder:
    """One search's private continuation and public immutable snapshot.

    Construct with set_zero_finder. Each state can run exactly one search;
    construct another to restart. Independent states may be interleaved or used
    by nested callbacks. Concurrent access to the *same* state is unsupported.
    """

    def __init__(self, settings: _Settings) -> None:
        self._settings = settings
        self._result: ZeroFinderResult | None = None
        self._search: Generator[float, float, ZeroFinderResult] | None = None
        self._mode: str | None = None
        self._evaluations = 0
        self._low, self._high = settings.low, settings.high

    @property
    def result(self) -> ZeroFinderResult | None:
        """Latest request/completion, or None before the first request."""
        return self._result

    def _finish(
        self,
        x: float,
        status: int = 0,
        crash_left: bool | None = None,
        crash_hi: bool | None = None,
    ) -> ZeroFinderResult:
        return ZeroFinderResult(
            status, x, self._evaluations, self._low, self._high, crash_left, crash_hi
        )

    def _exact(self, x: float) -> ZeroFinderResult:
        self._low = self._high = x
        return self._finish(x)

    def _advance(self, mode: str, fx: float | None, initial: float | None) -> ZeroFinderResult:
        if self._mode is not None and self._mode != mode:
            raise ValueError("Cannot switch search modes on an existing state")
        if self._result is not None and self._result.status != 1:
            raise ValueError("Search has finished; construct a new state to restart")
        if self._search is None:
            if fx is not None:
                raise ValueError("The first call must not supply fx")
            if mode == "step":
                if initial is None:
                    raise ValueError("The first step search call requires initial")
                initial = scalar(initial, "initial")
                if not self._low <= initial <= self._high:
                    raise ValueError("initial must lie inside the configured limits")
                if (
                    max(self._settings.absolute_step, self._settings.relative_step * abs(initial))
                    == 0
                ):
                    raise ValueError("The initial step must be positive")
            self._mode = mode
            self._search = self._solve(initial)
            value = None
        else:
            if initial is not None:
                raise ValueError("Supply initial only on the first call")
            if fx is None:
                raise ValueError("Supply the residual fx for the outstanding request")
            value = scalar(fx, "fx")
            self._evaluations += 1
        try:
            x = next(self._search) if value is None else self._search.send(value)
        except StopIteration as done:
            self._result = done.value
            return self._result
        if self._evaluations >= self._settings.budget:
            self._search.close()
            self._result = self._finish(x, -2)
            raise ArithmeticError("Root search exhausted max_evaluations")
        self._result = self._finish(x, 1)
        return self._result

    def _solve(self, initial: float | None) -> Generator[float, float, ZeroFinderResult]:
        a, b = self._low, self._high
        fa = yield a
        if fa == 0:
            return self._exact(a)
        fb = yield b
        if fb == 0:
            return self._exact(b)
        if (fa > 0) == (fb > 0):
            left = (fa < fb) if fa > 0 else (fa > fb)
            return self._finish(a if left else b, -1, left, fa > 0)
        if initial is not None:
            fi = fa if initial == a else fb if initial == b else (yield initial)
            if fi == 0:
                return self._exact(initial)
            increasing = fb > fa
            up = (fi < 0) == increasing
            step = max(self._settings.absolute_step, self._settings.relative_step * abs(initial))
            previous, fp = initial, fi
            while True:
                candidate = min(previous + step, b) if up else max(previous - step, a)
                if candidate == previous:
                    candidate = nextafter(previous, b if up else a)
                fc = fb if candidate == b else fa if candidate == a else (yield candidate)
                if fc == 0:
                    return self._exact(candidate)
                if (fc > 0) != (fp > 0):
                    if up:
                        a, fa, b, fb = previous, fp, candidate, fc
                    else:
                        a, fa, b, fb = candidate, fc, previous, fp
                    break
                previous, fp = candidate, fc
                step *= self._settings.multiplier
        return (yield from self._refine(a, fa, b, fb))

    def _refine(
        self,
        a: float,
        fa: float,
        b: float,
        fb: float,
    ) -> Generator[float, float, ZeroFinderResult]:
        old: tuple[float, float] | None = None
        saved_half_width = float("inf")
        iteration = 0
        while True:
            self._low, self._high = a, b
            middle = a / 2 + b / 2
            half_width = b / 2 - a / 2
            tol = self._settings.absolute + self._settings.relative * min(abs(a), abs(b))
            if half_width <= tol or nextafter(a, b) == b:
                return self._finish(middle)
            # At least halve the bracket every three proposals. This bounds
            # progress even for extremely flat residuals or rejected interpolants.
            bisect = iteration % 3 == 2 and half_width > saved_half_width / 2
            if iteration % 3 == 0:
                saved_half_width = half_width
            c = middle if bisect else _interpolate(a, fa, b, fb, old)
            if not isfinite(c) or not a < c < b:
                c = middle
            if not a < c < b:
                c = nextafter(a, b)
            fc = yield c
            if fc == 0:
                return self._exact(c)
            if (fa > 0) != (fc > 0):
                old, b, fb = (b, fb), c, fc
            else:
                old, a, fa = (a, fa), c, fc
            iteration += 1


def _interpolate(
    a: float,
    fa: float,
    b: float,
    fb: float,
    old: tuple[float, float] | None,
) -> float:
    """Normalize ordinates and abscissae to limit intermediate overflow."""
    scale = max(abs(fa), abs(fb), abs(old[1]) if old else 0)
    u, v = fa / scale, fb / scale
    if u == 0 or v == 0:
        return a / 2 + b / 2
    fraction = -u / (v - u)
    if old is not None:
        d, fd = old
        w = fd / scale
        if w != u and w != v:
            span = b / 2 - a / 2
            location = (d / 2 - a / 2) / span
            try:
                estimate = u * w / ((v - u) * (v - w))
                estimate += location * u * v / ((w - u) * (w - v))
            except ZeroDivisionError:
                estimate = fraction
            if isfinite(estimate) and 0 < estimate < 1:
                fraction = estimate
    return (1 - fraction) * a + fraction * b


def set_zero_finder(
    *,
    low_limit: float = -1e35,
    hi_limit: float = 1e35,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    abs_step: float = 1e-4,
    rel_step: float = 1,
    step_multiplier: float = 2,
    max_evaluations: int = 4096,
) -> ZeroFinder:
    """Create an independent search; defaults match CDFLIB except the work cap.

    Limits must be finite and increasing, tolerances/steps nonnegative (at least
    one tolerance positive), and step_multiplier > 1. Tolerance bounds the
    returned midpoint's absolute error for a continuous bracketed function by
    abs_tol + rel_tol*min(abs(left), abs(right)), subject to float resolution.
    """
    low, high = scalar(low_limit, "low_limit"), scalar(hi_limit, "hi_limit")
    absolute, relative = scalar(abs_tol, "abs_tol"), scalar(rel_tol, "rel_tol")
    astep, rstep = scalar(abs_step, "abs_step"), scalar(rel_step, "rel_step")
    multiplier = scalar(step_multiplier, "step_multiplier")
    if low >= high:
        raise ValueError("Require low_limit < hi_limit")
    if absolute < 0 or relative < 0 or max(absolute, relative) == 0:
        raise ValueError("Tolerances must be nonnegative and at least one positive")
    if astep < 0 or rstep < 0 or multiplier <= 1:
        raise ValueError("Steps must be nonnegative and step_multiplier must exceed one")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    return ZeroFinder(
        _Settings(low, high, absolute, relative, astep, rstep, multiplier, max_evaluations)
    )


def rc_interval_zf(local: ZeroFinder, fx: float | None = None) -> ZeroFinderResult:
    """Request/resume interval search; supply residual fx only after a request."""
    return local._advance("interval", fx, None)


def rc_step_zf(
    local: ZeroFinder,
    fx: float | None = None,
    *,
    initial: float | None = None,
) -> ZeroFinderResult:
    """Request/resume monotone step search; initial is required only at startup."""
    return local._advance("step", fx, initial)


def final_zf_state(local: ZeroFinder) -> ZeroFinderResult | None:
    """Return the latest immutable state, including pending/completed status."""
    return local.result


def _direct(
    function: Callable[[float], float],
    y: float,
    local: ZeroFinder,
    mode: str,
    initial: float | None,
) -> ZeroFinderResult:
    y = scalar(y, "y")
    if local.result is not None:
        raise ValueError("Direct search requires a fresh state")
    result = local._advance(mode, None, initial)
    while result.status == 1:
        residual = scalar(function(result.x), "function(x)") - y
        result = local._advance(mode, residual, None)
    return result


def interval_zf(
    function: Callable[[float], float],
    y: float = 0,
    *,
    local: ZeroFinder | None = None,
) -> ZeroFinderResult:
    """Solve continuous f(x)=y in configured bounds; return status and bracket."""
    return _direct(function, y, local if local is not None else set_zero_finder(), "interval", None)


def step_zf(
    function: Callable[[float], float],
    initial: float,
    y: float = 0,
    *,
    local: ZeroFinder | None = None,
) -> ZeroFinderResult:
    """Geometrically bracket a continuous monotone equation from initial."""
    return _direct(function, y, local if local is not None else set_zero_finder(), "step", initial)
