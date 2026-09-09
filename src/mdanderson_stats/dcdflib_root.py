"""Independent legacy DCDFLIB reverse-communication root searches."""

from dataclasses import dataclass
from typing import Literal

from .cdflib_root import ZeroFinder, ZeroFinderResult, _Settings, _validated_settings


class _LegacyEngine(ZeroFinder):
    def _refinement_goal(self, a: float, fa: float, b: float, fb: float) -> tuple[float, float]:
        best = b if abs(fb) < abs(fa) else a
        relative = self._settings.relative
        # Halve before multiplying, avoiding overflow of the unhalved tolerance.
        half_relative = relative * (abs(best) / 2) if relative < 1 else (relative / 2) * abs(best)
        return best, max(self._settings.absolute / 2, half_relative)


@dataclass(frozen=True)
class LegacyRootResult:
    """Status 1 requests fx at x, 0 succeeds, -1 cannot bracket, -2 exhausts budget.

    On success xlo=x is the endpoint with the smaller absolute residual; xhi is
    the opposite endpoint, not necessarily numerically larger. Bracket fields
    are None otherwise. qleft/qhi are defined only on status -1.
    """

    status: int
    x: float
    evaluations: int
    xlo: float | None = None
    xhi: float | None = None
    qleft: bool | None = None
    qhi: bool | None = None


class LegacyRootFinder:
    """One independent continuation, created by dstinv or dstzr.

    States can be interleaved. Each state runs one search; create a new state to
    restart. Concurrent access to the same state is unsupported.
    """

    def __init__(self, mode: Literal["step", "interval"], settings: _Settings) -> None:
        self._mode = mode
        self._engine = _LegacyEngine(settings)
        self._high = settings.high
        self._endpoints: list[float] = []
        self._result: LegacyRootResult | None = None

    @property
    def result(self) -> LegacyRootResult | None:
        """Latest immutable request/completion, or None before starting."""
        return self._result

    def _record(self, result: ZeroFinderResult) -> LegacyRootResult:
        if result.status == 0:
            other = result.bound_high if result.x == result.bound_low else result.bound_low
            mapped = LegacyRootResult(0, result.x, result.evaluations, result.x, other)
        elif result.status == -1:
            left = result.crash_left
            # Native DINVR treats equal endpoint values as decreasing. For a
            # constant negative residual its QLEFT differs from DZROR's.
            if self._mode == "step" and self._endpoints[0] == self._endpoints[1]:
                left = not result.crash_hi
            mapped = LegacyRootResult(
                -1, self._high, result.evaluations, qleft=left, qhi=result.crash_hi
            )
        else:
            mapped = LegacyRootResult(result.status, result.x, result.evaluations)
        self._result = mapped
        return mapped

    def _advance(
        self, mode: Literal["step", "interval"], fx: float | None, initial: float | None
    ) -> LegacyRootResult:
        if mode != self._mode:
            raise ValueError("Use dinvr with dstinv state and dzror with dstzr state")
        try:
            result = self._engine._advance(mode, fx, initial)
        except ArithmeticError:
            terminal = self._engine.result
            if terminal is not None and terminal.status == -2:
                self._record(terminal)
            raise
        if fx is not None and len(self._endpoints) < 2:
            self._endpoints.append(float(fx))
        return self._record(result)


def dstinv(
    small: float,
    big: float,
    absstp: float,
    relstp: float,
    stpmul: float,
    abstol: float,
    reltol: float,
    *,
    max_evaluations: int = 4096,
) -> LegacyRootFinder:
    """Configure a legacy monotone stepping search; return independent state."""
    settings = _validated_settings(
        low_limit=small,
        hi_limit=big,
        abs_step=absstp,
        rel_step=relstp,
        step_multiplier=stpmul,
        abs_tol=abstol,
        rel_tol=reltol,
        max_evaluations=max_evaluations,
        allow_equal=True,
    )
    return LegacyRootFinder("step", settings)


def dstzr(
    xlo: float, xhi: float, abstol: float, reltol: float, *, max_evaluations: int = 4096
) -> LegacyRootFinder:
    """Configure a legacy bracket refinement; return independent state."""
    return LegacyRootFinder(
        "interval",
        _validated_settings(
            low_limit=xlo,
            hi_limit=xhi,
            abs_tol=abstol,
            rel_tol=reltol,
            max_evaluations=max_evaluations,
            abs_step=0,
            rel_step=0,
            step_multiplier=2,
            allow_equal=True,
        ),
    )


def dinvr(
    local: LegacyRootFinder, fx: float | None = None, *, initial: float | None = None
) -> LegacyRootResult:
    """Start/resume dstinv search; initial is required only on the first call."""
    return local._advance("step", fx, initial)


def dzror(local: LegacyRootFinder, fx: float | None = None) -> LegacyRootResult:
    """Start/resume dstzr search; supply fx only for an outstanding request."""
    return local._advance("interval", fx, None)
