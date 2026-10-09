"""General Multc99 compound/conditional Dirichlet-outcome monitoring.

Adapted workflow follows Thall and Sung's Multc99 2.1. Upstream noncommercial
source terms remain applicable; see notices/mdanderson-multc99-readme.txt.
Native integration and memory defects are not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_difference

_MAX_EVENTS = 32
_MAX_ELEMENTS = 50
_MAX_SUBJECTS = 500
_MAX_INTEGRALS = 100_000


def _immutable[DType: np.generic](values: NDArray[DType]) -> NDArray[DType]:
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _real(value: float, name: str, minimum: float, maximum: float) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _mask(value: ArrayLike, name: str) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and len(value) > _MAX_ELEMENTS:
        raise ValueError(f"{name} exceeds {_MAX_ELEMENTS} elements")
    raw = np.asarray(value)
    if raw.ndim != 1 or not 2 <= raw.size <= _MAX_ELEMENTS or raw.dtype.kind not in "biu":
        raise ValueError(f"{name} must be a binary vector with 2--50 elements")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} must be binary")
    return tuple(int(x) for x in raw)


@dataclass(frozen=True)
class Multc99Event:
    """An event subset, optionally conditional on a larger subset.

    Definition is the numerator subset (the intersection for a conditional
    event). None disables a cutoff. Lower stopping uses P(E-S > margin) < pL;
    upper stopping uses P(E-S > margin) >= pU, matching source boundary code.
    """

    name: str
    definition: tuple[int, ...]
    conditioning_definition: tuple[int, ...] | None = None
    lower_margin: float = 0.0
    lower_cutoff: float | None = None
    upper_margin: float = 0.0
    upper_cutoff: float | None = None
    event_type: Literal["efficacy", "adverse", "other"] = "other"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("event name must be nonempty text of at most 100 characters")
        definition = _mask(self.definition, "definition")
        conditioning = (
            tuple(1 for _ in definition)
            if self.conditioning_definition is None
            else _mask(self.conditioning_definition, "conditioning_definition")
        )
        if (
            len(definition) != len(conditioning)
            or any(d > c for d, c in zip(definition, conditioning, strict=True))
            or not 0 < sum(definition) < sum(conditioning)
        ):
            raise ValueError("definition must be a nonempty proper subset of the conditioning set")
        if self.event_type not in {"efficacy", "adverse", "other"}:
            raise ValueError("event_type must be efficacy, adverse, or other")
        object.__setattr__(self, "definition", definition)
        if self.conditioning_definition is not None:
            object.__setattr__(self, "conditioning_definition", conditioning)
        for side in ("lower", "upper"):
            object.__setattr__(
                self,
                f"{side}_margin",
                _real(getattr(self, f"{side}_margin"), f"{side}_margin", -1, 1),
            )
            cutoff = getattr(self, f"{side}_cutoff")
            if cutoff is not None:
                object.__setattr__(self, f"{side}_cutoff", _real(cutoff, f"{side}_cutoff", 0, 1))


@dataclass(frozen=True)
class Multc99Probability:
    probability: float
    absolute_error: float
    event_count: int
    conditioning_count: int


@dataclass(frozen=True)
class Multc99State:
    """All event decisions at one look, preserving simultaneous boundary hits."""

    sample_size: int
    elementary_counts: NDArray[np.int64]
    event_names: tuple[str, ...]
    event_counts: NDArray[np.int64]
    conditioning_counts: NDArray[np.int64]
    lower_bounds: NDArray[np.int64]
    upper_bounds: NDArray[np.int64]
    lower_hits: NDArray[np.bool_]
    upper_hits: NDArray[np.bool_]
    decision: Literal["continue", "stop", "cap_complete"]


@dataclass(frozen=True)
class Multc99Design:
    """Construct with multc99_design; priors and precomputed bounds are immutable."""

    experimental_prior: FloatArray
    historical_prior: FloatArray
    historical_weights: FloatArray
    events: tuple[Multc99Event, ...]
    max_subjects: int
    min_subjects: int
    cohort_size: int
    lower_bounds: NDArray[np.int64]
    upper_bounds: NDArray[np.int64]
    _numerators: NDArray[np.int64]
    _denominators: NDArray[np.int64]
    _experimental_beta: FloatArray
    _historical_beta: FloatArray
    boundary_origin: Literal["posterior", "manual"] = "posterior"

    def _counts(self, counts: ArrayLike) -> NDArray[np.int64]:
        raw = np.asarray(counts)
        if raw.shape != self.experimental_prior.shape or raw.dtype.kind not in "iu":
            raise ValueError("elementary_counts must be an integer vector matching the priors")
        if np.any(raw < 0) or np.any(raw > self.max_subjects):
            raise ValueError("elementary_counts must be nonnegative and bounded by max_subjects")
        values = np.asarray(raw, dtype=np.int64)
        if int(values.sum()) > self.max_subjects:
            raise ValueError("elementary counts sum exceeds max_subjects")
        return values

    def event_probability(
        self,
        event_name: str,
        elementary_counts: ArrayLike,
        *,
        margin: float = 0.0,
        absolute_tolerance: float = 1e-9,
    ) -> Multc99Probability:
        """P(experimental event rate - independent historical rate > margin).

        Experimental Dirichlet counts update all elementary outcomes. A
        conditional event uses only observations in its conditioning subset.
        Historical mixture weights remain fixed; historical data are not updated.
        """
        counts = self._counts(elementary_counts)
        names = tuple(e.name for e in self.events)
        if event_name not in names:
            raise ValueError("unknown event_name")
        index = names.index(event_name)
        delta = _real(margin, "margin", -1, 1)
        tolerance = _real(absolute_tolerance, "absolute_tolerance", 1e-12, 1e-3)
        x = int(self._numerators[index] @ counts)
        n = int(self._denominators[index] @ counts)
        probability, error = _probability(
            self._experimental_beta[index],
            self._historical_beta[index],
            self.historical_weights,
            x,
            n,
            delta,
            tolerance,
        )
        return Multc99Probability(probability, error, x, n)

    def monitor_counts(self, elementary_counts: ArrayLike) -> Multc99State:
        """Check a caller-selected look; sample-size zero and the cap never stop.

        Before Nmin, compound rules use the source's semi-free-ride limits.
        Conditional rules index their tables by the parent-subset count; the
        source's lower-bound run-back still applies to those tables.
        """
        counts = self._counts(elementary_counts)
        total = int(counts.sum())
        x = self._numerators @ counts
        n = self._denominators @ counts
        indices = np.arange(len(self.events))
        lower = np.array(self.lower_bounds[indices, n], copy=True)
        upper = np.array(self.upper_bounds[indices, n], copy=True)
        if 0 < total < self.min_subjects:
            for j, event in enumerate(self.events):
                if event.conditioning_definition is None:
                    lower[j] = total + self.lower_bounds[j, self.min_subjects] - self.min_subjects
                    upper[j] = self.upper_bounds[j, self.min_subjects]
        lower_hits = x <= lower
        upper_hits = x >= upper
        if total in (0, self.max_subjects):
            lower_hits[:] = False
            upper_hits[:] = False
        decision: Literal["continue", "stop", "cap_complete"] = (
            "cap_complete"
            if total == self.max_subjects
            else "stop"
            if np.any(lower_hits | upper_hits)
            else "continue"
        )
        return Multc99State(
            total,
            _immutable(counts),
            tuple(e.name for e in self.events),
            _immutable(x),
            _immutable(n),
            _immutable(lower),
            _immutable(upper),
            _immutable(lower_hits),
            _immutable(upper_hits),
            decision,
        )


def _probability(
    experimental: FloatArray,
    historical: FloatArray,
    weights: FloatArray,
    x: int,
    n: int,
    margin: float,
    tolerance: float,
) -> tuple[float, float]:
    posterior = BetaBinomialPosterior(experimental[0] + x, experimental[1] + n - x)
    if margin == 0 and np.all(historical[weights > 0] == 1):
        # An independent uniform historical rate gives P(E > S) = E[E].
        # Avoid losing an exact cutoff tie in numerical integration.
        return float(posterior.alpha / (posterior.alpha + posterior.beta)), 0.0
    probability, error = 0.0, 0.0
    for prior, weight in zip(historical, weights, strict=True):
        if weight == 0:
            continue
        if margin == 0 and prior[0] == prior[1] and posterior.alpha == posterior.beta:
            p, e = 0.5, 0.0
        else:
            result = compare_beta_difference(
                BetaBinomialPosterior(prior[0], prior[1]),
                posterior,
                margin,
                absolute_tolerance=tolerance,
            )
            p, e = float(result.above_margin), float(result.absolute_error)
        probability += float(weight) * p
        error += float(weight) * e
    return probability, error


def _prior(values: ArrayLike, name: str, *, mixture: bool) -> FloatArray:
    if isinstance(values, (list, tuple)) and len(values) > _MAX_ELEMENTS:
        raise ValueError(f"{name} exceeds its dimension limit")
    if isinstance(values, (list, tuple)) and any(
        isinstance(row, (list, tuple, np.ndarray)) for row in values
    ):
        if not mixture or len(values) > 10:
            raise ValueError(f"{name} exceeds its dimension limit")
        for row in values:
            if isinstance(row, (list, tuple)) and (
                len(row) > _MAX_ELEMENTS
                or any(isinstance(x, (list, tuple, np.ndarray)) for x in row)
            ):
                raise ValueError(f"{name} exceeds its dimension limit")
            if isinstance(row, np.ndarray) and (row.ndim != 1 or row.size > _MAX_ELEMENTS):
                raise ValueError(f"{name} exceeds its dimension limit")
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.size > 500:
        raise ValueError(f"{name} must be a bounded positive real vector/matrix")
    if mixture and raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != (2 if mixture else 1) or not 2 <= raw.shape[-1] <= _MAX_ELEMENTS:
        raise ValueError(f"{name} must have 2--50 elementary-outcome parameters")
    if mixture and not 1 <= raw.shape[0] <= 10:
        raise ValueError("historical_prior allows 1--10 Dirichlet components")
    array = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0) or np.any(array.sum(axis=-1) > 1e6):
        raise ValueError(f"{name} must be positive, finite, with component total <= 1e6")
    return _freeze(array)


def multc99_design(
    experimental_prior: ArrayLike,
    historical_prior: ArrayLike,
    events: tuple[Multc99Event, ...] | list[Multc99Event],
    *,
    max_subjects: int,
    min_subjects: int = 1,
    cohort_size: int = 1,
    historical_weights: ArrayLike | None = None,
) -> Multc99Design:
    """Build compound/conditional monitoring from elementary Dirichlet priors.

    Bounds cover 0--max_subjects. Lower cutoffs are strict; upper cutoffs are
    inclusive as in the recovered C implementation, despite its strict prose.
    Disabled/unattainable bounds use -1 and n+1, not native +/-99 sentinels.
    """
    maximum = _integer(max_subjects, "max_subjects", 2, _MAX_SUBJECTS)
    minimum = _integer(min_subjects, "min_subjects", 1, maximum - 1)
    cohort = _integer(cohort_size, "cohort_size", 1, maximum)
    ep = _prior(experimental_prior, "experimental_prior", mixture=False)
    hp = _prior(historical_prior, "historical_prior", mixture=True)
    if hp.shape[1] != ep.size:
        raise ValueError("experimental and historical priors must have matching outcomes")
    if not isinstance(events, (tuple, list)) or not 1 <= len(events) <= _MAX_EVENTS:
        raise ValueError(f"events must be a sequence of 1--{_MAX_EVENTS} Multc99Event objects")
    ev = tuple(events)
    if any(not isinstance(e, Multc99Event) or len(e.definition) != ep.size for e in ev):
        raise ValueError("event definitions must match the priors")
    if len({e.name for e in ev}) != len(ev):
        raise ValueError("event names must be unique")
    if historical_weights is None:
        if hp.shape[0] != 1:
            raise ValueError("mixture historical_prior requires explicit historical_weights")
        weights = np.ones(1)
    else:
        raw = np.asarray(historical_weights)
        if raw.dtype.kind not in "iuf" or raw.shape != (hp.shape[0],):
            raise ValueError("historical_weights must match the number of components")
        weights = np.asarray(raw, dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or abs(weights.sum() - 1) > 1e-12:
            raise ValueError("historical_weights must be nonnegative and sum to one")
        weights = weights / weights.sum()
    weights = _freeze(weights)
    numerator = np.array([e.definition for e in ev], dtype=np.int64)
    denominator = np.array(
        [e.conditioning_definition or tuple(1 for _ in e.definition) for e in ev], dtype=np.int64
    )
    complement = denominator - numerator
    ebeta = np.column_stack((numerator @ ep, complement @ ep))
    hbeta = np.stack((numerator @ hp.T, complement @ hp.T), axis=-1)
    # Each binary search has <=10 probability evaluations, each with <=3
    # refinements. Budget by active cutoffs and nonzero mixture components.
    calls = sum(int(e.lower_cutoff is not None) + int(e.upper_cutoff is not None) for e in ev)
    calls *= maximum * int(np.count_nonzero(weights)) * 10 * 3
    if calls > _MAX_INTEGRALS:
        raise ValueError("design exceeds the 100,000-comparison construction budget")
    lower = np.full((len(ev), maximum + 1), -1, dtype=np.int64)
    upper = np.broadcast_to(np.arange(maximum + 1) + 1, lower.shape).copy()

    @lru_cache(maxsize=_MAX_INTEGRALS)
    def greater_or_equal(index: int, x: int, n: int, margin: float, cutoff: float) -> bool:
        for tolerance in (1e-9, 1e-11, 1e-12):
            probability, error = _probability(
                ebeta[index],
                hbeta[index],
                weights,
                x,
                n,
                margin,
                tolerance,
            )
            if error == 0 or abs(probability - cutoff) > error:
                return probability >= cutoff
        raise ArithmeticError(
            "Multc99 boundary comparison is unresolved after quadrature refinement"
        )

    def first_ge(index: int, n: int, margin: float, cutoff: float) -> int:
        if cutoff == 0 or margin == -1:
            return 0
        if cutoff == 1 or margin == 1:
            return n + 1
        low, high = 0, n + 1
        while low < high:
            mid = (low + high) // 2
            if greater_or_equal(index, mid, n, margin, cutoff):
                high = mid
            else:
                low = mid + 1
        return low

    for j, event in enumerate(ev):
        for n in range(1, maximum + 1):
            if event.lower_cutoff is not None:
                lower[j, n] = first_ge(j, n, event.lower_margin, event.lower_cutoff) - 1
            if event.upper_cutoff is not None:
                upper[j, n] = first_ge(j, n, event.upper_margin, event.upper_cutoff)
        if minimum > 1:
            for n in range(1, minimum):
                if event.lower_cutoff is not None:
                    lower[j, n] = lower[j, minimum] + n - minimum
                if event.upper_cutoff is not None and event.conditioning_definition is None:
                    upper[j, n] = upper[j, minimum]
    return Multc99Design(
        ep,
        hp,
        weights,
        ev,
        maximum,
        minimum,
        cohort,
        _immutable(lower),
        _immutable(upper),
        _immutable(numerator),
        _immutable(denominator),
        _freeze(ebeta),
        _freeze(hbeta),
    )
