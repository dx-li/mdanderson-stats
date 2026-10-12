"""Explicit safe and UAROET1.8-compatible decision primitives."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _integer, _real_matrix


def _profile(profile: str) -> None:
    if profile not in ("safe", "legacy_1_8"):
        raise ValueError("profile must be safe or legacy_1_8")


def _vector(
    value: ArrayLike, name: str, doses: int, *, probability=False, counts=False
) -> FloatArray:
    a = _real_matrix(value, name)
    if (
        a.shape != (doses,)
        or np.any(a < 0)
        or (probability and np.any(a > 1))
        or (counts and (np.any(a != np.floor(a)) or np.any(a > 10000)))
    ):
        raise ValueError(f"{name} has invalid dose vector values")
    return a


def uaroet_utility_cutoff(
    *, planned: int, steps: int, observed: int, initial: float, final: float, profile: str = "safe"
) -> float:
    """Legacy integer blocks; safe mode prevents the K=N zero-width collapse."""
    _profile(profile)
    N = _integer(planned, "planned", 1, 500)
    K = _integer(steps, "steps", 0, N)
    n = _integer(observed, "observed", 0, 10000)
    a = _real_matrix(initial, "initial")
    b = _real_matrix(final, "final")
    if a.ndim or b.ndim or not 0 <= float(b) <= float(a) <= 100:
        raise ValueError("require 0 <= final <= initial <= 100")
    if K == 0:
        return float(a)
    width = N // (K + 1)
    if profile == "safe":
        width = max(1, width)
    completed = min(K, n // width) if width else K
    return float(a) - (float(a) - float(b)) * completed / K


def uaroet_desktop_screen(
    toxicity_exceedance: ArrayLike,
    efficacy_shortfall: ArrayLike,
    mean_utility: ArrayLike,
    probability_best: ArrayLike,
    *,
    bad_toxicity: ArrayLike,
    bad_efficacy: ArrayLike,
    complete: ArrayLike,
    highest_allowed: int,
    toxicity_cutoff: float = 0.8,
    efficacy_cutoff: float = 0.8,
    utility_delta: float = 100,
    minimum_best: float = 0,
    minimum_bad_efficacy: int = 3,
    minimum_complete: int = 3,
    minimum_total_toxicity: int = 3,
    filter_utility: bool = True,
    profile: str = "safe",
) -> tuple[int, ...]:
    """Return1 eligible; -1 futility,-2 toxicity,-3 utility,-4 best,-5 no-skip.

    Legacy gates count only complete observations, can restore a toxic lowest
    dose, and propagate toxicity only when a higher dose's own cutoff also fails.
    Safe mode applies posterior safety/futility without observed-event gates and
    never restores a toxic dose. This is an explicit policy difference, not a
    claim of clinical validation. Utility filters retain the complete-count gate.
    """
    _profile(profile)
    raw = _real_matrix(mean_utility, "mean_utility")
    if raw.ndim != 1 or not 1 <= raw.size <= 5:
        raise ValueError("require one through five doses")
    D = raw.size
    u = _vector(raw, "mean_utility", D)
    t = _vector(toxicity_exceedance, "toxicity_exceedance", D, probability=True)
    e = _vector(efficacy_shortfall, "efficacy_shortfall", D, probability=True)
    best = _vector(probability_best, "probability_best", D, probability=True)
    bt = _vector(bad_toxicity, "bad_toxicity", D, counts=True)
    be = _vector(bad_efficacy, "bad_efficacy", D, counts=True)
    c = _vector(complete, "complete", D, counts=True)
    if np.any(bt > c) or np.any(be > c):
        raise ValueError("bad-outcome counts cannot exceed complete counts")
    highest = _integer(highest_allowed, "highest_allowed", 0, D - 1)
    for name, v in [
        ("toxicity_cutoff", toxicity_cutoff),
        ("efficacy_cutoff", efficacy_cutoff),
        ("minimum_best", minimum_best),
    ]:
        a = _real_matrix(v, name)
        if a.ndim or not 0 <= float(a) <= 1:
            raise ValueError(f"{name} must be a probability")
    delta = _real_matrix(utility_delta, "utility_delta")
    if delta.ndim or not 0 <= float(delta) <= 100:
        raise ValueError("utility_delta must be in [0,100]")
    mb = _integer(minimum_bad_efficacy, "minimum_bad_efficacy", 0, 10000)
    mc = _integer(minimum_complete, "minimum_complete", 0, 10000)
    mt = _integer(minimum_total_toxicity, "minimum_total_toxicity", 0, 10000)
    if not isinstance(filter_utility, (bool, np.bool_)):
        raise ValueError("filter_utility must be boolean")
    status = []
    lower_toxic = False
    for d in range(D):
        if d > highest:
            value = -5
        elif t[d] > toxicity_cutoff and (profile == "safe" or bt[d] > 0 or lower_toxic):
            value = -2
            lower_toxic = True
        elif e[d] > efficacy_cutoff and (profile == "safe" or be[d] >= mb):
            value = -1
        else:
            value = 1
        if value == 1 and c[d] >= mc and filter_utility:
            if np.max(u) - u[d] > float(delta):
                value = -3
            elif best[d] < minimum_best:
                value = -4
        status.append(value)
    if profile == "legacy_1_8" and 1 not in status and status[0] == -2 and bt.sum() < mt:
        status[0] = 1
    return tuple(status)


@dataclass(frozen=True)
class UAROETDesktopAssignment:
    selected: int | None
    weights: FloatArray
    reason: str


def uaroet_desktop_assignment(
    status: tuple[int, ...],
    mean_utility: ArrayLike,
    good_probability: ArrayLike,
    treated: ArrayLike,
    *,
    hybrid: bool = True,
    mean_weight: bool = True,
    surplus: int = 1,
    final: bool = False,
    uniform: float = 0.5,
    profile: str = "safe",
) -> UAROETDesktopAssignment:
    """Hybrid last-tie selection is legacy-only; plain/final keep the first tie.

    Explicit uniform input makes randomization reviewable and reproducible.
    Zero eligible weight returns no assignment rather than inventing a law.
    """
    _profile(profile)
    D = len(status)
    if not 1 <= D <= 5 or any(
        isinstance(s, (bool, np.bool_)) or s not in (1, -1, -2, -3, -4, -5) for s in status
    ):
        raise ValueError("invalid status vector")
    u = _vector(mean_utility, "mean_utility", D)
    g = _vector(good_probability, "good_probability", D, probability=True)
    n = _vector(treated, "treated", D, counts=True)
    s = _integer(surplus, "surplus", 0, 10000)
    v = _real_matrix(uniform, "uniform")
    if v.ndim or not 0 <= float(v) <= 1:
        raise ValueError("uniform must lie in [0,1]")
    if any(not isinstance(b, (bool, np.bool_)) for b in (hybrid, mean_weight, final)):
        raise ValueError("assignment flags must be boolean")
    eligible = [i for i, x in enumerate(status) if x == 1]
    w = np.zeros(D)
    if not eligible:
        return UAROETDesktopAssignment(None, _freeze(w), "no_eligible_dose")
    best = max(eligible, key=lambda i: (u[i], i if profile == "legacy_1_8" and not final else -i))
    allocations = sorted((n[i] for i in eligible), reverse=True)
    greedy = (
        final
        or len(eligible) == 1
        or (hybrid and (n[best] != allocations[0] or n[best] - allocations[1] < s))
    )
    if greedy:
        w[best] = 1
        return UAROETDesktopAssignment(best, _freeze(w), "greedy")
    w[eligible] = (u if mean_weight else g)[eligible]
    if w.sum() == 0:
        return UAROETDesktopAssignment(None, _freeze(w), "zero_eligible_weight")
    w /= w.sum()
    cumulative = w.cumsum()
    chosen = next((i for i in eligible if float(v) <= cumulative[i]), eligible[-1])
    return UAROETDesktopAssignment(chosen, _freeze(w), "randomized")


@dataclass(frozen=True)
class UAROETCohortState:
    """Live allocation bookkeeping, separate from the native saved-history replay."""

    first_seen: tuple[int, ...]
    total: int = 0
    last: int = 0
    new_open: bool = False
    old_open: bool = False
    new_start: int = 0
    old_start: int = 0

    def __post_init__(self) -> None:
        D = len(self.first_seen)
        if not 1 <= D <= 5:
            raise ValueError("cohort state requires one through five doses")
        total = _integer(self.total, "total", 0, 500)
        _integer(self.last, "last", 0, D - 1)
        first = tuple(_integer(n, "first_seen", 0, total) for n in self.first_seen)
        nonzero = [v for v in first if v]
        if (total and (not nonzero or min(nonzero) != 1)) or (
            not total and (self.new_open or self.old_open)
        ):
            raise ValueError("cohort history must include its first assignment")
        if (self.new_open or self.old_open) and not first[self.last]:
            raise ValueError("open cohort dose must have been assigned")
        if len(set(nonzero)) != len(nonzero):
            raise ValueError("first visit indices must be unique")
        for b in (self.new_open, self.old_open):
            if not isinstance(b, (bool, np.bool_)):
                raise ValueError("cohort flags must be boolean")
        for n in (self.new_start, self.old_start):
            _integer(n, "cohort_start", 0, total)
        object.__setattr__(self, "first_seen", first)

    def advance(
        self, dose: int, *, new_size: int, old_size: int, profile: str = "safe"
    ) -> "UAROETCohortState":
        _profile(profile)
        D = len(self.first_seen)
        chosen = _integer(dose, "dose", 0, D - 1)
        ns = _integer(new_size, "new_size", 1, 500)
        os = _integer(old_size, "old_size", 1, 500)
        first = list(self.first_seen)
        n = self.total + 1
        new_open = self.new_open
        old_open = self.old_open
        new_start = self.new_start
        old_start = self.old_start
        if profile == "safe" and chosen != self.last:
            new_open = old_open = False
        if first[chosen] == 0:
            first[chosen] = n
            new_start = n
            if ns > 1:
                new_open = True
        elif not new_open and not old_open and os > 1:
            old_open = True
            old_start = n
        if new_open:
            if n >= new_start + ns - 1:
                new_open = False
        elif old_open and n >= old_start + (ns if profile == "legacy_1_8" else os) - 1:
            old_open = False
        return UAROETCohortState(tuple(first), n, chosen, new_open, old_open, new_start, old_start)

    def choose(
        self,
        status: tuple[int, ...],
        mean_utility: ArrayLike,
        *,
        first_size: int,
        starting_dose: int = 0,
        profile: str = "safe",
    ) -> int | None:
        """Plain greedy cohort selection; source-compatible bypass requires legacy."""
        _profile(profile)
        D = len(self.first_seen)
        start = _integer(starting_dose, "starting_dose", 0, D - 1)
        first = _integer(first_size, "first_size", 1, 500)
        if len(status) != D or any(
            isinstance(v, (bool, np.bool_)) or v not in (1, -1, -2, -3, -4, -5) for v in status
        ):
            raise ValueError("status and cohort dimensions differ")
        utility = _vector(mean_utility, "mean_utility", D)
        if self.total < first and (profile == "legacy_1_8" or status[start] == 1):
            return start
        eligible = [i for i, s in enumerate(status) if s == 1]
        if not eligible:
            return None
        if (self.new_open or self.old_open) and status[self.last] == 1:
            return self.last
        return max(eligible, key=lambda i: (utility[i], -i))


def uaroet_replay_cohorts(
    doses: tuple[int, ...], *, dose_count: int, new_size: int, old_size: int, profile: str = "safe"
) -> UAROETCohortState:
    """Legacy replay bases cohort type on the last record's first-visit status."""
    _profile(profile)
    D = _integer(dose_count, "dose_count", 1, 5)
    ns = _integer(new_size, "new_size", 1, 500)
    os = _integer(old_size, "old_size", 1, 500)
    if len(doses) > 500:
        raise ValueError("history exceeds500 patients")
    validated = tuple(_integer(d, "dose", 0, D - 1) for d in doses)
    state = UAROETCohortState((0,) * D)
    if profile == "safe":
        for d in validated:
            state = state.advance(d, new_size=ns, old_size=os, profile=profile)
        return state
    seen = [0] * D
    trailing = 0
    last = -1
    new = False
    for n, d in enumerate(validated, 1):
        new = seen[d] == 0
        if new:
            seen[d] = n
        trailing = trailing + 1 if d == last else 1
        last = d
    size = ns if new else os
    incomplete = bool(validated) and size > 1 and trailing % size != 0
    # Native fresh-state replay never initializes either cohort-start counter.
    return UAROETCohortState(
        tuple(seen),
        len(validated),
        last if incomplete else 0,
        incomplete and new,
        incomplete and not new,
        0,
        0,
    )
