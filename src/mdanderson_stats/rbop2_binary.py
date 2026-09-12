"""Binary efficacy/toxicity evaluation for two-arm BOP2 designs."""

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_difference


def _counts(value: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(value)
    if (
        x.ndim != 1
        or not x.size
        or np.any(~np.isfinite(x))
        or np.any(x != np.floor(x))
        or np.any((x < 1) | (x > 200))
    ):
        raise ValueError(f"{name} must contain integer sizes in [1,200]")
    return x.astype(np.int64)


def _prior(value: ArrayLike) -> np.ndarray:
    x = finite(value, "prior")
    if x.shape != (2, 2) or np.any(x <= 0):
        raise ValueError("prior must be positive shape (2,2): experimental, control")
    return x


@dataclass(frozen=True)
class Rbop2BinaryMonitor:
    n_experimental: int
    n_control: int
    experimental_events: int
    control_events: int
    probability: float
    probability_error: float
    futile: bool
    superior: bool
    decision: str


@dataclass(frozen=True)
class Rbop2BinaryLookTable:
    look: int
    n_experimental: int
    n_control: int
    lower_cutoff: float
    upper_cutoff: float
    futile_events: np.ndarray
    superior_events: np.ndarray


@dataclass(frozen=True)
class Rbop2BinaryBoundaryTable:
    endpoint: str
    looks: tuple[Rbop2BinaryLookTable, ...]


@dataclass(frozen=True)
class Rbop2BinaryOperatingCharacteristics:
    experimental_rate: np.ndarray
    control_rate: np.ndarray
    stop_futility: np.ndarray
    stop_superiority: np.ndarray
    sample_size_probability: np.ndarray
    final_positive: np.ndarray
    final_negative: np.ndarray
    overall_positive: np.ndarray
    overall_negative: np.ndarray
    probability_no_early_stop: np.ndarray
    expected_experimental_sample_size: np.ndarray
    expected_control_sample_size: np.ndarray
    expected_total_sample_size: np.ndarray


def _exact_order(a: int, b: int, c: int, d: int) -> Fraction:
    norm = Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))
    return sum(
        (
            Fraction(comb(c + d - 1, k))
            * Fraction(
                factorial(a + k - 1) * factorial(b + c + d - k - 2), factorial(a + b + c + d - 2)
            )
            / norm
            for k in range(c, c + d)
        ),
        Fraction(0),
    )


def _probability(
    endpoint: str, prior: np.ndarray, e: int, c: int, ne: int, nc: int, margin: float, tol: float
) -> tuple[float, float]:
    ep = BetaBinomialPosterior(prior[0, 0] + e, prior[0, 1] + (ne - e))
    cp = BetaBinomialPosterior(prior[1, 0] + c, prior[1, 1] + (nc - c))
    result = (
        compare_beta_difference(cp, ep, margin, absolute_tolerance=tol)
        if endpoint == "efficacy"
        else compare_beta_difference(ep, cp, margin, absolute_tolerance=tol)
    )
    return float(result.above_margin), float(result.absolute_error)


def _pmf(n: int, k: int, p: float) -> float:
    if p == 0:
        return float(k == 0)
    if p == 1:
        return float(k == n)
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def _kernel(old: int, new: int, p: float) -> np.ndarray:
    out = np.zeros((new + 1, old + 1))
    for j in range(old + 1):
        for add in range(new - old + 1):
            out[j + add, j] = _pmf(new - old, add, p)
    return out


@dataclass(frozen=True)
class Rbop2BinaryDesign:
    looks: np.ndarray
    prior: np.ndarray
    endpoint: str
    margin: float
    lower_cutoffs: np.ndarray
    upper_cutoffs: np.ndarray
    absolute_tolerance: float = 1e-9

    def _decision(
        self, e: int, c: int, ne: int, nc: int, i: int
    ) -> tuple[float, float, bool, bool]:
        p, error = _probability(
            self.endpoint, self.prior, e, c, ne, nc, self.margin, self.absolute_tolerance
        )
        lower, upper = float(self.lower_cutoffs[i]), float(self.upper_cutoffs[i])
        near = error > 0 and (abs(p - lower) <= error or abs(p - upper) <= error)
        if near and self.margin == 0:
            shapes = (*self.prior[0] + (e, ne - e), *self.prior[1] + (c, nc - c))
            if all(float(x).is_integer() and x <= 1000 for x in shapes):
                exact = _exact_order(*map(int, shapes[:2]), *map(int, shapes[2:]))
                if self.endpoint == "toxicity":
                    exact = 1 - exact
                return (
                    float(exact),
                    0.0,
                    exact < Fraction(str(lower)),
                    exact >= Fraction(str(upper)),
                )
            raise ArithmeticError("posterior probability is too close to cutoff to resolve")
        if near:
            tighter = max(1e-12, self.absolute_tolerance / 10)
            if tighter < self.absolute_tolerance:
                p, error = _probability(
                    self.endpoint, self.prior, e, c, ne, nc, self.margin, tighter
                )
                near = error > 0 and (abs(p - lower) <= error or abs(p - upper) <= error)
            if near:
                raise ArithmeticError("posterior probability is too close to cutoff to resolve")
        return p, error, p < lower, p >= upper

    def monitor(
        self, experimental_events: int, control_events: int, sample_size: ArrayLike
    ) -> Rbop2BinaryMonitor:
        sizes = np.asarray(sample_size)
        if sizes.shape != (2,) or np.any(~np.isfinite(sizes)) or np.any(sizes != np.floor(sizes)):
            raise ValueError("sample_size must be [n_experimental,n_control]")
        ne, nc = (int(x) for x in sizes)
        if ne < 1 or nc < 1 or np.any(np.array([ne, nc]) > self.looks[-1]):
            raise ValueError("sample_size cannot exceed final arm sizes")
        e, c = (
            scalar(experimental_events, "experimental_events"),
            scalar(control_events, "control_events"),
        )
        if int(e) != e or int(c) != c or not 0 <= e <= ne or not 0 <= c <= nc:
            raise ValueError("event counts must be integers within arm sizes")
        p, error = _probability(
            self.endpoint, self.prior, int(e), int(c), ne, nc, self.margin, self.absolute_tolerance
        )
        hit = np.flatnonzero(np.all(self.looks == [ne, nc], axis=1))
        if not len(hit):
            return Rbop2BinaryMonitor(ne, nc, int(e), int(c), p, error, False, False, "continue")
        i = int(hit[0])
        p, error, futile, superior = self._decision(int(e), int(c), ne, nc, i)
        return Rbop2BinaryMonitor(
            ne,
            nc,
            int(e),
            int(c),
            p,
            error,
            futile,
            superior,
            "futility" if futile else "superiority" if superior else "continue",
        )

    def _tables(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (ne, nc) in enumerate(self.looks):
            f: np.ndarray = np.zeros((int(ne) + 1, int(nc) + 1), bool)
            s = np.zeros_like(f)
            for e in range(int(ne) + 1):
                for c in range(int(nc) + 1):
                    _, _, f[e, c], s[e, c] = self._decision(e, c, int(ne), int(nc), i)
            result.append((f, s))
        return tuple(result)

    def boundary_table(self) -> Rbop2BinaryBoundaryTable:
        tables = self._tables()
        rows = []
        for i, (ne, nc) in enumerate(self.looks):
            f, s = tables[i]
            rows.append(
                Rbop2BinaryLookTable(
                    i + 1,
                    int(ne),
                    int(nc),
                    float(self.lower_cutoffs[i]),
                    float(self.upper_cutoffs[i]),
                    _freeze(np.argwhere(f)),
                    _freeze(np.argwhere(s)),
                )
            )
        return Rbop2BinaryBoundaryTable(self.endpoint, tuple(rows))

    def operating_characteristics(
        self, experimental_rate: ArrayLike, control_rate: ArrayLike
    ) -> Rbop2BinaryOperatingCharacteristics:
        er, cr = np.broadcast_arrays(
            finite(experimental_rate, "experimental_rate"), finite(control_rate, "control_rate")
        )
        if np.any((er < 0) | (er > 1) | (cr < 0) | (cr > 1)):
            raise ValueError("rates must be in [0,1]")
        ne, nc = map(int, self.looks[-1])
        work = 0
        old_e = old_c = 0
        for look_e, look_c in self.looks:
            de, dc = int(look_e) - old_e, int(look_c) - old_c
            work += (int(look_e) + 1) * (old_e + 1) * (de + 1) + (int(look_e) + 1) * (
                int(look_c) + 1
            ) * (dc + 1)
            old_e, old_c = int(look_e), int(look_c)
        if er.size * work > 5_000_000:
            raise ValueError("operating-characteristic state space exceeds 5000000")
        tables, shape = self._tables(), er.shape
        values: list[list[object]] = [[] for _ in range(11)]
        for pe, pc in zip(er.flat, cr.flat, strict=True):
            for target, item in zip(
                values, self._oc_one(float(pe), float(pc), tables), strict=True
            ):
                target.append(item)
        sf = _freeze(np.asarray(values[0]).reshape(shape + (len(self.looks),)))
        ss = _freeze(np.asarray(values[1]).reshape(shape + (len(self.looks),)))
        sample_mass = _freeze(np.asarray(values[2]).reshape(shape + (len(self.looks),)))
        rest = [_freeze(np.asarray(x).reshape(shape)) for x in values[3:]]
        return Rbop2BinaryOperatingCharacteristics(
            _freeze(er), _freeze(cr), sf, ss, sample_mass, *rest
        )

    def _oc_one(
        self, pe: float, pc: float, tables: tuple[tuple[np.ndarray, np.ndarray], ...]
    ) -> tuple[object, ...]:
        alive = np.ones((1, 1))
        olde = oldc = 0
        sf = np.zeros(len(self.looks))
        ss = np.zeros(len(self.looks))
        ee = ec = reached = 0.0
        for i, (ne0, nc0) in enumerate(self.looks):
            ne, nc = int(ne0), int(nc0)
            current = _kernel(olde, ne, pe) @ alive @ _kernel(oldc, nc, pc).T
            if i == len(self.looks) - 1:
                reached = float(current.sum())
            f, s = tables[i]
            sf[i], ss[i] = current[f].sum(), current[s].sum()
            stopped = sf[i] + ss[i]
            if i == len(self.looks) - 1:
                ee += ne * reached
                ec += nc * reached
            else:
                ee += ne * stopped
                ec += nc * stopped
            alive = current.copy()
            alive[f | s] = 0
            olde, oldc = ne, nc
        overall = float(ss.sum())
        return (
            sf,
            ss,
            sf + ss,
            float(ss[-1]),
            float(sf[-1]),
            overall,
            float(sf.sum()),
            reached,
            ee,
            ec,
            ee + ec,
        )


def rbop2_binary_design(
    looks: ArrayLike,
    *,
    prior: ArrayLike,
    endpoint: str,
    margin: float,
    lower_cutoffs: ArrayLike,
    upper_cutoffs: ArrayLike,
    absolute_tolerance: float = 1e-9,
) -> Rbop2BinaryDesign:
    rows = np.asarray(looks)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("looks must be Kx2 [experimental,control]")
    ne, nc = (
        _counts(rows[:, 0], "experimental look sizes"),
        _counts(rows[:, 1], "control look sizes"),
    )
    if np.any(np.diff(ne) <= 0) or np.any(np.diff(nc) <= 0):
        raise ValueError("look sizes must be strictly increasing")
    lower, upper = finite(lower_cutoffs, "lower_cutoffs"), finite(upper_cutoffs, "upper_cutoffs")
    if (
        lower.shape != (len(ne),)
        or upper.shape != lower.shape
        or np.any((lower < 0) | (upper > 1) | (lower > upper))
        or lower[-1] != upper[-1]
    ):
        raise ValueError("invalid cutoffs")
    endpoint = str(endpoint).lower()
    delta, tol = scalar(margin, "margin"), scalar(absolute_tolerance, "absolute_tolerance")
    if endpoint not in ("efficacy", "toxicity") or not -1 <= delta <= 1 or not 1e-12 <= tol <= 1e-3:
        raise ValueError("invalid endpoint, margin, or tolerance")
    if np.sum((ne + 1) * (nc + 1)) > 20_000:
        raise ValueError("posterior boundary state space exceeds 20000")
    return Rbop2BinaryDesign(
        _freeze(np.column_stack((ne, nc))),
        _freeze(_prior(prior)),
        endpoint,
        delta,
        _freeze(lower),
        _freeze(upper),
        tol,
    )
