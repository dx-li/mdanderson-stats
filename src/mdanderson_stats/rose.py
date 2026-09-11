"""Randomized optimal selection (ROSE) and exact binomial design calculations."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import binom

from ._validation import scalar


def _integer(x: int, name: str, minimum: int = 1) -> int:
    value = scalar(x, name)
    if value != np.floor(value) or not minimum <= value <= 1000:
        raise ValueError(f"{name} must be an integer in [{minimum},1000]")
    return int(value)


def _probability(x: float, name: str) -> float:
    value = scalar(x, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return value


def _cut(boundary: float, denominator: int) -> int:
    # Interpret the supplied decimal boundary exactly at discrete response ties.
    value = Fraction(str(boundary)) * denominator
    return value.numerator // value.denominator


@dataclass(frozen=True)
class RoseDesign:
    n_low: int
    n_high: int
    boundary: float
    interim_low: int = 0
    interim_high: int = 0
    interim_boundary: float | None = None
    method: str = "specified"

    def __post_init__(self) -> None:
        for name, minimum in (("n_low", 1), ("n_high", 1), ("interim_low", 0), ("interim_high", 0)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        object.__setattr__(self, "boundary", _probability(self.boundary, "boundary"))
        if self.interim_boundary is None:
            if self.interim_low or self.interim_high:
                raise ValueError("interim sample sizes require an interim boundary")
        else:
            boundary = _probability(self.interim_boundary, "interim_boundary")
            if not (0 < self.interim_low < self.n_low and 0 < self.interim_high < self.n_high):
                raise ValueError("each arm must enroll patients in both stages")
            if boundary < self.boundary:
                raise ValueError("interim boundary must be at least the final boundary")
            object.__setattr__(self, "interim_boundary", boundary)


@dataclass(frozen=True)
class RoseOperatingCharacteristics:
    select_low: float
    select_high: float
    early_high: float
    mean_low: float
    mean_high: float


def _distribution(
    n_low: int, n_high: int, p_low: float, p_high: float, low_weight: int, high_weight: int
) -> tuple[np.ndarray, int]:
    low, high = np.arange(n_low + 1), np.arange(n_high + 1)
    index = high[:, None] * high_weight - low[None, :] * low_weight + n_low * low_weight
    mass = binom.pmf(high, n_high, p_high)[:, None] * binom.pmf(low, n_low, p_low)[None, :]
    return np.bincount(
        index.ravel(), weights=mass.ravel(), minlength=n_low * low_weight + n_high * high_weight + 1
    ), -n_low * low_weight


def _tails(mass: np.ndarray, offset: int, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = cuts - offset
    cdf = np.r_[0.0, np.cumsum(mass)]
    sf = np.r_[np.cumsum(mass[::-1])[::-1], 0.0]
    pos = np.clip(index + 1, 0, len(mass))
    return cdf[pos], sf[pos]


def _evaluator(
    nl: int, nh: int, il: int, ih: int, interim: float | None, pl: float, ph: float
) -> Callable[[float], RoseOperatingCharacteristics]:
    common = gcd(nl, nh)
    wl, wh, denominator = nh // common, nl // common, nl * nh // common
    if interim is None:
        mass, offset = _distribution(nl, nh, pl, ph, wl, wh)

        def evaluate(boundary: float) -> RoseOperatingCharacteristics:
            low, high = _tails(mass, offset, np.array(_cut(boundary, denominator)))
            return RoseOperatingCharacteristics(float(low), float(high), 0.0, float(nl), float(nh))

        return evaluate
    lows, highs = np.arange(il + 1), np.arange(ih + 1)
    joint = binom.pmf(highs, ih, ph)[:, None] * binom.pmf(lows, il, pl)[None, :]
    early = highs[:, None] * il - lows[None, :] * ih > _cut(interim, il * ih)
    early_probability = float(joint[early].sum())
    continuation = joint[~early]
    accumulated = (highs[:, None] * wh - lows[None, :] * wl)[~early]
    mass, offset = _distribution(nl - il, nh - ih, pl, ph, wl, wh)

    def evaluate_two(boundary: float) -> RoseOperatingCharacteristics:
        low, high = _tails(mass, offset, _cut(boundary, denominator) - accumulated)
        return RoseOperatingCharacteristics(
            float(continuation @ low),
            early_probability + float(continuation @ high),
            early_probability,
            il + (nl - il) * (1 - early_probability),
            ih + (nh - ih) * (1 - early_probability),
        )

    return evaluate_two


def rose_operating_characteristics(
    design: RoseDesign, p_low: float, p_high: float
) -> RoseOperatingCharacteristics:
    """Exact probabilities and expected enrollment, including unequal arm sizes.

    High is selected only for a strictly larger observed rate difference than the
    boundary. Two-stage designs stop early only for high; all other trials finish.
    """
    if not isinstance(design, RoseDesign):
        raise TypeError("design must be a RoseDesign")
    pl, ph = _probability(p_low, "p_low"), _probability(p_high, "p_high")
    return _evaluator(
        design.n_low,
        design.n_high,
        design.interim_low,
        design.interim_high,
        design.interim_boundary,
        pl,
        ph,
    )(design.boundary)


def rose_select(
    design: RoseDesign, responses_low: int, responses_high: int, *, interim: bool = False
) -> str:
    """Return low/high at final, or high/continue at the planned interim look."""
    if not isinstance(design, RoseDesign) or not isinstance(interim, (bool, np.bool_)):
        raise TypeError("require RoseDesign and boolean interim")
    low, high = (
        _integer(responses_low, "responses_low", 0),
        _integer(responses_high, "responses_high", 0),
    )
    nl, nh, boundary = design.n_low, design.n_high, design.boundary
    if interim:
        if design.interim_boundary is None:
            raise ValueError("design has no interim analysis")
        nl, nh, boundary = design.interim_low, design.interim_high, design.interim_boundary
    if low > nl or high > nh:
        raise ValueError("responses cannot exceed planned enrollment at this look")
    selected = high * nl - low * nh > _cut(boundary, nl * nh)
    return "high" if selected else "continue" if interim else "low"


def _bvn(a: float, b: float, rho: float) -> float:
    sd = np.sqrt(1 - rho * rho)
    result = quad(
        lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * ndtr((b - rho * x) / sd),
        -np.inf,
        a,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
        full_output=1,
    )
    value, error = result[:2]
    if len(result) != 3 or error > 1e-9 or not 0 <= value <= 1:
        raise ArithmeticError("bivariate normal integration failed")
    return float(value)


def _grid(end: float, step: float) -> list[float]:
    spacing, limit = Fraction(str(step)), Fraction(str(end))
    return [float(i * spacing) for i in range(int(limit // spacing) + 1)]


def rose_design(
    p_high: float,
    margin: float,
    *,
    pcs_low: float = 0.65,
    pcs_high: float = 0.65,
    ratio: float = 1.0,
    interim_fraction: float | None = None,
    method: str = "normal",
    step: float = 0.002,
    max_per_arm: int = 200,
) -> RoseDesign:
    """Search one-/two-stage ROSE with normal approximation or exact binomial PCS.

    ratio is high:low. Normal two-stage uses the paper's unrounded information
    fractions for design, then rounds enrollment upward. Exact designs evaluate
    rounded enrollment directly and use O'Brien-Fleming interim error spending.
    Exact searches return the first feasible final boundary on the specified grid.
    """
    ph, delta = scalar(p_high, "p_high"), scalar(margin, "margin")
    al, ah, c = scalar(pcs_low, "pcs_low"), scalar(pcs_high, "pcs_high"), scalar(ratio, "ratio")
    maximum = _integer(max_per_arm, "max_per_arm")
    if not (0 < ph < 1 and 0 < delta <= ph and 0.5 < al < 1 and 0.5 < ah < 1 and c > 0):
        raise ValueError("require 0 < margin <= p_high < 1, PCS in (.5,1), and ratio > 0")
    if method not in ("normal", "exact"):
        raise ValueError("method must be normal or exact")
    omega = None if interim_fraction is None else scalar(interim_fraction, "interim_fraction")
    if omega is not None and not 0 < omega < 1:
        raise ValueError("interim_fraction must be in (0,1)")
    sl = np.sqrt(ph * (1 - ph) * (1 + 1 / c))
    sh = np.sqrt((ph - delta) * (1 - ph + delta) + ph * (1 - ph) / c)
    if method == "normal" and omega is None:
        a, b = sl * ndtri(al), sh * ndtri(ah)
        continuous = ((a + b) / delta) ** 2
        if not np.isfinite(continuous) or continuous > maximum:
            raise ValueError("no normal design within max_per_arm")
        nl = max(1, ceil(continuous))
        nh = ceil(c * nl)
        if nh > maximum:
            raise ValueError("no normal design within max_per_arm")
        return RoseDesign(nl, nh, float(delta * a / (a + b)), method="normal")
    if omega is not None:
        spent = float(2 * ndtr(ndtri((1 - al) / 2) / np.sqrt(omega)))
        if method == "normal":
            if spent == 0:
                raise ArithmeticError("interim spending underflows; choose a later interim")
            first_z = float(-ndtri(spent))
            final_z = brentq(lambda z: _bvn(first_z, z, np.sqrt(omega)) - al, -12, 12, xtol=1e-11)
    if method == "exact":
        spacing = scalar(step, "step")
        if not 0.0001 <= spacing <= 1:
            raise ValueError("step must be in [.0001,1]")
        finals = _grid(delta, spacing)
        interims = _grid(1.0, spacing)
        if interims[-1] != 1.0:
            interims.append(1.0)
    for nl in range(1, maximum + 1):
        nh_real = c * nl
        if not np.isfinite(nh_real) or nh_real > maximum:
            break
        nh = ceil(nh_real)
        il, ih = (0, 0) if omega is None else (ceil(omega * nl), ceil(omega * nh))
        if omega is not None and (il >= nl or ih >= nh):
            continue
        if method == "normal":
            assert omega is not None
            # S5/S6 calculate power with continuous stage fractions, before rounding.
            z1 = (first_z * sl - delta * np.sqrt(omega * nl)) / sh
            z2 = (final_z * sl - delta * np.sqrt(nl)) / sh
            power = 1 - _bvn(z1, z2, np.sqrt(omega))
            if power < ah:
                continue
            first_normal = float(first_z * sl / np.sqrt(il))
            final = float(final_z * sl / np.sqrt(nl))
            if 0 <= final <= first_normal <= 1:
                return RoseDesign(nl, nh, final, il, ih, first_normal, "normal")
            continue
        first: float | None = None
        if omega is not None:
            mass, offset = _distribution(il, ih, ph, ph, ih, il)
            cuts = np.array([_cut(value, il * ih) for value in interims])
            _, upper = _tails(mass, offset, cuts)
            feasible = np.flatnonzero(upper <= spent)
            if not len(feasible):
                continue
            first = interims[int(feasible[0])]
        evaluate_low = _evaluator(nl, nh, il, ih, first, ph, ph)
        evaluate_high = _evaluator(nl, nh, il, ih, first, ph - delta, ph)
        # Low-selection probability increases with the final boundary. Locate
        # its first feasible grid point by bisection rather than scanning all points.
        candidates = finals if first is None else [x for x in finals if x <= first]
        left, right = 0, len(candidates)
        while left < right:
            middle = (left + right) // 2
            if evaluate_low(candidates[middle]).select_low >= al:
                right = middle
            else:
                left = middle + 1
        if left < len(candidates) and evaluate_high(candidates[left]).select_high >= ah:
            return RoseDesign(nl, nh, candidates[left], il, ih, first, "exact")
    raise ValueError("no feasible design within max_per_arm and the specified boundary grid")


@dataclass(frozen=True)
class RoseSimulation:
    operating_characteristics: RoseOperatingCharacteristics
    trials: int
    select_high_mcse: float
    early_high_mcse: float


def simulate_rose(
    design: RoseDesign,
    p_low: float,
    p_high: float,
    *,
    trials: int = 10000,
    rng: np.random.Generator | int | None = None,
) -> RoseSimulation:
    """Simulate independent binary outcomes with immediate stage-wise assessment.

    Reports empirical operating characteristics and binomial Monte Carlo standard
    errors. The low-selection MCSE equals the high-selection MCSE.
    """
    if not isinstance(design, RoseDesign):
        raise TypeError("design must be a RoseDesign")
    pl, ph = _probability(p_low, "p_low"), _probability(p_high, "p_high")
    size = scalar(trials, "trials")
    if size != np.floor(size) or not 1 <= size <= 10_000_000:
        raise ValueError("trials must be an integer in [1,10000000]")
    size = int(size)
    generator = np.random.default_rng(rng)
    nl, nh = design.n_low, design.n_high
    il, ih = design.interim_low, design.interim_high
    low = generator.binomial(il, pl, size)
    high = generator.binomial(ih, ph, size)
    early = np.zeros(size, dtype=bool)
    if design.interim_boundary is not None:
        early = high * il - low * ih > _cut(design.interim_boundary, il * ih)
    continued = ~early
    n_continue = int(continued.sum())
    low[continued] += generator.binomial(nl - il, pl, n_continue)
    high[continued] += generator.binomial(nh - ih, ph, n_continue)
    selected = early | (continued & (high * nl - low * nh > _cut(design.boundary, nl * nh)))
    selection, pet = float(selected.mean()), float(early.mean())
    return RoseSimulation(
        RoseOperatingCharacteristics(
            1 - selection, selection, pet, il + (nl - il) * (1 - pet), ih + (nh - ih) * (1 - pet)
        ),
        size,
        float(np.sqrt(selection * (1 - selection) / size)),
        float(np.sqrt(pet * (1 - pet) / size)),
    )
