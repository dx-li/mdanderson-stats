"""U2OET ordinal simulation scenarios with a Gaussian copula."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import _real, u2oet_standardize
from .u2oet_decision import _integer


def _marginal_grid(values: ArrayLike, name: str) -> FloatArray:
    p = _real(values, name)
    if (
        p.ndim != 3
        or any(not 2 <= n <= 5 for n in p.shape[:2])
        or not 2 <= p.shape[-1] <= 4
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(f"{name} must be a dose-by-dose-by-category probability grid")
    total = p.sum(axis=-1, keepdims=True)
    if np.any(np.abs(total - 1) > 1e-12):
        raise ValueError(f"{name} probabilities must sum to one at every dose pair")
    return p / total


def _boundaries(p: FloatArray) -> FloatArray:
    lower = np.cumsum(p)[:-1]
    upper = np.cumsum(p[::-1])[::-1][1:]
    z = np.where(lower <= 0.5, ndtri(lower), -ndtri(upper))
    return np.r_[-np.inf, z, np.inf]


def _rectangle(x0: float, x1: float, y0: float, y1: float, rho: float) -> tuple[float, float]:
    sd = np.sqrt((1 - rho) * (1 + rho))

    def integrand(x: float) -> float:
        lo, hi = (y0 - rho * x) / sd, (y1 - rho * x) / sd
        probability = ndtr(-lo) - ndtr(-hi) if lo >= 0 else ndtr(hi) - ndtr(lo)
        return float(np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi) * probability)

    # Split at sharp conditional transitions, including near-singular rho.
    # Gaussian-density anchors also keep infinite-interval quadrature localized.
    points = [-8.0, 0.0, 8.0]
    for y in (y0, y1) if abs(rho) > 1e-8 else ():
        if np.isfinite(y):
            center = y / rho
            width = 8 * sd / abs(rho)
            points.extend(v for v in (center - width, center, center + width) if -10 < v < 10)
    boundaries = [x0, *sorted(set(v for v in points if x0 < v < x1)), x1]
    value, error = 0.0, 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        result = quad(
            integrand,
            lo,
            hi,
            epsabs=2e-13 / len(boundaries),
            epsrel=5e-13,
            limit=200,
            full_output=1,
        )
        if len(result) != 3:
            raise ArithmeticError(f"Gaussian rectangle integration failed: {result[3]}")
        value += result[0]
        error += result[1]
    if not np.isfinite(value) or value < 0 or error > 2e-12:
        raise ArithmeticError("Gaussian rectangle did not meet the absolute error tolerance")
    return value, error


@dataclass(frozen=True)
class U2OETScenario:
    """Gaussian-copula correlation is latent-normal correlation, not ordinal Pearson r."""

    efficacy: FloatArray
    toxicity: FloatArray
    association: float
    joint: FloatArray
    quadrature_error: FloatArray


def u2oet_scenario(
    efficacy: ArrayLike, toxicity: ArrayLike, *, association: float = 0.1
) -> U2OETScenario:
    """Construct scenario probabilities from ordinal marginals; no fitted FGM model.

    Uses deterministic conditional-normal quadrature with absolute tolerances.
    Independence and perfectly correlated limiting copulas are evaluated directly.
    Tiny cells may round to zero; this is not a log-tail probability API.
    """
    e, t = _marginal_grid(efficacy, "efficacy"), _marginal_grid(toxicity, "toxicity")
    rho_array = _real(association, "association")
    if e.shape[:2] != t.shape[:2] or rho_array.ndim or abs(rho_array) > 1:
        raise ValueError("marginal dose grids must match and association must lie in [-1,1]")
    rho = float(rho_array)
    joint = np.empty((*e.shape, t.shape[-1]))
    errors = np.zeros_like(joint)
    for index in np.ndindex(e.shape[:2]):
        ep, tp = e[index], t[index]
        if rho == 0:
            joint[index] = ep[:, None] * tp
            continue
        if abs(rho) == 1:
            ec, tc = np.r_[0, np.cumsum(ep)], np.r_[0, np.cumsum(tp)]
            low, high = (tc[:-1], tc[1:]) if rho > 0 else (1 - tc[1:], 1 - tc[:-1])
            joint[index] = np.maximum(
                0, np.minimum(ec[1:, None], high) - np.maximum(ec[:-1, None], low)
            )
            continue
        eb, tb = _boundaries(ep), _boundaries(tp)
        for a, b in np.ndindex((ep.size, tp.size)):
            if ep[a] == 0 or tp[b] == 0:
                joint[*index, a, b] = 0
            else:
                joint[*index, a, b], errors[*index, a, b] = _rectangle(
                    float(eb[a]), float(eb[a + 1]), float(tb[b]), float(tb[b + 1]), rho
                )
    if (
        np.max(np.abs(joint.sum(axis=-1) - e)) > 1e-11
        or np.max(np.abs(joint.sum(axis=-2) - t)) > 1e-11
    ):
        raise ArithmeticError("Gaussian copula failed marginal preservation")
    return U2OETScenario(_freeze(e), _freeze(t), rho, _freeze(joint), _freeze(errors))


def _rows(path: str | Path) -> list[list[float]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(v) for v in line.split()]
        except ValueError as exc:
            raise ValueError(f"invalid numeric input on line {number}") from exc
        if not np.all(np.isfinite(row)):
            raise ValueError(f"nonfinite value on line {number}")
        rows.append(row)
    return rows


def read_u2oet_scenario(
    path: str | Path, *, dose_counts: tuple[int, int], efficacy_levels: int, toxicity_levels: int
) -> U2OETScenario:
    """Read guide section 2.1; native dose indices are one-based.

    Missing association header means independence. An invalid single-number
    header is rejected rather than silently interpreted as independence.
    """
    if len(dose_counts) != 2:
        raise ValueError("dose_counts must have two entries")
    shape = tuple(_integer(v, "dose count", 2, 5) for v in dose_counts)
    le = _integer(efficacy_levels, "efficacy_levels", 2, 4)
    lt = _integer(toxicity_levels, "toxicity_levels", 2, 4)
    rows = _rows(path)
    rho = 0.0
    if rows and len(rows[0]) == 1:
        rho = rows.pop(0)[0]
        if not -1 < rho < 1:
            raise ValueError("native association header must lie strictly between -1 and 1")
    e, t = np.empty((*shape, le)), np.empty((*shape, lt))
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if len(row) != 2 + le + lt:
            raise ValueError("scenario row width must match specified outcome levels")
        index = (
            _integer(row[0], "dose index", 1, shape[0]) - 1,
            _integer(row[1], "dose index", 1, shape[1]) - 1,
        )
        if index in seen:
            raise ValueError("duplicate scenario dose pair")
        seen.add(index)
        e[index], t[index] = row[2 : 2 + le], row[2 + le :]
    if len(seen) != int(np.prod(shape)):
        raise ValueError("scenario must specify every dose pair exactly once")
    return u2oet_scenario(e, t, association=rho)


def read_u2oet_doses(path: str | Path) -> tuple[FloatArray, FloatArray]:
    """Read the two positive ascending raw-dose lines from the native format."""
    rows = _rows(path)
    if len(rows) != 2:
        raise ValueError("raw-dose file must contain two nonempty lines")
    for row in rows:
        u2oet_standardize(row)
    return _freeze(rows[0]), _freeze(rows[1])


def read_u2oet_utility(
    path: str | Path, *, efficacy_levels: int, toxicity_levels: int
) -> FloatArray:
    """Read zero-based efficacy/toxicity/value rows into an efficacy-by-toxicity matrix."""
    le = _integer(efficacy_levels, "efficacy_levels", 2, 4)
    lt = _integer(toxicity_levels, "toxicity_levels", 2, 4)
    result = np.empty((le, lt))
    seen: set[tuple[int, int]] = set()
    for row in _rows(path):
        if len(row) != 3 or row[2] < 0:
            raise ValueError(
                "utility rows need efficacy index, toxicity index and nonnegative value"
            )
        index = (
            _integer(row[0], "efficacy index", 0, le - 1),
            _integer(row[1], "toxicity index", 0, lt - 1),
        )
        if index in seen:
            raise ValueError("duplicate utility cell")
        seen.add(index)
        result[index] = row[2]
    if len(seen) != le * lt:
        raise ValueError("utility file must specify every outcome pair exactly once")
    if np.any(np.diff(result, axis=0) < 0) or np.any(np.diff(result, axis=1) > 0):
        raise ValueError("utility must not decrease with efficacy or increase with toxicity")
    return _freeze(result)
