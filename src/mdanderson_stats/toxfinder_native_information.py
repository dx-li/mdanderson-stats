"""Recovered ToxFinder 1.1.1 MAP and expected observed-information determinant.

Public parameters retain the package's interleaved alpha/beta order. The native
optimizer uses alpha1,alpha2,alpha3,beta1,beta2,beta3 internally. See the native
workflow guide for runtime substitutions in the executable reference evidence.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .toxfinder_model import ToxFinderPrior

_NATIVE = [0, 2, 4, 1, 3, 5]
_PUBLIC = [0, 3, 1, 4, 2, 5]


def _frozen(value: ArrayLike) -> FloatArray:
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _history(doses: ArrayLike, toxicities: ArrayLike, subjects: ArrayLike) -> tuple:
    raw = np.asarray(doses)
    if raw.size == 0:
        raw = raw.reshape((0, 2))
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) > 100:
        raise ValueError("doses must have shape (0..100, 2)")
    d = finite(raw, "doses")
    y, n = count(toxicities, "toxicities"), count(subjects, "subjects")
    if y.shape != (len(d),) or n.shape != y.shape or np.any(y > n) or n.sum() > 10_000:
        raise ValueError("require matching valid counts and at most 10,000 subjects")
    if np.any(d < 0) or np.any(d > 1) or np.any((d == 0).all(axis=1) & (y > 0)):
        raise ValueError("native doses must be in [0,1], with no toxicity at the origin")
    return d, y, n


def _gamma(prior: ToxFinderPrior) -> tuple[FloatArray, FloatArray]:
    if not isinstance(prior, ToxFinderPrior):
        raise TypeError("prior must be a ToxFinderPrior")
    if np.any(prior.variance <= 0):
        raise ValueError("native information requires six nondegenerate gamma priors")
    # Arithmetic matches the native mean/variance conversion.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        shape = (prior.mean**2 / prior.variance)[_NATIVE]
        rate = (prior.mean / prior.variance)[_NATIVE]
    if not np.all(np.isfinite(shape)) or not np.all(np.isfinite(rate)):
        raise ValueError("native gamma shape/rate must be representable")
    return shape, rate


def _simplex(
    start: list[float], step: list[float], objective: Callable, budget: int = 250
) -> tuple[list[float], bool, int]:
    """Independent transcription of the recovered outer-iteration contract."""
    n = len(start)
    points = [start.copy() for _ in range(n + 1)]
    for i in range(n):
        points[i + 1][i] += step[i]
    values = [objective(point) for point in points]
    sums = [sum(row[j] for row in points) for j in range(n)]

    def trial(worst: int, factor: float) -> float:
        f1 = (1 - factor) / n
        f2 = factor - f1
        point = [f1 * sums[j] + f2 * points[worst][j] for j in range(n)]
        value = objective(point)
        if value < values[worst]:
            values[worst] = value
            for j in range(n):
                sums[j] += point[j] - points[worst][j]
            points[worst] = point
        return value

    for iteration in range(budget):
        low = high = 0
        second = 1
        for i in range(1, n + 1):
            if values[i] > values[high]:
                second, high = high, i
            if values[i] < values[low]:
                low = i
            if i != high and values[i] > values[second]:
                second = i
        relative = (
            2 * abs(values[low] - values[high]) / (abs(values[low]) + abs(values[high]) + 1e-10)
        )
        if relative < 1e-6 or values[high] < -1e10:
            return points[low], True, iteration
        value = trial(high, -1)
        if value <= values[low]:
            trial(high, 2)
        elif value >= values[second]:
            saved = values[high]
            if trial(high, 0.5) >= saved:
                for i in range(n + 1):
                    if i != low:
                        points[i] = [(a + b) * 0.5 for a, b in zip(points[i], points[low])]
                        values[i] = objective(points[i])
                sums = [sum(row[j] for row in points) for j in range(n)]
    # The legacy budget exit returns row zero, even when another row is better.
    return points[0], False, budget


@dataclass(frozen=True)
class ToxFinderNativeMAP:
    parameters: FloatArray
    pass_converged: tuple[bool, bool]
    pass_iterations: tuple[int, int]
    evaluations: int
    negative_log_posterior: float


def fit_toxfinder_native_map(
    doses: ArrayLike, toxicities: ArrayLike, subjects: ArrayLike, *, prior: ToxFinderPrior
) -> ToxFinderNativeMAP:
    """Two native bounded simplex passes; convergence status is exposed."""
    d, y, n = _history(doses, toxicities, subjects)
    shape, rate = _gamma(prior)
    start = [float(v) for v in prior.mean[_NATIVE]]
    if any(v < 0.005 or v > 30 for v in start):
        raise ValueError("native MAP prior means must lie in [.005,30]")
    step = [v / 2 for v in start]
    evaluations = 0

    def objective(t: list[float]) -> float:
        nonlocal evaluations
        evaluations += 1
        if any(v < 0.005 or v > 30 for v in t):
            return 1e10
        lp = sum((s - 1) * math.log(v) - r * v for v, s, r in zip(t, shape, rate))
        a1, a2, a3, b1, b2, b3 = t
        for (x1, x2), observed, total in zip(d, y, n):
            q = a1 * x1**b1 + a2 * x2**b2 + a3 * (x1**b1 * x2**b2) ** b3
            if observed:
                if q <= 0:
                    return 1e10
                lp += observed * math.log(q)
            lp -= total * math.log1p(q)
        return -float(lp)

    first, c1, i1 = _simplex(start, step, objective)
    second, c2, i2 = _simplex(first, step, objective)
    value = objective(second)
    return ToxFinderNativeMAP(
        _frozen(np.asarray(second)[_PUBLIC]), (c1, c2), (i1, i2), evaluations, value
    )


def _row_hessian(dose: FloatArray, y: float, n: float, t: FloatArray) -> FloatArray:
    a1, a2, a3, b1, b2, b3 = t
    x, z = dose
    lx = max(math.log(x), -999999.9) if x else -999999.9
    lz = max(math.log(z), -999999.9) if z else -999999.9
    xp, zp = x**b1, z**b2
    terms = (a1 * xp, a2 * zp, a3 * (xp * zp) ** b3)
    q = sum(terms)
    if q == 0:
        if y:
            raise ValueError("positive toxicity has zero representable probability")
        return np.zeros((6, 6))
    gradient = np.zeros(6)
    second = np.zeros((6, 6))
    for amount, alpha, beta, logdose, power in ((terms[0], 0, 3, lx, xp), (terms[1], 1, 4, lz, zp)):
        gradient[alpha] = power
        gradient[beta] = amount * logdose
        second[alpha, beta] = second[beta, alpha] = power * logdose
        second[beta, beta] = amount * logdose**2
    w = np.array([0, 0, 1 / a3, b3 * lx, b3 * lz, b1 * lx + b2 * lz])
    cross = np.zeros((6, 6))
    cross[2, 2] = -1 / a3**2
    cross[3, 5] = cross[5, 3] = lx
    cross[4, 5] = cross[5, 4] = lz
    gradient += terms[2] * w
    second += terms[2] * (np.outer(w, w) + cross)
    return (n / (1 + q) - y / q) * second + (y / q**2 - n / (1 + q) ** 2) * np.outer(
        gradient, gradient
    )


def toxfinder_native_hessian(
    doses: ArrayLike,
    toxicities: ArrayLike,
    subjects: ArrayLike,
    parameters: ArrayLike,
    *,
    prior: ToxFinderPrior,
) -> FloatArray:
    """Observed negative-log-posterior Hessian, in public interleaved order."""
    d, y, n = _history(doses, toxicities, subjects)
    public = finite(parameters, "parameters")
    if public.shape != (6,) or np.any(public < 0.005) or np.any(public > 30):
        raise ValueError("native Hessian requires six parameters in [.005,30]")
    t = public[_NATIVE]
    shape, _ = _gamma(prior)
    h = np.diag((shape - 1) / t**2)
    for dose, observed, total in zip(d, y, n):
        h += _row_hessian(dose, observed, total, t)
    if not np.all(np.isfinite(h)):
        raise ArithmeticError("native observed information is not representable")
    return _frozen(h[np.ix_(_PUBLIC, _PUBLIC)])


@dataclass(frozen=True)
class ToxFinderNativeInformation:
    """Reusable original-history MAP/Hessian for candidate information scores."""

    map_fit: ToxFinderNativeMAP
    observed_hessian: FloatArray

    def score(self, candidate: ArrayLike, predictive_toxicity: float) -> float:
        dose = finite(candidate, "candidate")
        probability = float(predictive_toxicity)
        if dose.shape != (2,) or np.any(dose < 0) or np.any(dose > 1):
            raise ValueError("candidate must contain two doses in [0,1]")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("predictive_toxicity must be in [0,1]")
        t = self.map_fit.parameters[_NATIVE]
        base = self.observed_hessian[np.ix_(_NATIVE, _NATIVE)]
        determinants = []
        for outcome in (0, 1):
            if outcome == 1 and probability == 0:
                determinants.append(0.0)
                continue
            h = base + _row_hessian(dose, outcome, 1, t)
            # Same determinant as reconstructing Q max(D,.1) Q' and Cholesky;
            # avoids unnecessary reconstruction roundoff, not a log determinant.
            determinant = float(np.maximum(np.linalg.eigvalsh(h), 0.1).prod())
            if not math.isfinite(determinant):
                raise ArithmeticError("information determinant is not representable")
            determinants.append(determinant)
        return (1 - probability) * determinants[0] + probability * determinants[1]


def toxfinder_native_information(
    doses: ArrayLike, toxicities: ArrayLike, subjects: ArrayLike, *, prior: ToxFinderPrior
) -> ToxFinderNativeInformation:
    """Fit once to observed data; hypothetical outcomes never change this MAP."""
    fitted = fit_toxfinder_native_map(doses, toxicities, subjects, prior=prior)
    hessian = toxfinder_native_hessian(doses, toxicities, subjects, fitted.parameters, prior=prior)
    return ToxFinderNativeInformation(fitted, hessian)
