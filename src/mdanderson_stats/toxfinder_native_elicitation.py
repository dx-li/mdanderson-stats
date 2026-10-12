"""Recovered native ToxFinder stochastic physician-prior calibration.

Independent NumPy Gamma streams replace the original RNG. Acceptance reproduces
native spread/cutoff logic; a separate Monte Carlo assessment reports target
quality, which native convergence alone does not establish.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .toxfinder_model import ToxFinderPrior
from .toxfinder_native_information import _frozen

_PAIRS = 30_000


def _stochastic_simplex(
    start: list[float], step: list[float], objective: Callable, budget: int = 20
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
        values = [objective(point) for point in points]
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
        if relative < 1e-6 or values[high] < 2e-6:
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


def _calibrate(probabilities: Callable[[list[float]], tuple[float, float]]) -> tuple:
    """Exact start/precheck/retry ordering, separately testable with native tapes."""
    starts = attempts = 0
    for alpha_variance in (0.1, 0.03, 0.3):
        for beta_variance in (3.0, 1.0, 10.0):
            starts += 1
            point = [math.log(alpha_variance), math.log(beta_variance)]
            step = [0.1, 0.1]
            flags = [False, False]
            for vertex in (point, [point[0] + step[0], point[1]], [point[0], point[1] + step[1]]):
                p = probabilities(vertex)
                for j in range(2):
                    flags[j] |= 1e-6 < p[j] < 0.999999
            if not all(flags):
                continue

            def objective(value: list[float]) -> float:
                p1, p2 = probabilities(value)
                return (p1 - 0.99) ** 2 + (p2 - 0.99) ** 2

            for _ in range(3):
                attempts += 1
                point, converged, iterations = _stochastic_simplex(point, step, objective)
                if converged:
                    return point, starts, attempts, iterations
                step = [v * -0.5 for v in step]
    raise ArithmeticError(
        "native prior calibration did not converge after all nine starts; no fallback"
    )


def _tail_probabilities(
    alpha: FloatArray,
    beta: FloatArray,
    ratios: tuple[float, float],
    thresholds: tuple[float, float],
) -> tuple[float, float]:
    """Use the same paired draws for both strict native tail comparisons."""
    with np.errstate(over="ignore", invalid="ignore"):
        low = alpha * ratios[0] ** beta
        high = alpha * ratios[1] ** beta
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ArithmeticError("native elicitation powers are outside the finite supported domain")
    return float(np.count_nonzero(low / (1 + low) < thresholds[0]) * (1.0 / len(alpha))), float(
        np.count_nonzero(high / (1 + high) > thresholds[1]) * (1.0 / len(alpha))
    )


@dataclass(frozen=True)
class ToxFinderNativeAgentElicitation:
    mean: FloatArray
    variance: FloatArray
    shape: FloatArray
    rate: FloatArray
    starts_considered: int
    optimizer_attempts: int
    final_attempt_iterations: int
    optimizer_probability_evaluations: int
    assessed_tail_probability: FloatArray
    assessed_probability_mcse: FloatArray
    assessed_target_residual: FloatArray
    assessed_objective: float
    assessed_cutoff_met: bool


@dataclass(frozen=True)
class ToxFinderNativeElicitation:
    prior: ToxFinderPrior
    agent1: ToxFinderNativeAgentElicitation
    agent2: ToxFinderNativeAgentElicitation
    gamma_pairs_per_evaluation: int
    rng_profile: str = "independent_numpy_gamma_with_float32_shape"


def _inputs(doses: ArrayLike, target: float, low: float, high: float) -> tuple:
    d = finite(doses, "agent_doses")
    if d.shape != (4,) or np.any(d <= 0) or not d[0] < d[1] < d[2] or not d[3] > d[1]:
        raise ValueError(
            "require four positive doses: negligible < reference < high; fourth > reference"
        )
    for value, name in (
        (target, "target"),
        (low, "negligible_probability"),
        (high, "high_toxicity_probability"),
    ):
        if not 0 < scalar(value, name) < 1:
            raise ValueError(f"{name} must be in (0,1)")
    ratios = d / d[1]
    ma = target / (1 - target)
    mb = math.log(high / ((1 - high) * ma)) / math.log(float(ratios[2]))
    if not math.isfinite(mb) or mb <= 0:
        raise ValueError("native elicitation requires a positive finite beta mean")
    return np.array([ma, mb]), (float(ratios[0]), float(ratios[3]))


def elicit_toxfinder_native_prior(
    agent1_doses: ArrayLike,
    agent2_doses: ArrayLike,
    target: float,
    *,
    interaction_mean: ArrayLike,
    interaction_variance: ArrayLike,
    rng: np.random.Generator,
    negligible_probability: float = 0.05,
    high_toxicity_probability: float = 0.60,
    max_probability_evaluations: int = 8_000,
) -> ToxFinderNativeElicitation:
    """Native calibration contract with independent RNG and explicit quality diagnostics.

    Each probability evaluation uses 30,000 independent Gamma pairs. The original
    float32 narrowing of each Gamma shape is preserved; random-stream identity is
    not claimed. Native convergence can accept poor targets. One additional draw
    batch independently assesses each returned prior without changing the fit.

    The native fixed beta-mean formula differs from the paper's Gamma odds-moment
    equation. Use elicit_toxfinder_prior for that separately validated formulation.
    No MAP parameter box is imposed on these log-variance optimizations.
    """
    first_mean, first_ratios = _inputs(
        agent1_doses, target, negligible_probability, high_toxicity_probability
    )
    second_mean, second_ratios = _inputs(
        agent2_doses, target, negligible_probability, high_toxicity_probability
    )
    means, variances = (
        finite(interaction_mean, "interaction_mean"),
        finite(interaction_variance, "interaction_variance"),
    )
    if (
        means.shape != (2,)
        or variances.shape != (2,)
        or np.any(means <= 0)
        or np.any(variances <= 0)
    ):
        raise ValueError("interaction moments require two positive finite means and variances")
    # Check representability before consuming the caller's random stream.
    ToxFinderPrior([1, 1, 1, 1, *means], [1, 1, 1, 1, *variances])
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if (
        isinstance(max_probability_evaluations, (bool, np.bool_))
        or not isinstance(max_probability_evaluations, (int, np.integer))
        or not 1 <= max_probability_evaluations <= 10_000
    ):
        raise ValueError("max_probability_evaluations must be an integer in [1,10000]")
    evaluations = 0

    def fit(mean: FloatArray, ratios: tuple[float, float]) -> ToxFinderNativeAgentElicitation:
        nonlocal evaluations
        beginning = evaluations

        def probabilities(log_variances: list[float]) -> tuple[float, float]:
            nonlocal evaluations
            evaluations += 1
            if evaluations > max_probability_evaluations:
                raise ValueError("native calibration exceeds probability evaluation budget")
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                variance = np.exp(log_variances)
                shape = mean**2 / variance
                rate = mean / variance
                native_shape = shape.astype(np.float32).astype(np.float64)
            if (
                np.any(variance <= 0)
                or not np.all(np.isfinite(variance))
                or np.any(native_shape <= 0)
                or not np.all(np.isfinite(native_shape))
                or np.any(rate <= 0)
                or not np.all(np.isfinite(rate))
            ):
                raise ArithmeticError(
                    "native Gamma parameters are outside the finite supported domain"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                alpha = rng.standard_gamma(float(native_shape[0]), _PAIRS) / rate[0]
                beta = rng.standard_gamma(float(native_shape[1]), _PAIRS) / rate[1]
            if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
                raise ArithmeticError("native Gamma draws are not representable")
            return _tail_probabilities(alpha, beta, ratios, (negligible_probability, target))

        point, starts, attempts, iterations = _calibrate(probabilities)
        optimization_evaluations = evaluations - beginning
        assessed = np.array(probabilities(point))
        variance = np.exp(point)
        residual = assessed - 0.99
        objective = float(residual @ residual)
        return ToxFinderNativeAgentElicitation(
            _frozen(mean),
            _frozen(variance),
            _frozen(mean**2 / variance),
            _frozen(mean / variance),
            starts,
            attempts,
            iterations,
            optimization_evaluations,
            _frozen(assessed),
            _frozen(np.sqrt(assessed * (1 - assessed) / _PAIRS)),
            _frozen(residual),
            objective,
            objective < 2e-6,
        )

    first = fit(first_mean, first_ratios)
    second = fit(second_mean, second_ratios)
    prior = ToxFinderPrior(
        [*first.mean, *second.mean, *means], [*first.variance, *second.variance, *variances]
    )
    return ToxFinderNativeElicitation(prior, first, second, _PAIRS)
