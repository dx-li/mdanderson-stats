"""Simulation calibration and sequential analysis for CATBUB designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .catbub import _inputs, _integer, _probabilities, catbub_compare, catbub_simulate_counts


def _spending(fractions: ArrayLike, alpha: float, rho: float) -> FloatArray:
    t = finite(fractions, "fractions")
    alpha, rho = scalar(alpha, "alpha"), scalar(rho, "rho")
    if t.ndim != 1 or len(t) == 0 or np.any(t <= 0) or np.any(t > 1) or np.any(np.diff(t) < 0):
        raise ValueError("fractions must be nondecreasing in (0, 1]")
    if not 0 < alpha < 0.5 or rho <= 0:
        raise ValueError("alpha must be in (0, .5) and rho must be positive")
    return alpha * t**rho


def catbub_thresholds(
    null_probability: ArrayLike, fractions: ArrayLike, *, alpha: float = 0.05, rho: float = 3
) -> FloatArray:
    """Calibrate survivor-conditional cutoffs, rounded upward to four decimals.

    null_probability has shape (trials, looks, 2), ordered B>A then A>B.
    Linear sample quantiles match R's default type 7. Crossings are strictly >.
    """
    p = finite(null_probability, "null_probability")
    spending = _spending(fractions, alpha, rho)
    if p.ndim != 3 or p.shape[1:] != (len(spending), 2) or p.shape[0] < 2:
        raise ValueError("null_probability must have shape (at least 2 trials, looks, 2)")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("null probabilities must be in [0, 1]")
    eligible = np.ones(len(p), dtype=bool)
    cutoffs = np.empty(len(spending))
    previous = 0.0
    maximum = p.max(axis=-1)
    for look, cumulative in enumerate(spending):
        if not np.any(eligible):
            raise ArithmeticError("no null replicates remain for threshold calibration")
        quantile = (1 - cumulative) / (1 - previous)
        cutoffs[look] = np.ceil(10_000 * np.quantile(maximum[eligible, look], quantile)) / 10_000
        eligible &= maximum[:, look] <= cutoffs[look]
        previous = cumulative
    return _freeze(cutoffs)


@dataclass(frozen=True)
class CatbubOperatingCharacteristics:
    """First stopping probabilities (look, B>A/A>B); sample sizes are per arm."""

    stopping_probability: FloatArray
    stopping_mcse: FloatArray
    superiority_probability: FloatArray
    superiority_mcse: FloatArray
    no_selection_probability: float
    mean_sample_size: FloatArray
    sample_size_mcse: FloatArray
    trials: int


def catbub_operating_characteristics(
    probability: ArrayLike, sample_sizes: ArrayLike, thresholds: ArrayLike
) -> CatbubOperatingCharacteristics:
    """Stop each trial at its first superiority crossing, then summarize it."""
    p, cuts = finite(probability, "probability"), finite(thresholds, "thresholds")
    if p.ndim != 3 or p.shape[-1] != 2 or len(p) < 2 or cuts.shape != (p.shape[1],):
        raise ValueError(
            "probability must be (at least 2 trials, looks, 2), with one cutoff per look"
        )
    if np.any((p < 0) | (p > 1)) or np.any((cuts < 0) | (cuts > 1)):
        raise ValueError("probabilities must be in [0,1] and cutoffs in [0,1]")
    ns = count(sample_sizes, "sample_sizes")
    if ns.ndim == 1:
        ns = np.broadcast_to(ns, (2, len(ns)))
    if ns.shape != (2, p.shape[1]) or np.any(ns <= 0) or np.any(np.diff(ns, axis=1) < 0):
        raise ValueError("sample_sizes must be nondecreasing positive (2, looks) or a look vector")
    crossings = p > cuts[None, :, None]
    if np.any(crossings.sum(axis=-1) > 1):
        raise ValueError("both superiority directions cannot cross at the same look")
    any_crossing = crossings.any(axis=-1)
    stopped = any_crossing.any(axis=1)
    first = any_crossing.argmax(axis=1)
    stop = np.zeros_like(p)
    rows = np.flatnonzero(stopped)
    stop[rows, first[rows]] = crossings[rows, first[rows]]
    frequencies = stop.mean(axis=0)
    superior = stop.sum(axis=1).mean(axis=0)
    last = np.where(stopped, first, p.shape[1] - 1)
    enrollment = ns[:, last].T
    return CatbubOperatingCharacteristics(
        _freeze(frequencies),
        _freeze(np.sqrt(frequencies * (1 - frequencies) / len(p))),
        _freeze(superior),
        _freeze(np.sqrt(superior * (1 - superior) / len(p))),
        float(np.mean(~stopped)),
        _freeze(enrollment.mean(axis=0)),
        _freeze(enrollment.std(axis=0, ddof=1) / np.sqrt(len(p))),
        len(p),
    )


def _trial_probabilities(
    ns: FloatArray,
    scenario: FloatArray,
    utility: ArrayLike,
    trials: int,
    prior_ess: float,
    prior_probability: ArrayLike | None,
    method: str,
    draws: int,
    rng: np.random.Generator,
) -> FloatArray:
    counts = catbub_simulate_counts(ns, scenario, trials=trials, rng=rng)
    shape = counts.shape
    tables = counts.reshape(-1, 2, shape[-1])
    if method == "beta":
        # Repeated multinomial tables have the same deterministic posterior.
        unique, inverse = np.unique(tables.reshape(len(tables), -1), axis=0, return_inverse=True)
        probabilities = catbub_compare(
            unique.reshape(-1, 2, shape[-1]),
            utility,
            prior_ess=prior_ess,
            prior_probability=prior_probability,
            method=method,
        ).probability[inverse]
    else:
        probabilities = catbub_compare(
            tables,
            utility,
            prior_ess=prior_ess,
            prior_probability=prior_probability,
            method=method,
            draws=draws,
            rng=rng,
        ).probability
    return probabilities.reshape(shape[0], shape[1], 2)


@dataclass(frozen=True)
class CatbubDesign:
    sample_sizes: FloatArray
    thresholds: FloatArray
    fractions: FloatArray
    null: CatbubOperatingCharacteristics
    alternatives: tuple[CatbubOperatingCharacteristics, ...]
    calibration_history: FloatArray
    alpha: float
    beta: float
    rho: float
    method: str


def catbub_design(
    scenarios: ArrayLike,
    fractions: ArrayLike,
    utility: ArrayLike,
    *,
    alpha: float = 0.05,
    beta: float = 0.20,
    rho: float = 3,
    epsilon: float = 0.005,
    null_trials: int = 50_000,
    alternative_trials: int = 25_000,
    prior_ess: float = 1,
    prior_probability: ArrayLike | None = None,
    method: str = "beta",
    draws: int = 100_000,
    max_iterations: int = 30,
    max_sample_size: int = 100_000,
    rng: np.random.Generator,
) -> CatbubDesign:
    """Calibrate per-arm maximum N and boundaries against the first alternative.

    The null uses arm A's probabilities in both arms. All later scenarios are
    evaluated at the resulting design. Calibration trials are reused for the
    reported null/target OCs; they are not independent validation trials.
    Failure to reach the requested power window raises RuntimeError.
    """
    scenarios = _probabilities(scenarios, "scenarios")
    if scenarios.ndim == 2:
        scenarios = scenarios[None]
    if scenarios.ndim != 3 or len(scenarios) == 0:
        raise ValueError("scenarios must have shape (scenarios, 2, K)")
    fractions = finite(fractions, "fractions")
    _spending(fractions, alpha, rho)
    if fractions[-1] != 1 or np.any(np.diff(fractions) <= 0):
        raise ValueError("planned fractions must be strictly increasing and end at 1")
    beta, epsilon = scalar(beta, "beta"), scalar(epsilon, "epsilon")
    if not 0 < beta < 0.5 or not 0 < epsilon < min(beta, 1 - beta):
        raise ValueError("beta must be in (0,.5); epsilon must be positive and smaller than beta")
    null_trials = _integer(null_trials, "null_trials")
    alternative_trials = _integer(alternative_trials, "alternative_trials")
    if min(null_trials, alternative_trials) < 2:
        raise ValueError("at least two null and alternative trials are required")
    max_iterations = _integer(max_iterations, "max_iterations", 1000)
    max_sample_size = _integer(max_sample_size, "max_sample_size")
    _, u, _ = _inputs(np.zeros(scenarios.shape[1:]), utility, prior_ess, prior_probability)
    scale = float(np.max(np.abs(u)))
    v = u / scale if scale else u.copy()
    v = v - v.min()
    means = scenarios[0] @ v
    delta = abs(float(means[1] - means[0]))
    if delta == 0:
        raise ValueError("target alternative must have nonzero expected utility difference")
    variance = float(np.sum(scenarios[0] * (v - means[:, None]) ** 2))
    initial = ((ndtri(1 - beta) + ndtri(1 - alpha / 2)) / delta) ** 2 * variance
    if not np.isfinite(initial) or initial > max_sample_size:
        raise ValueError("initial sample-size estimate exceeds max_sample_size")
    n = max(1, int(np.ceil(initial)))
    null_scenario = np.broadcast_to(scenarios[0, 0], scenarios[0].shape)
    history = []
    target_direction = 0 if means[1] > means[0] else 1
    for _ in range(max_iterations):
        ns = np.ceil(n * fractions)
        null_prob = _trial_probabilities(
            ns, null_scenario, u, null_trials, prior_ess, prior_probability, method, draws, rng
        )
        cuts = catbub_thresholds(null_prob, fractions, alpha=alpha, rho=rho)
        alt_prob = _trial_probabilities(
            ns,
            scenarios[0],
            u,
            alternative_trials,
            prior_ess,
            prior_probability,
            method,
            draws,
            rng,
        )
        target = catbub_operating_characteristics(alt_prob, ns, cuts)
        power = float(target.superiority_probability[target_direction])
        history.append((n, power, cuts[-1]))
        if abs((1 - power) - beta) < epsilon:
            break
        if not 0 < power < 1 or cuts[-1] >= 1:
            raise RuntimeError(
                f"calibration update undefined at N={n}, power={power}, cutoff={cuts[-1]}"
            )
        denominator = ndtri(power) + ndtri(cuts[-1])
        if denominator <= 0:
            raise RuntimeError(
                "normal-approximation sample-size update has nonpositive denominator"
            )
        update = n * ((ndtri(1 - beta) + ndtri(cuts[-1])) / denominator) ** 2
        if not np.isfinite(update) or update > max_sample_size or update < 1:
            raise RuntimeError("calibration update exceeds sample-size limits")
        n = int(np.ceil(update))
    else:
        raise RuntimeError(
            f"calibration did not converge after {max_iterations} iterations; last={history[-1]}"
        )
    alternatives = [target]
    for scenario in scenarios[1:]:
        probability = _trial_probabilities(
            ns, scenario, u, alternative_trials, prior_ess, prior_probability, method, draws, rng
        )
        alternatives.append(catbub_operating_characteristics(probability, ns, cuts))
    return CatbubDesign(
        _freeze(ns),
        cuts,
        _freeze(fractions),
        catbub_operating_characteristics(null_prob, ns, cuts),
        tuple(alternatives),
        _freeze(history),
        float(alpha),
        beta,
        float(rho),
        method,
    )


@dataclass(frozen=True)
class CatbubAnalysis:
    decision: str
    threshold: float
    probability: FloatArray
    thresholds: FloatArray
    incremental_alpha: float
    cumulative_alpha: float
    null: CatbubOperatingCharacteristics


def catbub_analysis(
    counts: ArrayLike,
    utility: ArrayLike,
    previous_sample_sizes: ArrayLike,
    maximum_sample_size: int,
    *,
    alpha: float = 0.05,
    rho: float = 3,
    null_trials: int = 50_000,
    prior_ess: float = 1,
    prior_probability: ArrayLike | None = None,
    method: str = "beta",
    draws: int = 100_000,
    rng: np.random.Generator,
) -> CatbubAnalysis:
    """Recalibrate using the pooled observed null and actual previous look sizes."""
    x, u, _ = _inputs(counts, utility, prior_ess, prior_probability)
    if x.ndim != 2 or np.any(x.sum(axis=-1) <= 0):
        raise ValueError("counts must be a (2,K) table with nonempty arms")
    maximum = _integer(maximum_sample_size, "maximum_sample_size")
    prev = count(previous_sample_sizes, "previous_sample_sizes")
    if prev.ndim == 1:
        prev = np.broadcast_to(prev, (2, len(prev)))
    if prev.ndim != 2 or prev.shape[0] != 2:
        raise ValueError("previous_sample_sizes must be a look vector or (2, looks)")
    ns = np.column_stack((prev, x.sum(axis=-1)))
    if np.any(ns > maximum):
        raise ValueError("observed or previous sample sizes exceed the planned maximum")
    fractions = ns.sum(axis=0) / (2 * maximum)
    spending = _spending(fractions, alpha, rho)
    pooled = x.sum(axis=0) / x.sum()
    null = _trial_probabilities(
        ns,
        np.broadcast_to(pooled, x.shape),
        u,
        null_trials,
        prior_ess,
        prior_probability,
        method,
        draws,
        rng,
    )
    cuts = catbub_thresholds(null, fractions, alpha=alpha, rho=rho)
    posterior = catbub_compare(
        x,
        u,
        prior_ess=prior_ess,
        prior_probability=prior_probability,
        method=method,
        draws=draws,
        rng=rng,
    ).probability
    decision = (
        "B>A"
        if posterior[0] > cuts[-1]
        else "A>B"
        if posterior[1] > cuts[-1]
        else (
            "No superiority at maximum sample size"
            if x.sum() == 2 * maximum
            else "Continue Enrollment"
        )
    )
    return CatbubAnalysis(
        decision,
        float(cuts[-1]),
        posterior,
        cuts,
        float(spending[-1] - (spending[-2] if len(spending) > 1 else 0)),
        float(spending[-1]),
        catbub_operating_characteristics(null, ns, cuts),
    )
