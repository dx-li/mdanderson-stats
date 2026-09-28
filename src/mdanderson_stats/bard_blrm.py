"""Explicit-prior Bayesian logistic dose-toxicity model for BARD's BF-BLRM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .hierarchical_binomial import ChainSummary, summarize_chains

_MAX_DOSES = 100
_MAX_RETAINED_CELLS = 2_000_000
_MAX_EVALUATIONS = 2_000_000
_MAX_WORK = 50_000_000
_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000
_MAX_CHAINS = 16
_ESS_BRACKET_LIMIT = 1000


@dataclass(frozen=True)
class BARDLogisticPrior:
    """Independent Normal priors for ``(log(alpha), log(beta))``.

    A zero standard deviation fixes that coordinate at its prior mean; it is
    an explicit point mass, not a nondegenerate Normal distribution.
    """

    mean: FloatArray
    standard_deviation: FloatArray

    def __init__(self, mean: ArrayLike, standard_deviation: ArrayLike) -> None:
        if np.shape(mean) != (2,) or np.shape(standard_deviation) != (2,):
            raise ValueError("prior mean and nonnegative standard_deviation must have length 2")
        if np.iscomplexobj(mean) or np.iscomplexobj(standard_deviation):
            raise ValueError("prior mean and standard_deviation must be real")
        location = finite(mean, "prior mean")
        scale = finite(standard_deviation, "prior standard_deviation")
        if location.shape != (2,) or scale.shape != (2,) or np.any(scale < 0):
            raise ValueError("prior mean and nonnegative standard_deviation must have length 2")
        object.__setattr__(self, "mean", _freeze(location))
        object.__setattr__(self, "standard_deviation", _freeze(scale))


@dataclass(frozen=True)
class BARDLogisticFit:
    """Retained BF-BLRM draws and posterior dose-toxicity summaries."""

    doses: FloatArray
    reference_dose: float
    target_interval: FloatArray
    prior: BARDLogisticPrior
    coefficient_draws: FloatArray
    probability_draws: FloatArray
    target_indicator_draws: NDArray[np.bool_]
    overdose_indicator_draws: NDArray[np.bool_]
    coefficient_summary: ChainSummary
    probability_summary: ChainSummary
    target_probability_summary: ChainSummary
    overdose_probability_summary: ChainSummary
    likelihood_evaluations: int
    work_units: int
    warmup: int

    @property
    def posterior_target_probability(self) -> FloatArray:
        """Monte Carlo estimate of ``Pr(gamma1 < p < gamma2 | data)`` by dose."""
        return _freeze(self.target_indicator_draws.mean(axis=(0, 1)))

    @property
    def posterior_overdose_probability(self) -> FloatArray:
        """Monte Carlo estimate of ``Pr(p >= gamma2 | data)`` by dose."""
        return _freeze(self.overdose_indicator_draws.mean(axis=(0, 1)))


def _doses(value: ArrayLike) -> FloatArray:
    shape = np.shape(value)
    if len(shape) != 1 or not 1 <= shape[0] <= _MAX_DOSES:
        raise ValueError(f"doses must be a vector with 1..{_MAX_DOSES} entries")
    if np.iscomplexobj(value):
        raise ValueError("doses must be real")
    doses = finite(value, "doses")
    if np.any(doses <= 0) or np.any(np.diff(doses) <= 0):
        raise ValueError("doses must be strictly increasing positive values")
    return doses


def _groups(
    doses: FloatArray, patients: ArrayLike, toxicities: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    if np.shape(patients) != doses.shape or np.shape(toxicities) != doses.shape:
        raise ValueError("patients and toxicities must match the dose vector shape")
    if np.iscomplexobj(patients) or np.iscomplexobj(toxicities):
        raise ValueError("patients and toxicities must be real counts")
    n = count(patients, "patients")
    y = count(toxicities, "toxicities")
    if np.any(y > n):
        raise ValueError("toxicities cannot exceed patients at a dose")
    return n, y


def _target_interval(value: ArrayLike) -> FloatArray:
    if np.shape(value) != (2,):
        raise ValueError("target_interval must satisfy 0 <= gamma1 < gamma2 <= 1")
    if np.iscomplexobj(value):
        raise ValueError("target_interval must be real")
    target = finite(value, "target_interval")
    if target.shape != (2,) or not 0 <= target[0] < target[1] <= 1:
        raise ValueError("target_interval must satisfy 0 <= gamma1 < gamma2 <= 1")
    return target


def _linear_predictor(
    coefficients: FloatArray, doses: FloatArray, reference_dose: float
) -> FloatArray:
    log_alpha = float(coefficients[0])
    log_beta = float(coefficients[1])
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ratio = doses / reference_dose
        log_ratio = np.where(
            (ratio > 0) & np.isfinite(ratio), np.log(ratio), np.log(doses) - np.log(reference_dose)
        )
        log_effect = log_beta + log_ratio
        effect = np.exp(log_effect)
        eta = log_alpha + effect
    if not np.all(np.isfinite(eta)):
        raise ArithmeticError("BF-BLRM linear predictor exceeds floating-point range")
    return eta


def bard_blrm_probability(
    doses: ArrayLike,
    reference_dose: float,
    log_alpha: float,
    log_beta: float,
) -> FloatArray:
    """Evaluate the paper's raw-ratio BF-BLRM toxicity curve.

    The source model is ``logit(p_j) = log(alpha) + beta * (d_j/d_star)``;
    it is not a regression on log dose. Inputs are the natural logarithms of
    the positive parameters ``alpha`` and ``beta``.
    """
    grid = _doses(doses)
    if np.iscomplexobj(reference_dose):
        raise ValueError("reference_dose must be real and positive")
    reference_array = finite(reference_dose, "reference_dose")
    if reference_array.ndim != 0 or float(reference_array) <= 0:
        raise ValueError("reference_dose must be finite and positive")
    reference = float(reference_array)
    if (
        np.asarray(log_alpha).ndim != 0
        or np.asarray(log_beta).ndim != 0
        or np.iscomplexobj(log_alpha)
        or np.iscomplexobj(log_beta)
    ):
        raise ValueError("log_alpha and log_beta must be real scalars")
    coefficients = finite([log_alpha, log_beta], "log coefficients")
    eta = _linear_predictor(coefficients, grid, reference)
    return _freeze(expit(eta))


def fit_bard_blrm(
    doses: ArrayLike,
    patients: ArrayLike,
    toxicities: ArrayLike,
    reference_dose: float,
    prior: BARDLogisticPrior,
    *,
    target_interval: ArrayLike,
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
    max_evaluations: int = _MAX_EVALUATIONS,
    max_work: int = _MAX_WORK,
) -> BARDLogisticFit:
    """Fit the explicit-prior BF-BLRM and summarize target/overdose posterior.

    ``patients`` and ``toxicities`` are grouped counts aligned to ``doses``.
    The independent Normal priors on log(alpha) and log(beta) are supplied by
    the caller; no native prior default is inferred. Sampling uses serial
    elliptical slice updates. Chain diagnostics are estimates, not convergence
    guarantees. The fit covers the dose-toxicity model only, not the BF-BLRM
    scheduler or backfill rules.
    """
    if not isinstance(prior, BARDLogisticPrior):
        raise TypeError("prior must be a BARDLogisticPrior")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    grid = _doses(doses)
    n, y = _groups(grid, patients, toxicities)
    target = _target_interval(target_interval)
    if np.iscomplexobj(reference_dose):
        raise ValueError("reference_dose must be real and positive")
    reference_arr = finite(reference_dose, "reference_dose")
    if reference_arr.ndim != 0 or float(reference_arr) <= 0:
        raise ValueError("reference_dose must be a finite positive scalar")
    reference = float(reference_arr)
    settings = finite([draws, warmup, chains, max_evaluations, max_work], "sampler settings")
    if np.any(settings != np.floor(settings)):
        raise ValueError("draws, warmup, chains and budgets must be integers")
    draw_count, warmup_count, chain_count, evaluation_limit, work_limit = map(int, settings)
    if not 8 <= draw_count <= _MAX_DRAWS:
        raise ValueError(f"draws must be in 8..{_MAX_DRAWS}")
    if not 0 <= warmup_count <= _MAX_WARMUP:
        raise ValueError(f"warmup must be in 0..{_MAX_WARMUP}")
    if not 2 <= chain_count <= _MAX_CHAINS:
        raise ValueError(f"chains must be in 2..{_MAX_CHAINS}")
    if not 1 <= evaluation_limit <= _MAX_EVALUATIONS:
        raise ValueError(f"max_evaluations must be in 1..{_MAX_EVALUATIONS}")
    if not 1 <= work_limit <= _MAX_WORK:
        raise ValueError(f"max_work must be in 1..{_MAX_WORK}")
    retained = chain_count * draw_count * (2 + 3 * grid.size)
    if retained > _MAX_RETAINED_CELLS:
        raise ValueError("retained BF-BLRM posterior arrays exceed two million cells")

    has_data = bool(np.any(n))
    free = prior.standard_deviation > 0
    steps = warmup_count + draw_count
    minimum_evaluations = chain_count * (
        (1 + steps) if has_data and np.any(free) else (1 if has_data else 0)
    )
    minimum_work = minimum_evaluations * int(grid.size) + chain_count * draw_count * int(grid.size)
    if minimum_evaluations > evaluation_limit:
        raise ValueError("max_evaluations is below the minimum likelihood-call count")
    if minimum_work > work_limit:
        raise ValueError("max_work is below the minimum patient/dose computation work")

    gamma1, gamma2 = map(float, target)
    logit1 = -np.inf if gamma1 == 0 else np.log(gamma1) - np.log1p(-gamma1)
    logit2 = np.inf if gamma2 == 1 else np.log(gamma2) - np.log1p(-gamma2)
    evaluations = 0
    work = 0

    def log_likelihood(coefficients: FloatArray) -> float:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit:
            raise RuntimeError("BF-BLRM likelihood evaluation budget exhausted")
        evaluations += 1
        work += int(grid.size)
        if work > work_limit:
            raise RuntimeError("BF-BLRM computation work budget exhausted")
        eta = _linear_predictor(coefficients, grid, reference)
        with np.errstate(over="ignore", invalid="ignore"):
            contributions = y * -np.logaddexp(0.0, -eta) + (n - y) * -np.logaddexp(0.0, eta)
            value = float(np.sum(contributions))
        return value if np.isfinite(value) else -np.inf

    samples = np.empty((chain_count, draw_count, 2), dtype=float)
    probabilities = np.empty((chain_count, draw_count, grid.size), dtype=float)
    target_indicators = np.empty((chain_count, draw_count, grid.size), dtype=bool)
    overdose_indicators = np.empty_like(target_indicators)
    mean = np.asarray(prior.mean)
    sd = np.asarray(prior.standard_deviation)

    for chain in range(chain_count):
        current = rng.normal(mean, sd) if has_data and np.any(free) else mean.copy()
        current_ll = log_likelihood(current) if has_data else 0.0
        if not np.isfinite(current_ll):
            raise ArithmeticError("initial state has an unrepresentable BF-BLRM likelihood")
        if not np.any(free):
            samples[chain] = current
        elif not has_data:
            samples[chain] = rng.normal(mean, sd, size=(draw_count, 2))
        else:
            for iteration in range(warmup_count + draw_count):
                direction = np.zeros(2, dtype=float)
                direction[free] = rng.normal(size=int(np.count_nonzero(free))) * sd[free]
                threshold = current_ll + np.log(max(float(rng.random()), np.finfo(float).tiny))
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                lower, upper = angle - 2.0 * np.pi, angle
                centered = current - mean
                for _ in range(_ESS_BRACKET_LIMIT):
                    proposal = mean + centered * np.cos(angle) + direction * np.sin(angle)
                    proposal_ll = log_likelihood(proposal)
                    if proposal_ll >= threshold:
                        current, current_ll = proposal, proposal_ll
                        break
                    if angle < 0:
                        lower = angle
                    else:
                        upper = angle
                    angle = float(rng.uniform(lower, upper))
                else:
                    raise ArithmeticError("BF-BLRM elliptical-slice bracket failed")
                if iteration >= warmup_count:
                    samples[chain, iteration - warmup_count] = current

        for draw in range(draw_count):
            work += int(grid.size)
            if work > work_limit:
                raise RuntimeError("BF-BLRM computation work budget exhausted")
            eta = _linear_predictor(samples[chain, draw], grid, reference)
            probabilities[chain, draw] = expit(eta)
            target_indicators[chain, draw] = (eta > logit1) & (eta < logit2)
            overdose_indicators[chain, draw] = eta >= logit2

    return BARDLogisticFit(
        _freeze(grid),
        reference,
        _freeze(target),
        prior,
        _freeze(samples),
        _freeze(probabilities),
        _freeze_bool(target_indicators),
        _freeze_bool(overdose_indicators),
        summarize_chains(samples),
        summarize_chains(probabilities),
        summarize_chains(target_indicators),
        summarize_chains(overdose_indicators),
        evaluations,
        work,
        warmup_count,
    )


def _freeze_bool(value: NDArray[np.bool_]) -> NDArray[np.bool_]:
    array = np.asarray(value, dtype=np.bool_)
    return np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)
