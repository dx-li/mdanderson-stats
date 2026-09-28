"""Continuation-ratio trinary EffTox likelihood and posterior fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from scipy.stats import truncnorm

from ._validation import FloatArray, finite
from .efftox_model import efftox_standardize
from .hierarchical_binomial import ChainSummary, summarize_chains

_MAX_OUTPUT_CELLS = 200_000
_MAX_SAMPLING_WORK = 2_000_000
_MAX_LIKELIHOOD_EVALUATIONS = 2_000_000


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


@dataclass(frozen=True)
class EffToxTrinaryPrior:
    """Independent Gaussian priors for ``(mu_T,beta_T,mu_Q,beta_Q)``.

    The two slope priors are conditioned to be positive. A zero standard
    deviation fixes a coefficient at its supplied mean; fixed slopes must be
    strictly positive.
    """

    mean: FloatArray
    sd: FloatArray

    def __post_init__(self) -> None:
        raw_mean, raw_sd = np.asarray(self.mean), np.asarray(self.sd)
        if raw_mean.shape != (4,) or raw_sd.shape != (4,):
            raise ValueError("prior mean and sd must each contain four coefficients")
        mean = finite(raw_mean, "prior mean")
        sd = finite(raw_sd, "prior sd")
        if np.any(np.abs(mean) > 1e4) or np.any(sd < 0) or np.any(sd > 100):
            raise ValueError("prior means must be within 1e4 and SDs in [0,100]")
        for index in (1, 3):
            if sd[index] == 0 and mean[index] <= 0:
                raise ValueError("fixed monotone efficacy/toxicity slopes must be positive")
        object.__setattr__(self, "mean", _owned(mean))
        object.__setattr__(self, "sd", _owned(sd))


def _components(dose_codes: FloatArray, parameters: FloatArray) -> tuple[FloatArray, ...]:
    mu_t, beta_t, mu_q, beta_q = np.moveaxis(parameters, -1, 0)
    eta_t = np.expand_dims(mu_t, -1) + np.expand_dims(beta_t, -1) * dose_codes
    eta_q = np.expand_dims(mu_q, -1) + np.expand_dims(beta_q, -1) * dose_codes
    log_t = -np.logaddexp(0.0, -eta_t)
    log_not_t = -np.logaddexp(0.0, eta_t)
    log_q = -np.logaddexp(0.0, -eta_q)
    log_not_q = -np.logaddexp(0.0, eta_q)
    log_neither = log_not_t + log_not_q
    log_efficacy = log_not_t + log_q
    logs = np.stack((log_neither, log_efficacy, log_t), axis=-1)
    log_not_marginal_efficacy = np.logaddexp(log_t, log_neither)
    marginal_efficacy_logit = log_efficacy - log_not_marginal_efficacy
    if any(np.any(np.isnan(value)) or np.any(np.isposinf(value)) for value in (logs, eta_t, eta_q)):
        raise ArithmeticError("trinary EffTox probabilities are nonfinite")
    return logs, marginal_efficacy_logit, eta_t, eta_q


def efftox_trinary_log_probabilities(dose_codes: ArrayLike, parameters: ArrayLike) -> FloatArray:
    """Return log probabilities in ``(neither, efficacy, toxicity)`` order.

    Parameters may have leading batch dimensions and end in four coefficients.
    The efficacy coefficient pair models efficacy conditional on no toxicity.
    """
    raw_x, raw_theta = np.asarray(dose_codes), np.asarray(parameters)
    if raw_x.ndim != 1 or not 2 <= raw_x.size <= 20:
        raise ValueError("dose_codes must contain between 2 and 20 values")
    if raw_theta.ndim < 1 or raw_theta.shape[-1] != 4:
        raise ValueError("parameters must end in four coefficients")
    if raw_theta.size // 4 * raw_x.size * 3 > _MAX_OUTPUT_CELLS:
        raise ValueError("batched trinary probability output exceeds 200000 cells")
    x, theta = finite(raw_x, "dose_codes"), finite(raw_theta, "parameters")
    if np.any(np.abs(x) > 1500) or np.any(np.abs(theta) > 1e4):
        raise ValueError("dose codes and coefficients exceed the supported range")
    return _owned(_components(x, theta)[0])


def efftox_trinary_log_likelihood(
    dose_codes: ArrayLike, counts: ArrayLike, parameters: ArrayLike
) -> FloatArray:
    """Return trinary multinomial log likelihood for each parameter row."""
    raw_x, raw_counts, raw_theta = (
        np.asarray(dose_codes),
        np.asarray(counts),
        np.asarray(parameters),
    )
    if raw_x.ndim != 1 or not 2 <= raw_x.size <= 20:
        raise ValueError("dose_codes must contain between 2 and 20 values")
    if raw_counts.shape != (raw_x.size, 3):
        raise ValueError("counts must have shape (dose,3): neither, efficacy, toxicity")
    if raw_theta.ndim < 1 or raw_theta.shape[-1] != 4:
        raise ValueError("parameters must end in four coefficients")
    if raw_theta.size // 4 * raw_x.size * 3 > _MAX_OUTPUT_CELLS:
        raise ValueError("batched trinary likelihood exceeds 200000 cells")
    x, n, theta = (
        finite(raw_x, "dose_codes"),
        finite(raw_counts, "counts"),
        finite(raw_theta, "parameters"),
    )
    if np.any(np.abs(x) > 1500) or np.any(np.abs(theta) > 1e4):
        raise ValueError("dose codes and coefficients must be within supported numeric limits")
    if np.any(n < 0) or np.any(n != np.floor(n)) or np.any(n > 10_000) or float(n.sum()) > 10_000:
        raise ValueError("counts must be nonnegative integers totaling at most 10000")
    logs = _components(x, theta)[0]
    contributions = np.zeros_like(logs)
    np.multiply(n, logs, out=contributions, where=n > 0)
    result = np.sum(contributions, axis=(-1, -2))
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ArithmeticError("trinary EffTox log likelihood is invalid")
    return _owned(result)


def efftox_trinary_predict(dose_codes: ArrayLike, parameters: ArrayLike) -> FloatArray:
    """Return joint probabilities ordered ``(neither, efficacy, toxicity)``."""
    probabilities = np.exp(efftox_trinary_log_probabilities(dose_codes, parameters))
    if not np.all(np.isfinite(probabilities)):
        raise ArithmeticError("trinary EffTox predicted probabilities are nonfinite")
    return _owned(probabilities)


def _sample_prior(
    prior: EffToxTrinaryPrior, size: tuple[int, int], rng: np.random.Generator
) -> FloatArray:
    values = prior.mean + prior.sd * rng.normal(size=(*size, 4))
    for index in (1, 3):
        sd = float(prior.sd[index])
        if sd == 0:
            values[..., index] = prior.mean[index]
        else:
            values[..., index] = truncnorm.rvs(
                -prior.mean[index] / sd,
                np.inf,
                loc=prior.mean[index],
                scale=sd,
                size=size,
                random_state=rng,
            )
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1e4):
        raise ArithmeticError("Gaussian prior draws exceed supported parameter range")
    if np.any(values[..., (1, 3)] <= 0):
        raise ArithmeticError("truncated Gaussian slopes must be positive")
    return values


def _ellipse_step(
    state: FloatArray,
    prior: EffToxTrinaryPrior,
    counts: FloatArray,
    dose_codes: FloatArray,
    current_ll: float,
    rng: np.random.Generator,
    evaluation_budget: int,
) -> tuple[FloatArray, float, int]:
    centered = state - prior.mean
    direction = prior.sd * rng.normal(size=4)
    height = current_ll + np.log1p(-rng.random())
    angle = rng.uniform(0.0, 2.0 * np.pi)
    lower, upper = angle - 2.0 * np.pi, angle
    evaluations = 0
    for _ in range(1000):
        proposal = prior.mean + centered * np.cos(angle) + direction * np.sin(angle)
        if np.any(proposal[[1, 3]] <= 0):
            ll = -np.inf
        else:
            if evaluations >= evaluation_budget:
                raise ValueError("trinary EffTox likelihood-evaluation budget exceeded")
            ll = float(efftox_trinary_log_likelihood(dose_codes, counts, proposal))
            evaluations += 1
        if ll >= height:
            return proposal, ll, evaluations
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise ArithmeticError("trinary EffTox elliptical-slice bracket failed")


@dataclass(frozen=True)
class EffToxTrinaryFit:
    """Posterior draws and outcome probabilities with chain/draw axes."""

    doses: FloatArray
    dose_codes: FloatArray
    counts: FloatArray
    parameters: FloatArray
    log_probabilities: FloatArray
    joint_probabilities: FloatArray
    efficacy_probabilities: FloatArray
    toxicity_probabilities: FloatArray
    conditional_efficacy_probabilities: FloatArray
    efficacy_logits: FloatArray
    toxicity_logits: FloatArray
    conditional_efficacy_logits: FloatArray
    log_likelihood: FloatArray
    summary: ChainSummary
    warmup: int
    likelihood_evaluations: int


def fit_efftox_trinary(
    doses: ArrayLike,
    counts: ArrayLike,
    *,
    prior: EffToxTrinaryPrior,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    zero_dose_shift: bool = True,
    rng: np.random.Generator,
) -> EffToxTrinaryFit:
    """Fit the original four-parameter continuation-ratio EffTox model.

    The efficacy model is conditional on no toxicity. Posterior efficacy
    probability is retained marginally as ``(1-t)*q``. Elliptical slice
    sampling uses the explicit Gaussian prior and conditions both slopes to
    be positive. Diagnostics describe retained chains; they do not certify
    convergence.
    """
    if not isinstance(prior, EffToxTrinaryPrior):
        raise ValueError("prior must be an EffToxTrinaryPrior")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    raw_doses, raw_counts = np.asarray(doses), np.asarray(counts)
    if raw_doses.ndim != 1 or not 2 <= raw_doses.size <= 20:
        raise ValueError("doses must be a 1D array of 2 to 20 values")
    if raw_counts.shape != (raw_doses.size, 3):
        raise ValueError("counts must have shape (dose,3): neither, efficacy, toxicity")
    dose_values, n = finite(raw_doses, "doses"), finite(raw_counts, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)) or float(n.sum()) > 10_000:
        raise ValueError("counts must be nonnegative integers totaling at most 10000")
    x = efftox_standardize(dose_values, zero_dose_shift=zero_dose_shift)
    draw_count = _integer(draws, "draws", 8, 10_000)
    warm = _integer(warmup, "warmup", 0, 10_000)
    chain_count = _integer(chains, "chains", 2, 4)
    if chain_count * draw_count * x.size * 3 > _MAX_OUTPUT_CELLS:
        raise ValueError("retained trinary probability draws exceed 200000 cells")
    iterations = warm + draw_count
    if chain_count * iterations * x.size * 4 > _MAX_SAMPLING_WORK:
        raise ValueError("trinary EffTox sampling work exceeds the supported budget")
    if chain_count * iterations * x.size > _MAX_LIKELIHOOD_EVALUATIONS:
        raise ValueError("trinary EffTox likelihood-evaluation budget exceeded")

    if initial is None:
        starts = np.tile(prior.mean, (chain_count, 1))
        for index in (1, 3):
            if starts[0, index] <= 0:
                starts[:, index] = max(float(prior.mean[index]), float(prior.sd[index]), 0.1)
    else:
        candidate_start = np.asarray(initial)
        if candidate_start.shape != (chain_count, 4):
            raise ValueError("initial must have one four-coefficient row per chain")
        starts = finite(candidate_start, "initial").copy()
        if np.any(np.abs(starts) > 1e4) or np.any(starts[:, (1, 3)] <= 0):
            raise ValueError("initial coefficients must be bounded with positive slopes")
        fixed = prior.sd == 0
        if np.any(starts[:, fixed] != prior.mean[fixed]):
            raise ValueError("initial values for fixed prior coefficients must equal their means")

    parameter_draws = np.empty((chain_count, draw_count, 4), dtype=np.float64)
    likelihood_draws = np.empty((chain_count, draw_count), dtype=np.float64)
    evaluations = 0
    if n.sum() == 0:
        parameter_draws = _sample_prior(prior, (chain_count, draw_count), rng)
        likelihood_draws.fill(0.0)
    else:
        for chain in range(chain_count):
            state = starts[chain].copy()
            if evaluations >= _MAX_LIKELIHOOD_EVALUATIONS:
                raise ValueError("trinary EffTox likelihood-evaluation budget exceeded")
            current_ll = float(efftox_trinary_log_likelihood(x, n, state))
            evaluations += 1
            for iteration in range(warm + draw_count):
                remaining_evaluations = _MAX_LIKELIHOOD_EVALUATIONS - evaluations
                state, current_ll, used = _ellipse_step(
                    state, prior, n, x, current_ll, rng, remaining_evaluations
                )
                evaluations += used
                if iteration >= warm:
                    index = iteration - warm
                    parameter_draws[chain, index] = state
                    likelihood_draws[chain, index] = current_ll

    logs, efficacy_logits, toxicity_logits, conditional_logits = _components(x, parameter_draws)
    joint = np.exp(logs)
    conditional = expit(conditional_logits)
    toxicity = expit(toxicity_logits)
    efficacy = np.exp(logs[..., 1])
    if not all(np.all(np.isfinite(value)) for value in (joint, conditional, toxicity, efficacy)):
        raise ArithmeticError("trinary EffTox posterior probabilities are nonfinite")
    return EffToxTrinaryFit(
        _owned(dose_values),
        x,
        _owned(n),
        _owned(parameter_draws),
        _owned(logs),
        _owned(joint),
        _owned(efficacy),
        _owned(toxicity),
        _owned(conditional),
        _owned(efficacy_logits),
        _owned(toxicity_logits),
        _owned(conditional_logits),
        _owned(likelihood_draws),
        summarize_chains(parameter_draws),
        warm,
        evaluations,
    )
