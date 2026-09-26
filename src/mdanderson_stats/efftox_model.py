"""Bayesian bivariate-binary EffTox dose model and posterior fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import truncnorm

from ._validation import FloatArray, finite
from .hierarchical_binomial import ChainSummary, summarize_chains


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _real(value: ArrayLike, name: str) -> FloatArray:
    return finite(value, name)


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


@dataclass(frozen=True)
class EffToxPrior:
    """Independent Gaussian coefficient prior; values must be supplied explicitly.

    Coefficient order is (mu_T, beta_T, mu_E, beta_E1, beta_E2, psi).
    A zero SD fixes that coefficient. With monotone toxicity, beta_T has a
    Gaussian distribution truncated to positive values.
    """

    mean: FloatArray
    sd: FloatArray
    monotone_toxicity: bool = True

    def __post_init__(self) -> None:
        raw_mean = np.asarray(self.mean, dtype=np.float64)
        raw_sd = np.asarray(self.sd, dtype=np.float64)
        if raw_mean.shape != (6,) or raw_sd.shape != (6,):
            raise ValueError("prior mean and sd must each contain six coefficients")
        mean = _real(raw_mean, "prior mean")
        sd = _real(raw_sd, "prior sd")
        if np.any(np.abs(mean) > 1e4) or np.any(sd < 0) or np.any(sd > 100):
            raise ValueError("prior means must be within 1e4 and SDs in [0,100]")
        if not isinstance(self.monotone_toxicity, (bool, np.bool_)):
            raise ValueError("monotone_toxicity must be boolean")
        if self.monotone_toxicity and sd[1] == 0 and mean[1] <= 0:
            raise ValueError("fixed monotone toxicity beta_T must be positive")
        object.__setattr__(self, "mean", _owned(mean))
        object.__setattr__(self, "sd", _owned(sd))
        object.__setattr__(self, "monotone_toxicity", bool(self.monotone_toxicity))


def efftox_standardize(doses: ArrayLike, *, zero_dose_shift: bool = True) -> FloatArray:
    """Return centered log doses, supporting the original zero-dose convention."""
    candidate = np.asarray(doses, dtype=np.float64)
    if candidate.ndim != 1 or not 2 <= candidate.size <= 20:
        raise ValueError("doses must be a 1D array of 2 to 20 nonnegative values")
    raw = _real(candidate, "doses")
    if np.any(raw < 0):
        raise ValueError("doses must be nonnegative")
    if np.any(np.diff(raw) <= 0):
        raise ValueError("doses must be strictly increasing")
    if not isinstance(zero_dose_shift, (bool, np.bool_)):
        raise ValueError("zero_dose_shift must be boolean")
    if np.any(raw == 0):
        if not zero_dose_shift or raw[0] != 0 or raw.size < 2:
            raise ValueError("zero dose requires zero_dose_shift and must be the first dose")
        logged = np.logaddexp(
            np.log(raw[1]),
            np.log(raw, where=raw > 0, out=np.full_like(raw, -np.inf)),
        )
    else:
        logged = np.log(raw)
    coded = logged - logged.mean()
    if not np.all(np.isfinite(coded)) or np.any(np.abs(coded) > 1500):
        raise ValueError("standardized dose values must be finite and within +/-1500")
    return _owned(coded)


def _log_cells_from_logits(eta_e: FloatArray, eta_t: FloatArray, psi: FloatArray) -> FloatArray:
    """Stable log cell probabilities; final axes are (efficacy,toxicity)."""
    log_e = -np.logaddexp(0.0, -eta_e)
    log_1e = -np.logaddexp(0.0, eta_e)
    log_t = -np.logaddexp(0.0, -eta_t)
    log_1t = -np.logaddexp(0.0, eta_t)
    logits = np.stack(
        (
            np.stack((log_1e + log_1t, log_1e + log_t), axis=-1),
            np.stack((log_e + log_1t, log_e + log_t), axis=-1),
        ),
        axis=-2,
    )
    # In a cell, the ratio of the association term to the independence term
    # is rho times the product of the opposite marginal factors.
    log_e_factor = np.stack((log_e, log_1e), axis=-1)
    log_1e_factor = np.stack((log_1e, log_e), axis=-1)
    log_t_factor = np.stack((log_t, log_1t), axis=-1)
    log_1t_factor = np.stack((log_1t, log_t), axis=-1)
    log_a = log_e_factor[..., :, None] + log_t_factor[..., None, :]
    log_one_minus_a = np.logaddexp(
        log_1e_factor[..., :, None],
        log_e_factor[..., :, None] + log_1t_factor[..., None, :],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abs_rho = np.log(np.abs(np.tanh(psi / 2.0)))
    log_one_minus_abs_rho = np.log(2.0) - np.logaddexp(0.0, np.abs(psi))
    # Cell parity is (-1)^(a+b); positive rho therefore subtracts in 01/10.
    parity = np.array([[1.0, -1.0], [-1.0, 1.0]])
    sign = parity * np.sign(np.tanh(psi / 2.0))[..., None, None]
    positive = np.logaddexp(0.0, log_abs_rho[..., None, None] + log_a)
    negative = np.logaddexp(
        log_one_minus_abs_rho[..., None, None],
        log_abs_rho[..., None, None] + log_one_minus_a,
    )
    correction = np.where(sign < 0, negative, positive)
    correction = np.where(np.isneginf(log_abs_rho)[..., None, None], 0.0, correction)
    result = logits + correction
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ArithmeticError("joint EffTox cell log probability is nonfinite")
    return result


def efftox_log_joint_probabilities(dose_codes: ArrayLike, parameters: ArrayLike) -> FloatArray:
    """Evaluate log Pr(E=a,T=b) for every supplied coefficient vector and dose."""
    raw_x = np.asarray(dose_codes, dtype=np.float64)
    raw_theta = np.asarray(parameters, dtype=np.float64)
    if raw_x.ndim != 1 or not 2 <= raw_x.size <= 20:
        raise ValueError("dose_codes must be 1D and parameters must end in six coefficients")
    if raw_theta.ndim < 1 or raw_theta.shape[-1] != 6:
        raise ValueError("dose_codes must be 1D and parameters must end in six coefficients")
    if (raw_theta.size // 6) * raw_x.size > 200_000:
        raise ValueError("batched dose-probability prediction exceeds 200000 cells")
    x = _real(raw_x, "dose_codes")
    theta = _real(raw_theta, "parameters")
    if np.any(np.abs(x) > 1500):
        raise ValueError("dose_codes must lie within +/-1500")
    if np.any(np.abs(theta) > 1e4):
        raise ValueError("parameter values must be within 1e4")
    mu_t, beta_t, mu_e, beta_e1, beta_e2, psi = np.moveaxis(theta, -1, 0)
    eta_t = np.expand_dims(mu_t, -1) + np.expand_dims(beta_t, -1) * x
    eta_e = (
        np.expand_dims(mu_e, -1)
        + np.expand_dims(beta_e1, -1) * x
        + np.expand_dims(beta_e2, -1) * x**2
    )
    return _owned(_log_cells_from_logits(eta_e, eta_t, np.expand_dims(psi, -1)))


def efftox_log_likelihood(
    dose_codes: ArrayLike, counts: ArrayLike, parameters: ArrayLike
) -> FloatArray:
    """Return log likelihood for integer dose-by-efficacy-by-toxicity counts."""
    candidate_x = np.asarray(dose_codes, dtype=np.float64)
    if candidate_x.ndim != 1 or not 2 <= candidate_x.size <= 20:
        raise ValueError("dose_codes must contain between 2 and 20 dose values")
    candidate = np.asarray(counts, dtype=np.float64)
    if candidate.shape != (candidate_x.size, 2, 2):
        raise ValueError("counts must be nonnegative integers with shape (dose,2,2)")
    x = _real(candidate_x, "dose_codes")
    n = _real(candidate, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("counts must be nonnegative integers with shape (dose,2,2)")
    if n.sum() > 10_000:
        raise ValueError("total patient count must not exceed 10000")
    logp = efftox_log_joint_probabilities(x, parameters)
    contributions = np.zeros_like(logp)
    np.multiply(n, logp, out=contributions, where=n > 0)
    result = np.sum(contributions, axis=(-1, -2, -3))
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ArithmeticError("EffTox log likelihood is invalid")
    return _owned(result)


def efftox_predict(dose_codes: ArrayLike, parameters: ArrayLike) -> FloatArray:
    """Return joint cell probabilities for coefficient draws and dose codes."""
    result = np.exp(efftox_log_joint_probabilities(dose_codes, parameters))
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("EffTox predicted probabilities are nonfinite")
    return _owned(result)


@dataclass(frozen=True)
class EffToxFit:
    """Posterior draws and dosewise outcome probabilities, with chain axes."""

    doses: FloatArray
    dose_codes: FloatArray
    counts: FloatArray
    parameters: FloatArray
    joint_probabilities: FloatArray
    efficacy_probabilities: FloatArray
    toxicity_probabilities: FloatArray
    efficacy_logits: FloatArray
    toxicity_logits: FloatArray
    log_likelihood: FloatArray
    summary: ChainSummary
    warmup: int
    likelihood_evaluations: int


def _sample_prior(
    prior: EffToxPrior, size: tuple[int, ...], rng: np.random.Generator
) -> FloatArray:
    values = prior.mean + prior.sd * rng.normal(size=(*size, 6))
    if prior.monotone_toxicity:
        sd = float(prior.sd[1])
        if sd == 0:
            values[..., 1] = prior.mean[1]
        else:
            values[..., 1] = truncnorm.rvs(
                -prior.mean[1] / sd,
                np.inf,
                loc=prior.mean[1],
                scale=sd,
                size=size,
                random_state=rng,
            )
    if np.any(np.abs(values) > 1e4) or not np.all(np.isfinite(values)):
        raise ArithmeticError("Gaussian prior draws exceed supported parameter range")
    if prior.monotone_toxicity and np.any(values[..., 1] <= 0):
        raise ArithmeticError("truncated-normal beta_T draws must be positive")
    return values


def _ellipse_step(
    state: FloatArray,
    prior: EffToxPrior,
    counts: FloatArray,
    dose_codes: FloatArray,
    current_ll: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, float, int]:
    centered = state - prior.mean
    direction = prior.sd * rng.normal(size=6)
    height = current_ll + np.log1p(-rng.random())
    angle = rng.uniform(0.0, 2.0 * np.pi)
    lower, upper = angle - 2.0 * np.pi, angle
    evaluations = 0
    for _ in range(1000):
        proposal = prior.mean + centered * np.cos(angle) + direction * np.sin(angle)
        if prior.monotone_toxicity and proposal[1] <= 0:
            ll = -np.inf
        else:
            ll = float(efftox_log_likelihood(dose_codes, counts, proposal))
            evaluations += 1
        if ll >= height:
            return proposal, ll, evaluations
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise ArithmeticError("EffTox elliptical-slice bracket failed to accept a draw")


def fit_efftox(
    doses: ArrayLike,
    counts: ArrayLike,
    *,
    prior: EffToxPrior,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    zero_dose_shift: bool = True,
    rng: np.random.Generator,
) -> EffToxFit:
    """Fit EffTox by elliptical slice sampling under explicit Gaussian priors.

    With no observed outcomes, draws are sampled directly from the prior. An
    explicit NumPy Generator is required for reproducibility.
    """
    if not isinstance(prior, EffToxPrior):
        raise ValueError("prior must be an EffToxPrior")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    candidate_doses = np.asarray(doses, dtype=np.float64)
    if candidate_doses.ndim != 1 or not 2 <= candidate_doses.size <= 20:
        raise ValueError("doses must be a 1D array of 2 to 20 values")
    raw_doses = _real(candidate_doses, "doses")
    x = efftox_standardize(raw_doses, zero_dose_shift=zero_dose_shift)
    candidate_counts = np.asarray(counts, dtype=np.float64)
    if candidate_counts.shape != (x.size, 2, 2):
        raise ValueError("counts must be nonnegative integers with shape (dose,2,2)")
    n = _real(candidate_counts, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("counts must be nonnegative integers with shape (dose,2,2)")
    if n.sum() > 10_000:
        raise ValueError("total patient count must not exceed 10000")
    draw_count = _integer(draws, "draws", 8, 10_000)
    warm = _integer(warmup, "warmup", 0, 10_000)
    chain_count = _integer(chains, "chains", 2, 4)
    if chain_count * x.size * draw_count > 200_000:
        raise ValueError("retained dose-probability draws exceed 200000 cells")
    if chain_count * x.size * (draw_count + warm) * 6 > 2_000_000:
        raise ValueError("EffTox sampling work exceeds the supported budget")
    if initial is None:
        starts = np.tile(prior.mean, (chain_count, 1))
        if prior.monotone_toxicity and starts[0, 1] <= 0:
            starts[:, 1] = max(float(prior.mean[1]), float(prior.sd[1]), 1e-8)
    else:
        raw_starts = np.asarray(initial, dtype=np.float64)
        if raw_starts.shape != (chain_count, 6):
            raise ValueError("initial must contain one valid six-coefficient row per chain")
        starts = _real(raw_starts, "initial").copy()
        if np.any(np.abs(starts) > 1e4):
            raise ValueError("initial must contain one valid six-coefficient row per chain")
        fixed = prior.sd == 0
        if np.any(starts[:, fixed] != prior.mean[fixed]):
            raise ValueError("initial values for fixed prior coefficients must equal their means")
        if prior.monotone_toxicity and np.any(starts[:, 1] <= 0):
            raise ValueError("initial beta_T values must be positive")

    parameter_draws = np.empty((chain_count, draw_count, 6), dtype=np.float64)
    likelihood_draws = np.empty((chain_count, draw_count), dtype=np.float64)
    evaluations = 0
    if n.sum() == 0:
        parameter_draws = _sample_prior(prior, (chain_count, draw_count), rng)
    else:
        for chain in range(chain_count):
            state = starts[chain].copy()
            ll = float(efftox_log_likelihood(x, n, state))
            for iteration in range(warm + draw_count):
                state, ll, count_eval = _ellipse_step(state, prior, n, x, ll, rng)
                evaluations += count_eval
                if iteration >= warm:
                    index = iteration - warm
                    parameter_draws[chain, index] = state
                    likelihood_draws[chain, index] = ll
    if n.sum() == 0:
        likelihood_draws.fill(0.0)
    joint = efftox_predict(x, parameter_draws)
    efficacy = joint[..., 1, :].sum(axis=-1)
    toxicity = joint[..., :, 1].sum(axis=-1)
    mu_t, beta_t, mu_e, beta_e1, beta_e2 = np.moveaxis(parameter_draws, -1, 0)[:5]
    eta_t = mu_t[..., None] + beta_t[..., None] * x
    eta_e = mu_e[..., None] + beta_e1[..., None] * x + beta_e2[..., None] * x**2
    return EffToxFit(
        _owned(raw_doses),
        x,
        _owned(n),
        _owned(parameter_draws),
        joint,
        _owned(efficacy),
        _owned(toxicity),
        _owned(eta_e),
        _owned(eta_t),
        _owned(likelihood_draws),
        summarize_chains(parameter_draws),
        warm,
        evaluations,
    )
