"""Six-dose response model and observed-outcome snapshots from P12Xuelin C++."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaincc, expit

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .hierarchical_binomial import ChainSummary, summarize_chains

# Preserve the coordinates and prior SD as written in SetupTrial, not rounded
# raw dose labels or the conflicting trial-description observation windows.
_S = 0.7071067811865
_DOSES = np.array(
    [
        [0, -_S, -_S, -1],
        [0, -_S, -_S, 1],
        [-_S, _S, 0, -1],
        [-_S, _S, 0, 1],
        [-_S, 0, _S, -1],
        [-_S, 0, _S, 1],
    ]
)
_MEAN = (6.2445, 2.0815, 2.0815, 0.0)
_SD = (3.16227766,) * 4


def _tally(value: ArrayLike) -> FloatArray:
    x = count(value, "tally")
    if x.shape != (6, 4) or np.any(x.sum(axis=1) >= 2**53):
        raise ValueError("tally must be (6,4), with per-dose totals smaller than 2**53")
    return x


def phase12_response_probabilities(coefficients: ArrayLike) -> FloatArray:
    """Return (..., six doses) logistic probabilities in the native dose order."""
    b = finite(coefficients, "coefficients")
    if b.ndim < 1 or b.shape[-1] != 4:
        raise ValueError("coefficients must end in four model parameters")
    with np.errstate(over="ignore", invalid="ignore"):
        eta = b @ _DOSES.T
    if not np.all(np.isfinite(eta)):
        raise ArithmeticError("linear predictors exceed floating-point range")
    return _freeze(expit(eta))


def phase12_response_loglikelihood(coefficients: ArrayLike, tally: ArrayLike) -> FloatArray:
    """Binomial log likelihood without combinatorial constants; pending outcomes omitted.

    tally columns are no response, response, no toxicity, toxicity. Only the
    first two columns contribute to the shared response regression likelihood.
    """
    b, x = finite(coefficients, "coefficients"), _tally(tally)
    if b.ndim < 1 or b.shape[-1] != 4:
        raise ValueError("coefficients must end in four model parameters")
    with np.errstate(over="ignore", invalid="ignore"):
        eta = b @ _DOSES.T
        result = -np.sum(x[:, 1] * np.logaddexp(0, -eta) + x[:, 0] * np.logaddexp(0, eta), axis=-1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("response log likelihood exceeds floating-point range")
    return _freeze(result)


@dataclass(frozen=True)
class Phase12Snapshot:
    tally: FloatArray
    enrolled: FloatArray
    pending_efficacy: FloatArray
    pending_toxicity: FloatArray
    time: float


def phase12_snapshot(records: ArrayLike, *, time: float) -> Phase12Snapshot:
    """Snapshot rows (dose, entry, response, response_time, toxicity, toxicity_time).

    Times are absolute in a common user-selected unit. Binary outcomes become
    available at time <= analysis time, independently for the two endpoints.
    Future entrants contribute nothing. This does not perform trial allocation.
    """
    x = finite(records, "records")
    if x.size == 0:
        x = np.empty((0, 6))
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("records must have six columns")
    if np.any(x[:, 0] != np.floor(x[:, 0])) or np.any((x[:, 0] < 0) | (x[:, 0] > 5)):
        raise ValueError("doses must be integers from 0 to 5")
    if np.any((x[:, [2, 4]] != 0) & (x[:, [2, 4]] != 1)):
        raise ValueError("response and toxicity must be binary")
    if np.any(x[:, 1] < 0) or np.any(x[:, 3] < x[:, 1]) or np.any(x[:, 5] < x[:, 1]):
        raise ValueError("entry must be nonnegative and outcome times cannot precede entry")
    t = scalar(time, "time")
    if t < 0:
        raise ValueError("time must be nonnegative")
    active = x[x[:, 1] <= t]
    arm = active[:, 0].astype(int)
    enrolled = np.bincount(arm, minlength=6)
    tally = np.zeros((6, 4))
    for outcome, column in [(2, 0), (4, 2)]:
        observed = active[:, outcome + 1] <= t
        np.add.at(tally, (arm[observed], column + active[observed, outcome].astype(int)), 1)
    return Phase12Snapshot(
        _freeze(tally),
        _freeze(enrolled),
        _freeze(enrolled - tally[:, :2].sum(axis=1)),
        _freeze(enrolled - tally[:, 2:].sum(axis=1)),
        t,
    )


@dataclass(frozen=True)
class Phase12ModelFit:
    coefficients: FloatArray
    response_probability: FloatArray
    coefficient_summary: ChainSummary
    response_summary: ChainSummary
    reference_superiority: FloatArray
    efficacy_probability: FloatArray
    future_probability: FloatArray
    pairwise_superiority: FloatArray
    toxicity_probability: FloatArray
    likelihood_evaluations: int
    warmup: int


def fit_phase12_model(
    tally: ArrayLike,
    *,
    prior_mean: ArrayLike = _MEAN,
    prior_sd: ArrayLike = _SD,
    efficacy_target: float = 0.30,
    future_target: float = 0.10,
    toxicity_target: float = 0.33,
    toxicity_prior: ArrayLike = (0.1, 0.9),
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> Phase12ModelFit:
    """Fit the C++ six-dose model using elliptical-slice normal-prior sampling.

    This samples the published normal-prior logistic posterior directly; it does
    not reproduce the executable's adaptive mixture importance sampler. Retained
    chains and diagnostics allow assessment of Monte Carlo precision.
    """
    x = _tally(tally)
    mu, sd = finite(prior_mean, "prior_mean"), finite(prior_sd, "prior_sd")
    if mu.shape != (4,) or sd.shape != (4,) or np.any(sd <= 0):
        raise ValueError("prior_mean and positive prior_sd must have four entries")
    targets = finite([efficacy_target, future_target, toxicity_target], "targets")
    if np.any((targets <= 0) | (targets >= 1)):
        raise ValueError("targets must be in (0,1)")
    prior = finite(toxicity_prior, "toxicity_prior")
    if prior.shape != (2,) or np.any(prior <= 0) or not np.isfinite(prior.sum()):
        raise ValueError("toxicity_prior must contain two positive shapes with finite sum")
    settings = finite([draws, warmup, chains], "sampler settings")
    if np.any(settings != np.floor(settings)) or not (
        8 <= draws <= 100000 and 0 <= warmup <= 100000 and 2 <= chains <= 16
    ):
        raise ValueError("require draws 8..100000, warmup 0..100000 and chains 2..16")
    draws, warmup, chains = int(draws), int(warmup), int(chains)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    samples = np.empty((chains, draws, 4))
    evaluations = 0
    failure, success = x[:, 0], x[:, 1]

    def loglik(b: FloatArray) -> float:
        nonlocal evaluations
        evaluations += 1
        with np.errstate(over="ignore", invalid="ignore"):
            eta = b @ _DOSES.T
            value = -float(np.sum(success * np.logaddexp(0, -eta) + failure * np.logaddexp(0, eta)))
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite posterior likelihood evaluation")
        return value

    if not np.any(x[:, :2]):
        samples[:] = rng.normal(mu, sd, size=samples.shape)
    else:
        for chain in range(chains):
            current = rng.normal(mu, sd)
            current_ll = loglik(current)
            for iteration in range(warmup + draws):
                direction = rng.normal(size=4) * sd
                threshold = current_ll + np.log(max(rng.random(), np.finfo(float).tiny))
                angle = rng.uniform(0, 2 * np.pi)
                lower, upper = angle - 2 * np.pi, angle
                centered = current - mu
                for _ in range(1000):
                    proposal = mu + centered * np.cos(angle) + direction * np.sin(angle)
                    proposal_ll = loglik(proposal)
                    if proposal_ll >= threshold:
                        current, current_ll = proposal, proposal_ll
                        break
                    if angle < 0:
                        lower = angle
                    else:
                        upper = angle
                    angle = rng.uniform(lower, upper)
                else:
                    raise ArithmeticError("elliptical-slice bracket failed to accept a draw")
                if iteration >= warmup:
                    samples[chain, iteration - warmup] = current
    eta = samples @ _DOSES.T
    if not np.all(np.isfinite(eta)):
        raise ArithmeticError("retained predictors exceed floating-point range")
    p = expit(eta)
    # Compare logits to avoid false ties from expit rounding to 0 or 1.
    pairwise = (eta[..., :, None] > eta[..., None, :]).mean(axis=(0, 1))
    reference = pairwise[:, 0].copy()
    reference[0] = 0.5
    efficacy = (eta >= np.log(targets[0]) - np.log1p(-targets[0])).mean(axis=(0, 1))
    future = (eta > np.log(targets[1]) - np.log1p(-targets[1])).mean(axis=(0, 1))
    toxicity = betaincc(prior[0] + x[:, 3], prior[1] + x[:, 2], targets[2])
    return Phase12ModelFit(
        _freeze(samples),
        _freeze(p),
        summarize_chains(samples),
        summarize_chains(p),
        _freeze(reference),
        _freeze(efficacy),
        _freeze(future),
        _freeze(pairwise),
        _freeze(toxicity),
        evaluations,
        warmup,
    )
