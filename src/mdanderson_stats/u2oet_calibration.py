"""U2OET pseudo-trial averaging for prior-center elicitation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .u2oet import _real, u2oet_standardize
from .u2oet_decision import _integer
from .u2oet_fit import fit_u2oet, u2oet_parameter_names


@dataclass(frozen=True)
class U2OETCalibration:
    """Pseudo-posterior means in Python coordinates, excluding association.

    Between-trial standard errors include pseudo-data variation and MCMC noise.
    Inspect every trial's diagnostics before treating the centers as calibrated.
    """

    names: tuple[str, ...]
    prior_mean: FloatArray
    trial_means: FloatArray
    standard_error: FloatArray
    trial_mcse: FloatArray
    trial_split_rhat: FloatArray
    counts: FloatArray
    patients_per_pair: int


def calibrate_u2oet_prior(
    doses1: ArrayLike,
    doses2: ArrayLike,
    scenario_joint: ArrayLike,
    *,
    repetitions: int,
    patients_per_pair: int = 100,
    pseudo_prior_sd: float = 100.0,
    model: str = "pds",
    centering: str = "log",
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    rng: np.random.Generator,
) -> U2OETCalibration:
    """Guide section 1.3: balanced pseudo data, diffuse zero-mean prior, averaging.

    The guide recommends >=1000 pseudo trials. Repetitions are explicit to make
    the potentially substantial computation visible. No convergence threshold
    is silently waived or used to discard difficult pseudo trials.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    p = _real(scenario_joint, "scenario_joint")
    if (
        p.ndim != 4
        or p.shape[:2] != (d1.size, d2.size)
        or any(not 2 <= n <= 4 for n in p.shape[2:])
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(
            "scenario must be an agent1-by-agent2-by-efficacy-by-toxicity probability grid"
        )
    total = p.sum(axis=(-2, -1), keepdims=True)
    if np.any(np.abs(total - 1) > 1e-12):
        raise ValueError("scenario probabilities must sum to one at every dose pair")
    p = p / total  # Only remove admitted floating-point normalization error.
    repetitions = _integer(repetitions, "repetitions", 2, 100_000)
    patients_per_pair = _integer(patients_per_pair, "patients_per_pair", 1, 1_000_000)
    scale = _real(pseudo_prior_sd, "pseudo_prior_sd")
    if scale.ndim or scale <= 0:
        raise ValueError("pseudo_prior_sd must be a positive scalar")
    names = u2oet_parameter_names(p.shape[2], p.shape[3], model=model)[:-1]
    if centering not in ("log", "linear") or (model == "cmi" and centering != "log"):
        raise ValueError("invalid centering; CMI requires log")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    if draws * chains * p.size > 20_000_000:
        raise ValueError("retained joint draws per pseudo trial exceed 20 million cells")
    if repetitions * p.size > 20_000_000:
        raise ValueError("retained pseudo-trial counts exceed 20 million cells")
    means = np.empty((repetitions, len(names)))
    mcse, rhat = np.empty_like(means), np.empty_like(means)
    counts = np.empty((repetitions, *p.shape))
    for repetition in range(repetitions):
        for i in range(d1.size):
            for j in range(d2.size):
                counts[repetition, i, j] = rng.multinomial(
                    patients_per_pair, p[i, j].ravel()
                ).reshape(p.shape[2:])
        fit = fit_u2oet(
            d1,
            d2,
            counts[repetition],
            prior_mean=np.zeros(len(names)),
            prior_sd=np.full(len(names), float(scale)),
            model=model,
            centering=centering,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=rng,
        )
        summary = summarize_chains(fit.parameters[..., :-1])
        means[repetition] = summary.mean
        mcse[repetition] = summary.batch_mean_mcse
        rhat[repetition] = summary.split_rhat
    return U2OETCalibration(
        names,
        _freeze(means.mean(axis=0)),
        _freeze(means),
        _freeze(means.std(axis=0, ddof=1) / np.sqrt(repetitions)),
        _freeze(mcse),
        _freeze(rhat),
        _freeze(counts),
        patients_per_pair,
    )
