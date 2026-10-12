"""Bounded continuous-chain UAROET inference with per-chain utility precision."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _integer, _real_matrix, uaroet_parameter_names
from .uaroet_desktop_study import UAROETDesktopStudy
from .uaroet_fit import UAROETFit, _positive_mask, _summarize_uaroet_chains, fit_uaroet


@dataclass(frozen=True)
class UAROETPrecisionFit:
    """Posterior averages and retained utilities; precision is not clinical validation."""

    mean_joint: FloatArray
    mean_parameters: FloatArray
    mean_association: float
    utility_samples: FloatArray
    utility_mcse_ratio: FloatArray
    utility_split_rhat: FloatArray
    probability_best: FloatArray
    toxicity_exceedance: FloatArray
    efficacy_shortfall: FloatArray
    monitored_doses: tuple[int, ...]
    target: float
    precision_met: bool
    draws_per_chain: int
    work_evaluations: int


def _utility_precision(samples: FloatArray) -> FloatArray:
    """Independent batch-means MCSE/SD for every chain and dose."""
    chains, draws, doses = samples.shape
    size = max(2, int(np.sqrt(draws)))
    batches = draws // size
    means = samples[:, : batches * size].reshape(chains, batches, size, doses).mean(2)
    mcse = means.std(axis=1, ddof=1) / np.sqrt(batches)
    sd = samples.std(axis=1, ddof=1)
    # A stuck chain is not evidence of precision, even if every batch agrees.
    ratio = np.full((chains, doses), np.inf)
    np.divide(mcse, sd, out=ratio, where=sd > 0)
    return ratio


def fit_uaroet_desktop(
    study: UAROETDesktopStudy,
    *,
    rng: np.random.Generator,
    target: float | None = None,
    conduct: bool = True,
    chunk_draws: int = 1000,
    max_draws: int = 100_000,
    max_evaluations: int = 2_000_000,
    initial_parameters: ArrayLike | None = None,
    initial_association: ArrayLike | None = None,
    on_chunk: Callable[[int, int, UAROETFit], None] | None = None,
) -> UAROETPrecisionFit:
    """Continue each posterior chain until its monitored utility ratios pass.

    Uses the study burn-in once per chain and then resumes from each chain's
    final parameter and correlation state. Joint distributions are accumulated
    online; only utilities are retained across chunks. The lowest, upper-middle
    and highest doses are monitored. A draw-limit result can have precision_met
    False and must not drive a precision-qualified recommendation. Work-budget
    exhaustion raises from the sampler. Native RNG and exact batch partition
    equivalence are not claimed. Between-chain R-hat is reported separately.
    Optional on_chunk(chain_index, first_draw_index, fit) receives read-only
    chunks for streaming distributions; indices are zero-based. Callback errors
    propagate, and partial output must not be treated as an accepted fit.
    """
    if not isinstance(study, UAROETDesktopStudy) or not isinstance(rng, np.random.Generator):
        raise ValueError("require a UAROETDesktopStudy and explicit NumPy Generator")
    if not isinstance(conduct, (bool, np.bool_)):
        raise ValueError("conduct must be boolean")
    if on_chunk is not None and not callable(on_chunk):
        raise ValueError("on_chunk must be callable")
    settings = study.simulation_parameters
    if "TargetMCSERatio_Sim" in settings and not conduct and target is None:
        raise ValueError("supply target explicitly when TargetMCSERatio_Sim is present")
    if target is None:
        target = settings.get(
            "TargetMCSERatio_Treat" if conduct else "TargetMCSERatio", 0.005 if conduct else 0.03
        )
    scalar = _real_matrix(target, "target")
    if scalar.ndim != 0 or not 0 < float(scalar) < 1:
        raise ValueError("target must be strictly between zero and one")
    target = float(scalar)
    chunk = _integer(chunk_draws, "chunk_draws", 8, 100_000)
    limit = _integer(max_draws, "max_draws", 8, 1_000_000)
    budget = _integer(max_evaluations, "max_evaluations", 1, 128_000_000)
    chains = int(settings.get("NumChains", 4))
    warmup = int(settings.get("BurnInSamples", 1000))
    minimum = max(8, warmup)
    if limit < minimum or chunk > limit:
        raise ValueError("max_draws must cover burn-in length and chunk_draws")
    if chains * limit * study.dose_count > 20_000_000:
        raise ValueError("retained utility budget exceeds twenty million values")
    counts, partial, _ = study.outcome_counts()
    observed = bool(np.any(counts) or np.any(partial))
    starts = (
        None
        if initial_parameters is None
        else _real_matrix(initial_parameters, "initial_parameters")
    )
    if starts is not None and starts.shape == study.prior_mean.shape:
        starts = np.broadcast_to(starts, (chains, starts.size)).copy()
    if starts is not None and starts.shape != (chains, study.prior_mean.size):
        raise ValueError("initial_parameters must match chains and parameters")
    rho = (
        None
        if initial_association is None
        else _real_matrix(initial_association, "initial_association")
    )
    if rho is not None and rho.ndim == 0:
        rho = np.full(chains, float(rho))
    if rho is not None and (rho.shape != (chains,) or np.any(np.abs(rho) >= 1)):
        raise ValueError("initial_association must be interior and match chains")
    if not observed and (starts is not None or rho is not None):
        raise ValueError("initial states require observed data")
    names = uaroet_parameter_names(study.dose_count, study.efficacy_levels, study.toxicity_levels)
    if starts is not None and np.any(starts[:, _positive_mask(names)] < 0):
        raise ValueError("initial increments must be nonnegative")
    if chunk * counts.size > 2_000_000:
        raise ValueError("chunk joint posterior exceeds two million cells")
    # Separate streams preserve each chain's transition sequence across chunks.
    streams = [np.random.default_rng(s) for s in rng.integers(0, 2**63, size=chains)]
    utility = np.empty((chains, limit, study.dose_count))
    joint_sum = np.zeros_like(counts, dtype=float)
    parameter_sum = np.zeros_like(study.prior_mean)
    rho_sum = 0.0
    best = np.zeros(study.dose_count)
    toxic = np.zeros(study.dose_count)
    futile = np.zeros(study.dose_count)
    states: list[tuple[FloatArray, float] | None] = [None] * chains
    n = work = 0
    monitored = tuple(sorted({0, study.dose_count // 2, study.dose_count - 1}))
    p = study.trial_parameters
    bad = int(p["LowestUnacceptableToxLevel"])
    good = int(p["LowestAcceptableEffLevel"])
    met = False
    while n < limit:
        take = min(chunk, limit - n)
        if take < 8:
            break
        for c, stream in enumerate(streams):
            previous = states[c]
            fit = fit_uaroet(
                counts,
                toxicity_only_counts=partial,
                prior_mean=study.prior_mean,
                prior_sd=study.prior_sd,
                initial_parameters=(
                    previous[0]
                    if previous is not None and observed
                    else (None if starts is None else starts[c])
                ),
                initial_association=(
                    previous[1]
                    if previous is not None and observed
                    else (None if rho is None else float(rho[c]))
                ),
                chains=1,
                draws=take,
                warmup=warmup if n == 0 else 0,
                rng=stream,
                max_evaluations=budget - work,
            )
            states[c] = (fit.parameters[0, -1].copy(), float(fit.association[0, -1]))
            work += fit.work_evaluations
            if on_chunk is not None:
                on_chunk(c, n, fit)
            j = fit.joint[0]
            u = np.einsum("ndet,et->nd", j, study.utility)
            utility[c, n : n + take] = u
            joint_sum += j.sum(0)
            parameter_sum += fit.parameters[0].sum(0)
            rho_sum += float(fit.association.sum())
            best += np.bincount(np.argmax(u, axis=1), minlength=study.dose_count)
            toxic += (j[..., bad:].sum((-2, -1)) > p["MaxProbTox"]).sum(0)
            futile += (j[:, :, good:, :].sum((-2, -1)) < p["MinProbEff"]).sum(0)
        n += take
        ratios = _utility_precision(utility[:, :n])
        if np.ptp(study.utility) == 0:
            ratios[:] = 0  # Analytically constant utility, independent of chain mixing.
        met = n >= minimum and bool(np.all(ratios[:, monitored] <= target))
        if met:
            break
    samples = utility[:, :n]
    total = chains * n
    return UAROETPrecisionFit(
        _freeze(joint_sum / total),
        _freeze(parameter_sum / total),
        rho_sum / total,
        _freeze(samples.copy()),
        _freeze(ratios),
        _summarize_uaroet_chains(samples).split_rhat,
        _freeze(best / total),
        _freeze(toxic / total),
        _freeze(futile / total),
        monitored,
        target,
        met,
        n,
        work,
    )
