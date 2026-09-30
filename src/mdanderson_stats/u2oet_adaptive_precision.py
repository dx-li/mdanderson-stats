"""Bounded within-chain utility-precision monitoring for U2OET fits.

The native guide specifies four corner utilities, monitored separately within
each chain, and stops extension when batch-means MCSE / posterior SD is small
enough. Batch length, extension schedule, and a finite cap are Python policies.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .u2oet import _real, u2oet_standardize
from .u2oet_decision import _integer
from .u2oet_fit import (
    U2OETFit,
    _joint,
    _marginal,
    _parameters,
    fit_u2oet,
    u2oet_parameter_names,
)

_MAX_RETAINED_JOINT_CELLS = 4_000_000
_MAX_LIVE_CELLS = 12_000_000
_MAX_WORK = 20_000_000_000


@dataclass(frozen=True)
class U2OETAdaptivePrecisionResult:
    """Adaptive fit and per-chain MC precision for the four grid corners."""

    fit: U2OETFit
    corner_indices: tuple[tuple[int, int], ...]
    posterior_sd: FloatArray
    mcse: FloatArray
    mcse_ratio: FloatArray
    corner_split_rhat: FloatArray
    target_mcse_ratio: float
    target_met: bool
    termination: str
    draws_per_chain: int
    max_draws_per_chain: int
    batch_draws: int


def _corner_values(joint: FloatArray, utility: FloatArray) -> FloatArray:
    """Return chain-by-draw corner utilities without a dose-surface temporary."""
    n1, n2, ne, nt = joint.shape[2:]
    corners = ((0, 0), (0, n2 - 1), (n1 - 1, 0), (n1 - 1, n2 - 1))
    scale = float(np.max(utility))
    if scale == 0:
        return np.zeros((joint.shape[0], joint.shape[1], 4))
    if np.all(utility == utility.flat[0]):
        return np.full((joint.shape[0], joint.shape[1], 4), utility.flat[0])
    scaled_utility = utility / scale
    values = np.empty((joint.shape[0], joint.shape[1], 4))
    with np.errstate(over="ignore", invalid="ignore"):
        for index, (dose1, dose2) in enumerate(corners):
            values[..., index] = (
                np.sum(joint[:, :, dose1, dose2, :, :] * scaled_utility, axis=(-2, -1)) * scale
            )
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("corner utility exceeds floating-point range")
    return values


def _diagnostics(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Nonoverlapping batch-means MCSE and SD, separately for each chain/corner."""
    draws = values.shape[1]
    batch = max(2, int(np.sqrt(draws)))
    batches = draws // batch
    if batches < 2:
        raise ValueError("at least four batch means are required for precision monitoring")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    normalized = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    normalized_batches = (
        normalized[:, : batches * batch]
        .reshape(values.shape[0], batches, batch, values.shape[2])
        .mean(axis=2)
    )
    normalized_sd = normalized.std(axis=1, ddof=1)
    normalized_mcse = normalized_batches.std(axis=1, ddof=1) / np.sqrt(batches)
    ratio = np.full_like(normalized_sd, np.nan)
    positive = normalized_sd > 0
    ratio[positive] = normalized_mcse[positive] / normalized_sd[positive]
    ratio[(~positive) & (normalized_mcse > 0)] = np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        sd = normalized_sd * scale[:, 0]
        mcse = normalized_mcse * scale[:, 0]
    if not np.all(np.isfinite(sd)) or not np.all(np.isfinite(mcse)):
        raise ArithmeticError("corner utility dispersion exceeds floating-point range")
    return sd, mcse, ratio


def _validate_inputs(
    doses1: ArrayLike,
    doses2: ArrayLike,
    counts: ArrayLike,
    toxicity_only: ArrayLike | None,
    utility: ArrayLike,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    model: str,
    centering: str,
    initial: ArrayLike | None,
    coordinate_updates: bool,
    chains: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray, FloatArray]:
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    n = _real(counts, "counts")
    if (
        n.ndim != 4
        or n.shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in n.shape[2:])
        or np.any(n < 0)
        or np.any(n != np.floor(n))
        or n.sum() >= 2**53
    ):
        raise ValueError("counts must be integer agent1-by-agent2-by-efficacy-by-toxicity cells")
    nt = None if toxicity_only is None else _real(toxicity_only, "toxicity_only")
    if nt is not None and (
        nt.shape != (*n.shape[:2], n.shape[-1])
        or np.any(nt < 0)
        or np.any(nt != np.floor(nt))
        or n.sum() + nt.sum() >= 2**53
    ):
        raise ValueError("toxicity_only must be integer dose-by-dose-by-toxicity counts")
    u = _real(utility, "utility")
    if u.shape != n.shape[2:] or np.any(u < 0):
        raise ValueError("utility must be a nonnegative efficacy-by-toxicity matrix")
    names = u2oet_parameter_names(n.shape[2], n.shape[3], model=model)
    if centering not in ("log", "linear") or (model == "cmi" and centering != "log"):
        raise ValueError("invalid centering; CMI requires log")
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    dimension = len(names) - 1
    if mu.shape != (dimension,) or sd.shape != mu.shape or np.any(sd <= 0):
        raise ValueError(
            "prior arrays must match named coordinates excluding association; SDs positive"
        )
    if not isinstance(coordinate_updates, (bool, np.bool_)):
        raise ValueError("coordinate_updates must be boolean")
    names = u2oet_parameter_names(n.shape[2], n.shape[3], model=model)
    dimension = len(names) - 1
    if initial is not None:
        start = _real(initial, "initial")
        if start.shape != (chains, dimension + 1):
            raise ValueError("initial must have one full coordinate row per chain")
        if np.any(np.abs(start[:, -1]) > 1):
            raise ValueError("initial association must lie in [-1,1]")
    else:
        start = np.tile(np.r_[mu, 0.0], (chains, 1))
        for index, name in enumerate(names[:-1]):
            if ".slope." in name:
                start[:, index] = max(mu[index], sd[index])
    split = next(i for i, name in enumerate(names) if name.startswith("toxicity."))
    observed = n > 0
    observed_toxicity = None if nt is None else nt > 0
    for row in start:
        e = _parameters(row[:split], n.shape[2], model)
        t = _parameters(row[split:dimension], n.shape[3], model)
        if e is None or t is None:
            raise ValueError("initial slopes must be positive")
        log_e = _marginal(d1, d2, e, centering)
        log_t = _marginal(d1, d2, t, centering)
        log_joint = _joint(log_e, log_t, float(row[-1]))
        likelihood = float(np.sum(n[observed] * log_joint[observed]))
        if nt is not None and observed_toxicity is not None:
            likelihood += float(np.sum(nt[observed_toxicity] * log_t[observed_toxicity]))
        if not np.isfinite(likelihood):
            raise ValueError("every initial chain state must have finite likelihood")
    return d1, d2, n, nt, u, start


def fit_u2oet_adaptive_precision(
    doses1: ArrayLike,
    doses2: ArrayLike,
    counts: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    utility: ArrayLike,
    target_mcse_ratio: float,
    max_draws_per_chain: int,
    initial_draws: int,
    batch_draws: int = 256,
    toxicity_only: ArrayLike | None = None,
    model: str = "pds",
    centering: str = "log",
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    coordinate_updates: bool = False,
    max_work: int = _MAX_WORK,
    rng: np.random.Generator,
) -> U2OETAdaptivePrecisionResult:
    """Continue U2OET chains until four within-chain corner precision checks pass.

    The guide allows targets in [.001, .05]. ``initial_draws`` includes the
    guide's minimum of at least ``warmup`` retained draws. Python uses
    nonoverlapping batch means with batch length floor(sqrt(draws)) (minimum 2),
    and appends batches of at most ``batch_draws`` until the target or cap.
    Warmup occurs once. Every later fit call starts from the previous retained
    last state for each chain; no parameter state or adaptation is reset.

    This wrapper supports the PDS, CMI and PDS+CMI models accepted by
    ``fit_u2oet``. GAO is a separate sampler and is not covered here. The
    existing sampler supports 2–16 chains, narrower than the native guide's
    1–20 range. Chunk boundaries consume RNG sequentially, so results are not
    promised to match a single fixed-budget call with the same seed.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    target = _real(target_mcse_ratio, "target_mcse_ratio")
    if target.ndim != 0 or not 0.001 <= float(target) <= 0.05:
        raise ValueError("target_mcse_ratio must lie in [.001, .05]")
    target_mcse_ratio = float(target)
    initial_draws = _integer(initial_draws, "initial_draws", 8, 100_000)
    max_draws_per_chain = _integer(max_draws_per_chain, "max_draws_per_chain", 8, 100_000)
    batch_draws = _integer(batch_draws, "batch_draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 10_000)
    chains = _integer(chains, "chains", 2, 16)
    if initial_draws < warmup:
        raise ValueError("initial_draws must be at least warmup")
    if initial_draws > max_draws_per_chain:
        raise ValueError("initial_draws must not exceed max_draws_per_chain")
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK)
    d1, d2, n, nt, u, start = _validate_inputs(
        doses1,
        doses2,
        counts,
        toxicity_only,
        utility,
        prior_mean,
        prior_sd,
        model,
        centering,
        initial,
        coordinate_updates,
        chains,
    )
    names = u2oet_parameter_names(n.shape[2], n.shape[3], model=model)
    dimension = len(names) - 1
    groups = 2 + (dimension if coordinate_updates else 0)
    schedule: list[int] = []
    remaining = max_draws_per_chain - initial_draws
    while remaining >= 8:
        size = min(batch_draws, remaining)
        tail = remaining - size
        if 0 < tail < 8:
            size += tail
        schedule.append(size)
        remaining -= size
    effective_cap = initial_draws + sum(schedule)
    iterations = warmup + effective_cap
    work = chains * iterations * (groups * 1000 + 3) * n.size
    chunks = 1 + len(schedule)
    work += chunks * chains * n.size
    work += chunks * chains * effective_cap * 4
    if work > max_work:
        raise ValueError("worst-case adaptive sampler work exceeds max_work")
    retained_cells = chains * effective_cap * n.size
    if retained_cells > _MAX_RETAINED_JOINT_CELLS:
        raise ValueError("adaptive retained posterior exceeds the 4-million-cell memory cap")
    live_cells = (
        chains
        * effective_cap
        * (
            3 * (n.size + dimension + 2)  # retained chunks plus concatenate/freeze copies
            + 4  # retained corner utility traces
            + 16  # normalized traces, batch means, and dispersion scratch
            + 8  # split-Rhat input, sort, interval, split, and batch scratch
        )
    )
    if live_cells > _MAX_LIVE_CELLS:
        raise ValueError("adaptive retained and temporary arrays exceed the live-cell memory cap")

    parameter_parts: list[FloatArray] = []
    joint_parts: list[FloatArray] = []
    likelihood_parts: list[FloatArray] = []
    utility_values = np.empty((chains, effective_cap, 4))
    acceptance_weighted = np.zeros(chains)
    evaluations = 0
    retained = 0
    warmup_fit = warmup
    current_initial = initial
    pending_sizes = iter(schedule)
    while retained < effective_cap:
        requested = initial_draws if retained == 0 else next(pending_sizes)
        batch = fit_u2oet(
            d1,
            d2,
            n,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            toxicity_only=nt,
            model=model,
            centering=centering,
            draws=requested,
            warmup=warmup_fit,
            chains=chains,
            initial=current_initial,
            coordinate_updates=coordinate_updates,
            rng=rng,
        )
        parameter_parts.append(batch.parameters)
        joint_parts.append(batch.joint)
        likelihood_parts.append(batch.log_likelihood)
        utility_values[:, retained : retained + requested] = _corner_values(batch.joint, u)
        acceptance_weighted += np.asarray(batch.association_acceptance) * requested
        evaluations += batch.likelihood_evaluations
        retained += requested
        current_initial = batch.parameters[:, -1, :]
        warmup_fit = 0
        del batch
        utilities = utility_values[:, :retained]
        sd, mcse, ratio = _diagnostics(utilities)
        if np.all(np.isfinite(ratio)) and np.all(ratio <= target_mcse_ratio):
            break

    all_parameters = _freeze(np.concatenate(parameter_parts, axis=1))
    all_joint = _freeze(np.concatenate(joint_parts, axis=1))
    all_likelihood = _freeze(np.concatenate(likelihood_parts, axis=1))
    final_fit = U2OETFit(
        names=names,
        parameters=all_parameters,
        joint=all_joint,
        log_likelihood=all_likelihood,
        association_acceptance=_freeze(acceptance_weighted / retained),
        likelihood_evaluations=evaluations,
        warmup=warmup,
        model=model,
        coordinate_updates=bool(coordinate_updates),
    )
    corner_split_rhat = np.empty(4)
    for corner in range(4):
        scale = float(np.max(np.abs(utility_values[:, :retained, corner])))
        normalized = (
            utility_values[:, :retained, corner] / scale
            if scale > 0
            else np.zeros((chains, retained))
        )
        corner_split_rhat[corner] = summarize_chains(normalized).split_rhat
    corners = ((0, 0), (0, n.shape[1] - 1), (n.shape[0] - 1, 0), (n.shape[0] - 1, n.shape[1] - 1))
    met = bool(np.all(np.isfinite(ratio)) and np.all(ratio <= target_mcse_ratio))
    return U2OETAdaptivePrecisionResult(
        fit=final_fit,
        corner_indices=corners,
        posterior_sd=_freeze(sd),
        mcse=_freeze(mcse),
        mcse_ratio=_freeze(ratio),
        corner_split_rhat=_freeze(corner_split_rhat),
        target_mcse_ratio=float(target_mcse_ratio),
        target_met=met,
        termination="target_met" if met else "draw_cap",
        draws_per_chain=retained,
        max_draws_per_chain=max_draws_per_chain,
        batch_draws=batch_draws,
    )
