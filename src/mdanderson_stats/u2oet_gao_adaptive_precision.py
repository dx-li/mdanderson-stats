"""Bounded corner-utility precision monitoring for explicit-prior GAO fits.

The four-corner target follows the U2OET guide. Chunking, batch length and
resource limits are Python policies; native GAO defaults are not inferred.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .u2oet import _real
from .u2oet_adaptive_precision import _corner_values, _diagnostics
from .u2oet_decision import _integer
from .u2oet_gao import _dose_grid
from .u2oet_gao_fit import (
    _MAX_LIKELIHOOD_EVALUATIONS,
    _MAX_WORK_UNITS,
    U2OETGAOFit,
    _input_shape,
    _likelihood,
    fit_u2oet_gao,
    u2oet_gao_parameter_names,
)

_MAX_RETAINED_JOINT_CELLS = 4_000_000
_MAX_LIVE_CELLS = 12_000_000


@dataclass(frozen=True)
class U2OETGAOAdaptivePrecisionResult:
    """Adaptive GAO posterior plus per-chain precision for four corners."""

    fit: U2OETGAOFit
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


def _draw_schedule(initial: int, maximum: int, batch: int) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = maximum - initial
    while remaining >= 8:
        size = min(batch, remaining)
        tail = remaining - size
        if 0 < tail < 8:
            size += tail
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def fit_u2oet_gao_adaptive_precision(
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
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    max_likelihood_evaluations: int = _MAX_LIKELIHOOD_EVALUATIONS,
    max_work: int = _MAX_WORK_UNITS,
    rng: np.random.Generator,
) -> U2OETGAOAdaptivePrecisionResult:
    """Continue explicit-prior GAO chains until corner MCSE/SD targets pass.

    Warmup is performed once. Later chunks resume from each chain's complete
    retained coordinate vector. Four utilities (the dose-grid corners) are
    monitored separately per chain using nonoverlapping batch means with
    length ``max(2, floor(sqrt(draws)))``. A trailing incomplete batch is
    omitted only from MCSE; SD and the returned fit use every draw.
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
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    max_likelihood_evaluations = _integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    if initial_draws < warmup:
        raise ValueError("initial_draws must be at least warmup")
    if initial_draws > max_draws_per_chain:
        raise ValueError("initial_draws must not exceed max_draws_per_chain")

    d1, d2 = _dose_grid(doses1, "doses1"), _dose_grid(doses2, "doses2")
    count_shape = _input_shape(counts, "counts")
    if (
        len(count_shape) != 4
        or count_shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in count_shape[2:])
    ):
        raise ValueError("counts must have dose-by-dose-by-category shape with 2–4 categories")
    n = _real(counts, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)) or n.sum() >= 2**53:
        raise ValueError("counts must be nonnegative integer GAO outcome cells")
    if toxicity_only is None:
        nt = np.zeros((*n.shape[:2], n.shape[-1]))
    else:
        expected = (*n.shape[:2], n.shape[-1])
        if _input_shape(toxicity_only, "toxicity_only") != expected:
            raise ValueError("toxicity_only must have dose-by-dose-by-toxicity shape")
        nt = _real(toxicity_only, "toxicity_only")
    if (
        nt.shape != (*n.shape[:2], n.shape[-1])
        or np.any(nt < 0)
        or np.any(nt != np.floor(nt))
        or n.sum() + nt.sum() >= 2**53
    ):
        raise ValueError("toxicity_only must be nonnegative integer counts")

    e_levels, t_levels = n.shape[-2:]
    names = u2oet_gao_parameter_names(e_levels, t_levels)
    dimension = len(names)
    for value, name in ((prior_mean, "prior_mean"), (prior_sd, "prior_sd")):
        if _input_shape(value, name) != (dimension,):
            raise ValueError(f"{name} must have one value per named GAO coordinate")
    mu, prior_scale = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mu.shape != (dimension,) or prior_scale.shape != mu.shape or np.any(prior_scale < 0):
        raise ValueError("prior arrays must match named GAO coordinates; SDs may be zero")
    if initial is None:
        starts = np.broadcast_to(mu, (chains, dimension)).copy()
    else:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((dimension,), (chains, dimension)):
            raise ValueError("initial must have one coordinate vector or one vector per chain")
        starts = _real(initial, "initial").copy()
        if starts.shape == (dimension,):
            starts = np.broadcast_to(starts, (chains, dimension)).copy()
    if np.any(starts[:, prior_scale == 0] != mu[prior_scale == 0]):
        raise ValueError("initial fixed coordinates must equal their prior means")

    u = _real(utility, "utility")
    if u.shape != n.shape[-2:] or np.any(u < 0):
        raise ValueError("utility must be a nonnegative efficacy-by-toxicity matrix")
    schedule = _draw_schedule(initial_draws, max_draws_per_chain, batch_draws)
    effective_draws = initial_draws + sum(schedule)
    joint_shape = (*n.shape[:2], e_levels, t_levels)
    joint_cells = int(np.prod(joint_shape))
    free = int(np.count_nonzero(prior_scale > 0))
    chunks = 1 + len(schedule)
    iterations = warmup + effective_draws
    # Each chunk starts with one likelihood evaluation per chain. The sampler
    # then needs at least one accepted proposal per free-coordinate sweep.
    # Slice rejection can require more; remaining caps are passed through so
    # that actual work, rather than the 1000-step theoretical ceiling, governs.
    minimum_evaluations = chains * (1 + chunks + (iterations if free else 0))
    minimum_work = minimum_evaluations * joint_cells
    if minimum_evaluations > max_likelihood_evaluations:
        raise ValueError("minimum adaptive GAO evaluations exceed max_likelihood_evaluations")
    if minimum_work > max_work:
        raise ValueError("minimum adaptive GAO work exceeds max_work")
    retained_joint_cells = chains * effective_draws * joint_cells
    largest_chunk = max((initial_draws, *schedule))
    live_cells = chains * (
        effective_draws * (2 * (joint_cells + dimension + 1) + 14 * dimension + 2 + 4 + 16 + 8)
        + largest_chunk * (14 * dimension + 2 * joint_cells + 2)
    )
    if retained_joint_cells > _MAX_RETAINED_JOINT_CELLS:
        raise ValueError("adaptive retained GAO posterior exceeds the 4-million-cell cap")
    if live_cells > _MAX_LIVE_CELLS:
        raise ValueError("adaptive GAO arrays exceed the 12-million-cell live-memory cap")

    # Validate all chain starts without consuming randomness. The aggregate
    # work and memory limits above are checked first.
    for chain, start in enumerate(starts):
        ll, _ = _likelihood(d1, d2, n, nt, start)
        if not np.isfinite(ll):
            raise ValueError(f"initial state has zero GAO likelihood in chain {chain}")

    parameter_parts: list[FloatArray] = []
    joint_parts: list[FloatArray] = []
    likelihood_parts: list[FloatArray] = []
    utility_values = np.empty((chains, effective_draws, 4))
    current_initial: ArrayLike | None = initial
    retained = 0
    evaluations = chains
    work = chains * joint_cells
    remaining_evaluations = max_likelihood_evaluations - evaluations
    remaining_work = max_work - work
    for index, requested in enumerate((initial_draws, *schedule)):
        chunk_warmup = warmup if index == 0 else 0
        chunk_minimum_evaluations = chains * (1 + (chunk_warmup + requested if free else 0))
        chunk_minimum_work = chunk_minimum_evaluations * joint_cells
        if remaining_evaluations < chunk_minimum_evaluations:
            raise ArithmeticError(
                "aggregate GAO likelihood-evaluation budget exhausted before chunk"
            )
        if remaining_work < chunk_minimum_work:
            raise ArithmeticError("aggregate GAO likelihood-work budget exhausted before chunk")
        batch = fit_u2oet_gao(
            d1,
            d2,
            n,
            prior_mean=mu,
            prior_sd=prior_scale,
            toxicity_only=nt,
            draws=requested,
            warmup=chunk_warmup,
            chains=chains,
            initial=current_initial,
            max_likelihood_evaluations=remaining_evaluations,
            max_work=remaining_work,
            rng=rng,
        )
        parameter_parts.append(batch.parameters)
        joint_parts.append(batch.joint)
        likelihood_parts.append(batch.log_likelihood)
        utility_values[:, retained : retained + requested] = _corner_values(batch.joint, u)
        retained += requested
        evaluations += batch.likelihood_evaluations
        work += batch.likelihood_work_units
        remaining_evaluations -= batch.likelihood_evaluations
        remaining_work -= batch.likelihood_work_units
        current_initial = batch.parameters[:, -1, :]
        sd, mcse, ratio = _diagnostics(utility_values[:, :retained])
        if np.all(np.isfinite(ratio)) and np.all(ratio <= target_mcse_ratio):
            break

    parameters = _freeze(np.concatenate(parameter_parts, axis=1))
    joint = _freeze(np.concatenate(joint_parts, axis=1))
    log_likelihood = _freeze(np.concatenate(likelihood_parts, axis=1))
    fit = U2OETGAOFit(
        names=names,
        parameters=parameters,
        joint=joint,
        log_likelihood=log_likelihood,
        parameter_summary=summarize_chains(parameters),
        counts=_freeze(n),
        toxicity_only=_freeze(nt),
        prior_mean=_freeze(mu),
        prior_sd=_freeze(prior_scale),
        likelihood_evaluations=evaluations,
        likelihood_work_units=work,
        warmup=warmup,
        efficacy_levels=e_levels,
        toxicity_levels=t_levels,
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
    met = bool(np.all(np.isfinite(ratio)) and np.all(ratio <= target_mcse_ratio))
    return U2OETGAOAdaptivePrecisionResult(
        fit=fit,
        corner_indices=((0, 0), (0, d2.size - 1), (d1.size - 1, 0), (d1.size - 1, d2.size - 1)),
        posterior_sd=_freeze(sd),
        mcse=_freeze(mcse),
        mcse_ratio=_freeze(ratio),
        corner_split_rhat=_freeze(corner_split_rhat),
        target_mcse_ratio=target_mcse_ratio,
        target_met=met,
        termination="target_met" if met else "draw_cap",
        draws_per_chain=retained,
        max_draws_per_chain=max_draws_per_chain,
        batch_draws=batch_draws,
    )
