"""Batched original TPI trials with precomputed posterior decision probabilities."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite
from .boin import _owned
from .mtpi_simulation import MTPISimulation
from .tpi import TPIDesign, _tpi_move

_MAX_DOSE_PRIOR_LOOKUP_CELLS = 5_000_000


def simulate_tpi(
    design: TPIDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> MTPISimulation:
    """Fully observed binomial cohorts; selection bins are no MTD, dose 1, ..., J.

    Uses the same count-summary result type as simulate_mtpi, but original TPI
    probabilities, priors and safety rules. Final isotonic weights are equal.
    """
    if not isinstance(design, TPIDesign):
        raise TypeError("design must be TPIDesign")
    if np.iscomplexobj(true_toxicity):
        raise ValueError("true_toxicity must be real")
    p = finite(true_toxicity, "true_toxicity")
    settings = count([cohorts, cohort_size, trials, start_dose], "simulation settings")
    if p.ndim != 1 or not 1 <= p.size <= 100 or np.any((p < 0) | (p > 1)):
        raise ValueError("require 1..100 toxicity probabilities in [0,1]")
    if design.prior_dose_count is not None and p.size != design.prior_dose_count:
        raise ValueError("true_toxicity length must match the dose-specific prior count")
    if np.any(settings < 1):
        raise ValueError("simulation settings must be positive integers")
    nc, size, repetitions, start = map(int, settings)
    maximum = nc * size
    if maximum > 200 or repetitions > 100000 or start > p.size or repetitions * p.size > 2000000:
        raise ValueError(
            "require <=200 patients, <=100000 trials, <=2 million cells and valid start"
        )
    dose_prior_lookup: (
        tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.bool_], NDArray[np.int64]] | None
    ) = None
    if design.has_dose_specific_prior:
        prior_index: dict[tuple[float, float], int] = {}
        dose_lookup_index = np.empty(p.size, dtype=np.int64)
        unique_priors: list[tuple[float, float]] = []
        prior_shapes = design.dose_prior_shapes
        assert prior_shapes is not None
        for dose_index, pair in enumerate(prior_shapes):
            if pair not in prior_index:
                prior_index[pair] = len(unique_priors)
                unique_priors.append(pair)
            dose_lookup_index[dose_index] = prior_index[pair]
        lookup_cells = len(unique_priors) * (maximum + 1) ** 2
        estimated_lookup_bytes = lookup_cells * 3 + (maximum + 1) ** 2 * 64
        if lookup_cells > _MAX_DOSE_PRIOR_LOOKUP_CELLS or estimated_lookup_bytes > 64 * 1024 * 1024:
            raise ValueError("dose-specific TPI decision lookup exceeds its bounded storage")
        nn, yy = np.indices((maximum + 1, maximum + 1))
        valid = yy <= nn
        moves = np.zeros((len(unique_priors), maximum + 1, maximum + 1), dtype=np.int8)
        barred_moves = np.zeros_like(moves)
        lookup_unsafe = np.zeros(moves.shape, dtype=bool)
        for prior_index_value in range(len(unique_priors)):
            representative_dose = int(np.flatnonzero(dose_lookup_index == prior_index_value)[0]) + 1
            posterior = design.posterior(nn[valid], yy[valid], dose=representative_dose)
            moves[prior_index_value, nn[valid], yy[valid]] = posterior.move
            barred_probabilities = posterior.probability.copy()
            barred_probabilities[:, 0] = -1
            barred_moves[prior_index_value, nn[valid], yy[valid]] = _tpi_move(barred_probabilities)
            lookup_unsafe[prior_index_value, nn[valid], yy[valid]] = posterior.unsafe
        dose_prior_lookup = moves, barred_moves, lookup_unsafe, dose_lookup_index
    else:
        nn, yy = np.indices((maximum + 1, maximum + 1))
        valid = yy <= nn
        posterior = design.posterior(nn[valid], yy[valid])
        probabilities = np.zeros((*nn.shape, 3))
        unsafe = np.zeros(nn.shape, dtype=bool)
        probabilities[valid], unsafe[valid] = posterior.probability, posterior.unsafe
    generator = np.random.default_rng(rng)
    n = np.zeros((repetitions, p.size), dtype=np.int64)
    y = np.zeros_like(n)
    excluded = np.zeros(n.shape, dtype=bool)
    dose = np.full(repetitions, start - 1, dtype=np.int64)
    stopped = np.zeros(repetitions, dtype=bool)
    for _ in range(nc):
        rows = np.flatnonzero(~stopped)
        if not rows.size:
            break
        j = dose[rows]
        n[rows, j] += size
        y[rows, j] += generator.binomial(size, p[j])
        total, events = n[rows, j], y[rows, j]
        if dose_prior_lookup is None:
            excluded[rows, j] |= unsafe[total, events]
            scores = probabilities[total, events].copy()
        else:
            moves, barred_moves, lookup_unsafe, lookup_index = dose_prior_lookup
            selected_prior = lookup_index[j]
            excluded[rows, j] |= lookup_unsafe[selected_prior, total, events]
        barred = (j < p.size - 1) & excluded[rows, np.minimum(j + 1, p.size - 1)]
        if dose_prior_lookup is None:
            scores[barred, 0] = -1
            proposed_move = _tpi_move(scores)
        else:
            proposed_move = np.where(
                barred,
                barred_moves[selected_prior, total, events],
                moves[selected_prior, total, events],
            )
        move = np.where(excluded[rows, j], -1, proposed_move)
        proposed = np.clip(j + move, 0, p.size - 1)
        stop = excluded[rows, 0] | excluded[rows, proposed]
        stopped[rows[stop]] = True
        excluded[rows[excluded[rows, 0]]] = True
        dose[rows[~stop]] = proposed[~stop]
    selected = np.zeros(repetitions, dtype=np.int64)
    for trial in np.flatnonzero(~stopped):
        result = design.select_mtd(n[trial], y[trial], eliminated=excluded[trial])
        selected[trial] = 0 if result.dose is None else result.dose
    frequency = np.bincount(selected, minlength=p.size + 1) / repetitions
    return MTPISimulation(
        _owned(n),
        _owned(y),
        _owned(excluded),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(n.mean(axis=0)),
        _owned(y.mean(axis=0)),
        _owned(stopped),
    )
