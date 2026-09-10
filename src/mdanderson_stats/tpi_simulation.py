"""Batched original TPI trials with precomputed posterior decision probabilities."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .boin import _owned
from .mtpi_simulation import MTPISimulation
from .tpi import TPIDesign, _tpi_move


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
    p = finite(true_toxicity, "true_toxicity")
    settings = count([cohorts, cohort_size, trials, start_dose], "simulation settings")
    if p.ndim != 1 or not 1 <= p.size <= 100 or np.any((p < 0) | (p > 1)):
        raise ValueError("require 1..100 toxicity probabilities in [0,1]")
    if np.any(settings < 1):
        raise ValueError("simulation settings must be positive integers")
    nc, size, repetitions, start = map(int, settings)
    maximum = nc * size
    if maximum > 200 or repetitions > 100000 or start > p.size or repetitions * p.size > 2000000:
        raise ValueError(
            "require <=200 patients, <=100000 trials, <=2 million cells and valid start"
        )
    generator = np.random.default_rng(rng)
    nn, yy = np.indices((maximum + 1, maximum + 1))
    valid = yy <= nn
    posterior = design.posterior(nn[valid], yy[valid])
    probabilities = np.zeros((*nn.shape, 3))
    unsafe = np.zeros(nn.shape, dtype=bool)
    probabilities[valid], unsafe[valid] = posterior.probability, posterior.unsafe
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
        excluded[rows, j] |= unsafe[total, events]
        scores = probabilities[total, events].copy()
        barred = (j < p.size - 1) & excluded[rows, np.minimum(j + 1, p.size - 1)]
        scores[barred, 0] = -1
        move = np.where(excluded[rows, j], -1, _tpi_move(scores))
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
