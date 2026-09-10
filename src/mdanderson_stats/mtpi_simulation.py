"""Batched complete-outcome mTPI trials using the paper's two safety rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import _owned
from .mtpi import MTPIDesign


@dataclass(frozen=True)
class MTPISimulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    stopped_safety: NDArray[np.bool_]


def simulate_mtpi(
    design: MTPIDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> MTPISimulation:
    """Simulate fully observed binomial cohorts with one-based dose labels.

    Selection/MCSE bins are [no MTD, dose 1, ..., dose J]. Follows the paper's
    safety timing: stop if dose 1 is unsafe; bar escalation into an unsafe dose.
    Final selection uses equal-weight isotonic posterior means. No RNG parity
    with the original Excel/R software is implied.
    """
    if not isinstance(design, MTPIDesign):
        raise TypeError("design must be MTPIDesign")
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
            "require <=200 patients, <=100000 trials, <=2000000 trial-dose cells and a valid start"
        )
    generator = np.random.default_rng(rng)
    nn, yy = np.indices((maximum + 1, maximum + 1))
    valid = (nn > 0) & (yy <= nn)
    posterior = design.posterior(nn[valid], yy[valid])
    move_table = np.zeros(nn.shape, dtype=np.int64)
    unsafe_table = np.zeros(nn.shape, dtype=bool)
    move_table[valid], unsafe_table[valid] = posterior.move, posterior.unsafe
    n = np.zeros((repetitions, p.size), dtype=np.int64)
    y = np.zeros_like(n)
    excluded = np.zeros(n.shape, dtype=bool)
    unsafe = np.zeros(n.shape, dtype=bool)
    dose = np.full(repetitions, start - 1, dtype=np.int64)
    stopped = np.zeros(repetitions, dtype=bool)
    for _ in range(nc):
        rows = np.flatnonzero(~stopped)
        if rows.size == 0:
            break
        j = dose[rows]
        n[rows, j] += size
        y[rows, j] += generator.binomial(size, p[j])
        total, events = n[rows, j], y[rows, j]
        unsafe[rows, j] = unsafe_table[total, events]
        stop = unsafe[rows, 0]
        stopped[rows[stop]] = True
        excluded[rows[stop]] = True
        proposed = np.clip(j + move_table[total, events], 0, p.size - 1)
        barred = (proposed > j) & (unsafe[rows, proposed] | excluded[rows, proposed])
        excluded[rows] |= barred[:, None] & (np.arange(p.size) >= proposed[:, None])
        dose[rows] = np.where(barred, j, proposed)
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
