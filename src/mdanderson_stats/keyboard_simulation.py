"""Batched complete-outcome Keyboard trials with explicit random state."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import _owned
from .keyboard import KeyboardDesign


@dataclass(frozen=True)
class KeyboardSimulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    stop_reason: tuple[str, ...]


def simulate_keyboard(
    design: KeyboardDesign,
    true_toxicity: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> KeyboardSimulation:
    """Simulate fully observed binomial cohorts; zero selection means no MTD.

    Decision tables are precomputed once. Simulations honor custom target keys,
    safety settings and precision stops, including options ignored by R 0.1.3.
    Selection/MCSE bins are [no MTD, dose 1, ..., dose J].
    """
    if not isinstance(design, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
    p = finite(true_toxicity, "true_toxicity")
    if p.ndim != 1 or not 2 <= p.size <= 100 or np.any((p < 0) | (p > 1)):
        raise ValueError("require 2..100 toxicity probabilities in [0,1]")
    sizes = count([cohorts, cohort_size, trials, start_dose], "simulation settings")
    if sizes.shape != (4,) or np.any(sizes < 1):
        raise ValueError("simulation settings must be positive integers")
    nc, size, repetitions, start = map(int, sizes)
    if nc * size > 200 or repetitions > 1000000 or start > p.size:
        raise ValueError("require at most 200 patients, 1000000 trials and a valid starting dose")
    rng = np.random.default_rng(rng)
    table = design.boundary_table(nc * size)
    n = np.zeros((repetitions, p.size), dtype=np.int64)
    y = np.zeros_like(n)
    excluded = np.zeros(n.shape, dtype=bool)
    dose = np.full(repetitions, start - 1)
    active = np.ones(repetitions, dtype=bool)
    reason = np.full(repetitions, "max_patients", dtype="U16")
    for _ in range(nc):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        j = dose[rows]
        n[rows, j] += size
        y[rows, j] += rng.binomial(size, p[j])
        total, events = n[rows, j], y[rows, j]
        unsafe = events >= table.eliminate_min[total - 1]
        excluded[rows] |= unsafe[:, None] & (np.arange(p.size) >= j[:, None])
        lowest_stop = (j == 0) & (events >= table.lowest_stop_min[total - 1])
        excluded[rows[lowest_stop]] = True
        precision = (
            np.zeros(rows.size, dtype=bool)
            if design.early_stop_patients is None
            else total >= design.early_stop_patients
        )
        precision &= ~unsafe
        reason[rows[precision & ~lowest_stop]] = "stop_precision"
        reason[rows[lowest_stop]] = "stop_safety"
        active[rows[precision | lowest_stop]] = False
        move = np.where(
            events <= table.escalate_max[total - 1],
            1,
            np.where(events >= table.deescalate_min[total - 1], -1, 0),
        )
        proposed = np.clip(j + move, 0, p.size - 1)
        dose[rows] = np.where(excluded[rows, proposed], j, proposed)
    selected = np.zeros(repetitions, dtype=np.int64)
    for trial in range(repetitions):
        result = design.select_mtd(n[trial], y[trial], eliminated=excluded[trial])
        selected[trial] = 0 if result.dose is None else result.dose
    frequency = np.bincount(selected, minlength=p.size + 1) / repetitions
    return KeyboardSimulation(
        _owned(n),
        _owned(y),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(n.mean(axis=0)),
        _owned(y.mean(axis=0)),
        tuple(reason),
    )
