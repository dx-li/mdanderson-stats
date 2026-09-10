"""Batched complete-cohort 1+2+3 trials with correlated binary endpoints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from ._validation import FloatArray, finite
from .bayesian_monitoring import _integer
from .boin import _owned
from .merit_simulation import _correlation, _draw, _rates
from .rare_disease_123 import RareDisease123Design, _next_batch


@dataclass(frozen=True)
class RareDisease123Simulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray


def simulate_rare_disease_123(
    design: RareDisease123Design,
    toxicity_rates: ArrayLike,
    efficacy_rates: ArrayLike,
    *,
    trials: int = 10000,
    correlation: float = 0.1,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> RareDisease123Simulation:
    """Simulate full outcomes after each cohort, without calendar-time staggering.

    Correlation is between latent standard normals, not binary endpoints.
    Selected dose 0 means no OBD; selection arrays index [none, dose 1, ...].
    Means are per dose. Each trial enrolls at most six patients per dose.
    """
    rates = finite(toxicity_rates, "toxicity_rates")
    if rates.ndim != 1 or not 2 <= rates.size <= 20:
        raise ValueError("require 2..20 dose probabilities")
    doses = rates.size
    trials = _integer(trials, "trials")
    start = _integer(start_dose, "start_dose") - 1
    if not 1 <= trials <= 100000 or trials * doses > 2000000:
        raise ValueError("require 1..100000 trials and at most 2 million trial-dose cells")
    if not 0 <= start < doses:
        raise ValueError("start_dose must identify a dose")
    qt = ndtri(_rates(rates, doses, "toxicity_rates"))
    qe = ndtri(_rates(efficacy_rates, doses, "efficacy_rates"))
    rho = _correlation(correlation)
    generator = np.random.default_rng(rng)
    n = np.zeros((trials, doses), dtype=np.int64)
    t, r = np.zeros_like(n), np.zeros_like(n)
    excluded = np.zeros(n.shape, dtype=bool)
    current = np.full(trials, start, dtype=np.int64)
    selected = np.zeros(trials, dtype=np.int64)
    active = np.arange(trials)
    cohort = np.array([1, 2, 0, 3, 0, 0, 0])
    for _ in range(3 * doses):
        if active.size == 0:
            break
        j = current[active]
        size = cohort[n[active, j]]
        zt, ze = _draw(generator, active.size, 3, rho)
        treated = np.arange(3) < size[:, None]
        n[active, j] += size
        t[active, j] += ((zt <= qt[j, None]) & treated).sum(1)
        r[active, j] += ((ze <= qe[j, None]) & treated).sum(1)
        proposed, masks, _ = _next_batch(
            design, n[active], t[active], r[active], j, excluded[active]
        )
        excluded[active] = masks
        full = (proposed >= 0) & (n[active, np.maximum(proposed, 0)] == 6)
        selected[active[full]] = proposed[full] + 1
        continuing = (proposed >= 0) & ~full
        active = active[continuing]
        current[active] = proposed[continuing]
    if active.size:
        raise ArithmeticError("1+2+3 simulation exceeded its finite cohort bound")
    probability = np.bincount(selected, minlength=doses + 1) / trials
    return RareDisease123Simulation(
        _owned(n),
        _owned(t),
        _owned(r),
        _owned(excluded),
        _owned(selected),
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        _owned(n.mean(0)),
        _owned(t.mean(0)),
        _owned(r.mean(0)),
    )
