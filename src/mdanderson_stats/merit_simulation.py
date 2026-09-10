"""Vectorized Gaussian-latent endpoint sampling for MERIT."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .merit import MERITDesign, _integer


def _correlation(value: float) -> float:
    rho = scalar(value, "correlation")
    if not -1 <= rho <= 1:
        raise ValueError("latent normal correlation must lie in [-1,1]")
    return rho


def _rates(value: ArrayLike, doses: int, name: str) -> FloatArray:
    result = finite(value, name)
    if result.shape != (doses,) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} requires one probability in [0,1] per dose")
    return result


def _draw(
    generator: np.random.Generator, trials: int, doses: int, rho: float
) -> tuple[FloatArray, FloatArray]:
    z = generator.normal(size=(2, trials, doses))
    return z[0], rho * z[0] + np.sqrt((1 - rho) * (1 + rho)) * z[1]


@dataclass(frozen=True)
class MERITSimulation:
    toxicity: NDArray[np.int64]
    efficacy: NDArray[np.int64]
    admissible: NDArray[np.bool_]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    any_selection_probability: float
    power_one: float | None
    power_two: float | None


def simulate_merit(
    design: MERITDesign,
    toxicity_rates: ArrayLike,
    efficacy_rates: ArrayLike,
    *,
    correlation: float = 0.5,
    trials: int = 10000,
    truly_admissible: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> MERITSimulation:
    """Simulate a fixed-size trial, without interim stopping.

    Correlation refers to latent normals, not observed binary endpoints.
    Independent arms; optional truth labels define the two generalized powers.
    """
    trials = _integer(trials, "trials", 1, 1000000)
    if trials * design.doses > 4000000:
        raise ValueError("at most 4 million trial-dose cells")
    rho = _correlation(correlation)
    qt = ndtri(_rates(toxicity_rates, design.doses, "toxicity_rates"))
    qe = ndtri(_rates(efficacy_rates, design.doses, "efficacy_rates"))
    truth = None if truly_admissible is None else np.asarray(truly_admissible)
    if truth is not None and (truth.shape != (design.doses,) or truth.dtype != bool):
        raise ValueError("truly_admissible must be a boolean dose vector")
    generator = np.random.default_rng(rng)
    t = np.zeros((trials, design.doses), dtype=np.int64)
    e = np.zeros_like(t)
    for _ in range(design.patients_per_arm):
        zt, ze = _draw(generator, trials, design.doses, rho)
        t += zt <= qt
        e += ze <= qe
    selected = design.select(t, e).admissible
    probability = selected.mean(0)
    any_selection = selected.any(1)
    p1 = p2 = None
    if truth is not None:
        p1 = float((any_selection & ~selected[:, ~truth].any(1)).mean())
        p2 = float(selected[:, truth].any(1).mean())
    return MERITSimulation(
        _owned(t),
        _owned(e),
        selected,
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        float(any_selection.mean()),
        p1,
        p2,
    )
