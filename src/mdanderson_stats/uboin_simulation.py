"""Bounded simulation helpers for complete-outcome U-BOIN designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .uboin_conduct import UBOINDesign


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _integer(value: int, name: str, low: int, high: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not low <= candidate <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(candidate)


def uboin_gumbel_probabilities(
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    *,
    association: float = 0.2,
) -> FloatArray:
    """Return joint Bernoulli cells from the U-BOIN Gumbel association model.

    Rows are efficacy ``0,1`` and columns are toxicity ``0,1``.  The
    factored form remains finite for association values such as ``+/-1000``.
    """
    tox_shape = np.shape(toxicity)
    eff_shape = np.shape(efficacy)
    if len(tox_shape) != 1 or len(eff_shape) != 1 or tox_shape != eff_shape:
        raise ValueError("toxicity and efficacy must be same-shaped one-dimensional vectors")
    if not 1 <= tox_shape[0] <= 100:
        raise ValueError("toxicity and efficacy must contain 1..100 doses")
    tox = finite(toxicity, "toxicity")
    eff = finite(efficacy, "efficacy")
    if np.any((tox < 0) | (tox > 1)) or np.any((eff < 0) | (eff > 1)):
        raise ValueError("toxicity and efficacy probabilities must lie in [0,1]")
    association = scalar(association, "association")
    rho = np.tanh(association / 2)
    p00 = (1 - eff) * (1 - tox) * (1 + eff * tox * rho)
    p01 = (1 - eff) * tox * (1 - eff * (1 - tox) * rho)
    p10 = eff * (1 - tox) * (1 - (1 - eff) * tox * rho)
    p11 = eff * tox * (1 + (1 - eff) * (1 - tox) * rho)
    result = np.stack((p00, p01, p10, p11), axis=1).reshape(-1, 2, 2)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ArithmeticError("Gumbel probabilities are not finite nonnegative values")
    return _readonly(result)


@dataclass(frozen=True)
class UBOINSimulation:
    selections: NDArray[np.int64]
    joint_counts: NDArray[np.int64]
    stop_reason: NDArray[np.str_]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray
    early_stop_probability: float
    early_stop_mcse: float


def simulate_uboin(
    design: UBOINDesign,
    joint_probabilities: ArrayLike,
    *,
    cohort_size: int = 3,
    trials: int = 1000,
    seed: int | None = None,
) -> UBOINSimulation:
    """Simulate complete categorical outcomes under a U-BOIN conduct design.

    A cohort is sampled at the current dose, then the controller is called.
    Thus a cohort can overshoot ``s1`` or ``s2`` by at most ``cohort_size-1``.
    No patient-level history or decision trace is retained.
    """
    if not isinstance(design, UBOINDesign):
        raise TypeError("design must be a UBOINDesign")
    try:
        shape = np.shape(joint_probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_probabilities must have shape (D,E,T)") from exc
    if len(shape) != 3:
        raise ValueError("joint_probabilities must have shape (D,E,T)")
    d, e, t = shape
    if not 1 <= d <= 100 or (e, t) != design.prior.shape[-2:]:
        raise ValueError("joint_probabilities dimensions must match design")
    probabilities = finite(joint_probabilities, "joint_probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("joint_probabilities entries must lie in [0,1]")
    row_sums = probabilities.sum(axis=(1, 2))
    if not np.all(np.isclose(row_sums, 1.0, rtol=0, atol=1e-12)):
        raise ValueError("each dose joint probability table must sum to one")
    probabilities = probabilities / row_sums[:, None, None]
    cohort = _integer(cohort_size, "cohort_size", 1, 100)
    repetitions = _integer(trials, "trials", 1, 100_000)
    if repetitions * design.max_patients > 100_000:
        raise ValueError("trials * max_patients must be at most 100,000")
    if repetitions * d * e * t > 100_000:
        raise ValueError("trials * D * E * T must be at most 100,000")
    if repetitions * design.max_patients * d > 1_000_000:
        raise ValueError("trials * max_patients * D must be at most 1,000,000")
    # Validate all controller dimensions and scalar design settings before the
    # simulation loop, including a valid zero-data starting decision.
    empty = np.zeros((d, e, t), dtype=np.int64)
    design.decision(empty, current_dose=design.starting_dose, stage=1)

    rng = np.random.default_rng(seed)
    all_counts = np.zeros((repetitions, d, e, t), dtype=np.int64)
    selections: NDArray[np.int64] = np.zeros(repetitions, dtype=np.int64)
    reasons: NDArray[np.str_] = np.empty(repetitions, dtype="U32")
    patients = np.zeros((repetitions, d), dtype=np.float64)
    toxicities = np.zeros((repetitions, d), dtype=np.float64)
    responses = np.zeros((repetitions, d), dtype=np.float64)

    for trial in range(repetitions):
        counts = np.zeros((d, e, t), dtype=np.int64)
        current = design.starting_dose
        stage = 1
        eliminated = None
        while True:
            total = int(counts.sum())
            size = min(cohort, design.max_patients - total)
            if size <= 0:
                result = design.decision(
                    counts, current_dose=current, stage=stage, eliminated=eliminated
                )
                reasons[trial] = result.action
                selections[trial] = result.selected_dose or 0
                break
            draw = rng.multinomial(size, probabilities[current - 1].reshape(-1))
            counts[current - 1] += draw.reshape(e, t)
            result = design.decision(
                counts, current_dose=current, stage=stage, eliminated=eliminated
            )
            if not np.any(result.allocation_probabilities):
                reasons[trial] = result.action
                selections[trial] = result.selected_dose or 0
                break
            allocation = result.allocation_probabilities
            if np.count_nonzero(allocation) == 1:
                current = int(np.flatnonzero(allocation == 1)[0] + 1)
            else:
                current = int(rng.choice(d, p=allocation)) + 1
            stage = result.stage
            eliminated = result.eliminated
        all_counts[trial] = counts
        patients[trial] = counts.sum(axis=(1, 2))
        toxicities[trial] = counts[:, :, design.dlt_level :].sum(axis=(1, 2))
        responses[trial] = counts[:, design.response_level :, :].sum(axis=(1, 2))

    selection_probability = np.bincount(selections, minlength=d + 1).astype(float) / repetitions
    selection_mcse = np.sqrt(selection_probability * (1 - selection_probability) / repetitions)
    early = patients.sum(axis=1) < design.max_patients
    early_probability = float(np.mean(early))
    early_mcse = float(np.sqrt(early_probability * (1 - early_probability) / repetitions))
    return UBOINSimulation(
        _readonly(selections, dtype=np.int64),
        _readonly(all_counts, dtype=np.int64),
        _readonly(reasons, dtype=np.str_),
        _readonly(selection_probability),
        _readonly(selection_mcse),
        _readonly(patients.mean(axis=0)),
        _readonly(toxicities.mean(axis=0)),
        _readonly(responses.mean(axis=0)),
        early_probability,
        early_mcse,
    )
