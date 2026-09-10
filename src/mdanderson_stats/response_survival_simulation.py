"""Adaptive two-arm trial simulation using response and censored survival."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .response_survival import _comparison, _draw_settings, _prior


@dataclass(frozen=True)
class ResponseSurvivalSimulation:
    """Per-trial selected_arm is -1=inconclusive, 0=A, 1=B."""

    selected_arm: NDArray[np.int64]
    enrollment: FloatArray
    stopped_early: NDArray[np.bool_]
    analysis_time: FloatArray
    probability_b_superior: FloatArray
    selection_probability: FloatArray
    selection_standard_error: FloatArray
    mean_enrollment: FloatArray
    enrollment_standard_error: FloatArray
    posterior_draws: int


def simulate_response_survival(
    response_probabilities: ArrayLike,
    survival_means: ArrayLike,
    *,
    response_prior: ArrayLike,
    survival_shape: ArrayLike,
    survival_scale: ArrayLike,
    max_patients: int = 120,
    initial_patients: int = 20,
    accrual_per_period: int = 1,
    additional_followup: float = 40,
    cutoff: float = 0.975,
    replicates: int = 100,
    posterior_draws: int = 10000,
    seed: int | None = None,
) -> ResponseSurvivalSimulation:
    """Simulate the archive's immediate-response, exponential-mixture design.

    Patients arrive in batches of accrual_per_period at times 1,2,... . Initial
    assignments are independent fair coin flips. Before subsequent enrollments,
    accrued responses and censored survival determine P(B superior), the B
    allocation probability and symmetric stopping decision. A stopped trial
    enrolls no additional patient. Otherwise analyze at final arrival+followup.
    All survival and calendar inputs must use the same unit.
    """
    _draw_settings(posterior_draws, seed)
    for name, value, upper in (
        ("max_patients", max_patients, 10000),
        ("initial_patients", initial_patients, max_patients),
        ("accrual_per_period", accrual_per_period, max_patients),
        ("replicates", replicates, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in [1,{upper}]")
    if any(np.iscomplexobj(a) for a in (response_probabilities, survival_means)):
        raise ValueError("response probabilities and survival means must be real")
    probs, means = (
        finite(response_probabilities, "response_probabilities"),
        finite(survival_means, "survival_means"),
    )
    if (
        probs.ndim != 2
        or probs.shape[0] != 2
        or not 1 <= probs.shape[1] <= 100
        or means.shape != probs.shape
    ):
        raise ValueError("response probabilities and survival means require shape (2,K), 1<=K<=100")
    totals = probs.sum(axis=1)
    if np.any(probs < 0) or np.any(np.abs(totals - 1) > 1e-12) or np.any(means <= 0):
        raise ValueError(
            "response probability rows must sum to one; survival means must be positive"
        )
    probs = probs / totals[:, None]
    k = probs.shape[1]
    if replicates * max_patients * posterior_draws * k > 1_000_000_000:
        raise ValueError("simulation exceeds 1 billion patient/draw/category work units")
    alpha = _prior(response_prior, k, "response_prior")
    shape = _prior(survival_shape, k, "survival_shape")
    scale = _prior(survival_scale, k, "survival_scale")
    followup, boundary = (
        scalar(additional_followup, "additional_followup"),
        scalar(cutoff, "cutoff"),
    )
    if followup < 0 or not 0.5 < boundary < 1:
        raise ValueError("require additional_followup>=0 and .5<cutoff<1")
    arrival = np.arange(max_patients) // accrual_per_period + 1.0
    final = float(arrival[-1] + followup)
    if not np.isfinite(final):
        raise ValueError("final follow-up time overflows")
    rng = np.random.default_rng(seed)
    selected = np.full(replicates, -1, dtype=np.int64)
    enrollment = np.zeros((replicates, 2))
    early = np.zeros(replicates, dtype=bool)
    times = np.zeros(replicates)
    probabilities = np.zeros(replicates)
    for trial in range(replicates):
        arm = np.empty(max_patients, dtype=np.int64)
        category = np.empty(max_patients, dtype=np.int64)
        lifetime = np.empty(max_patients)
        enrolled = 0
        probability = 0.5
        while enrolled <= max_patients:
            if enrolled >= initial_patients:
                now = final if enrolled == max_patients else float(arrival[enrolled])
                elapsed = now - arrival[:enrolled]
                exposure = np.minimum(elapsed, lifetime[:enrolled])
                events = lifetime[:enrolled] <= elapsed
                cells = arm[:enrolled] * k + category[:enrolled]
                counts = np.bincount(cells, minlength=2 * k).reshape(2, k)
                deaths = np.bincount(cells, weights=events, minlength=2 * k).reshape(2, k)
                observed = np.bincount(cells, weights=exposure, minlength=2 * k).reshape(2, k)
                posterior_scale = scale + observed
                if not np.all(np.isfinite(posterior_scale)):
                    raise ArithmeticError("posterior exposure overflows")
                probability, _ = _comparison(
                    alpha + counts, shape + deaths, posterior_scale, posterior_draws, rng
                )
                choice = (
                    1 if probability >= boundary else (0 if probability <= 1 - boundary else -1)
                )
                if choice != -1 or enrolled == max_patients:
                    selected[trial] = choice
                    early[trial] = enrolled < max_patients
                    times[trial] = now
                    probabilities[trial] = probability
                    enrollment[trial] = np.bincount(arm[:enrolled], minlength=2)
                    break
            assigned = int(rng.random() < probability)
            response = int(rng.choice(k, p=probs[assigned]))
            duration = float(rng.exponential(means[assigned, response]))
            if not np.isfinite(duration):
                raise ArithmeticError("simulated survival time overflows")
            arm[enrolled], category[enrolled], lifetime[enrolled] = assigned, response, duration
            enrolled += 1
    selection = np.array([np.mean(selected == a) for a in (0, 1)])
    return ResponseSurvivalSimulation(
        np.frombuffer(selected.tobytes(), dtype=np.int64),
        _freeze(enrollment),
        np.frombuffer(early.tobytes(), dtype=np.bool_),
        _freeze(times),
        _freeze(probabilities),
        _freeze(selection),
        _freeze(np.sqrt(selection * (1 - selection) / replicates)),
        _freeze(enrollment.mean(axis=0)),
        _freeze(
            enrollment.std(axis=0, ddof=1) / np.sqrt(replicates)
            if replicates > 1
            else np.full(2, np.nan)
        ),
        posterior_draws,
    )
