"""Explicit stochastic generation for the BARD BF-BLRM stage-one replay.

This module supplies a documented Python calendar/outcome convention around
the existing BF-BLRM trial engine. It does not claim native RNG or timing
parity, and it requires the escalation cap because the paper's calibrated
cap is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import finite, scalar
from .bard_blrm import BARDLogisticPrior
from .bard_blrm_trial import BARDBLRMTrial, run_bard_blrm_trial
from .bard_response import BARDResponseModel
from .bf_boin_simulation import (
    _bard_conditional_response_probability,
    _validate_bard_response_truth,
    _weibull_endpoint,
)

_MAX_ARRIVALS = 2_000
_MAX_DOSES = 100
_MAX_OUTCOME_CELLS = 200_000
_MAX_SEED = 2**64 - 1


def _owned(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif (
        isinstance(value, (float, np.floating)) and np.isfinite(value) and value == np.floor(value)
    ):
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    if not lower <= parsed <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return parsed


def _bounded_vector(value: ArrayLike, name: str, maximum: int) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= maximum or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real vector with 1..{maximum} values")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= maximum or any(
            isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_)) for item in value
        ):
            raise ValueError(f"{name} must be a real vector with 1..{maximum} values")
    else:
        raise ValueError(f"{name} must be a bounded real vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = finite(value, name)
    return result


@dataclass(frozen=True, slots=True)
class BARDBLRMSimulationDesign:
    """Captured settings for one generated BF-BLRM stage-one replay."""

    doses: ArrayLike
    reference_dose: float
    prior: BARDLogisticPrior
    target_interval: ArrayLike
    eta: float
    cohort_size: int
    max_escalation_patients: int
    backfill_evaluable_cap: int
    draws: int
    warmup: int
    chains: int
    max_arrivals: int
    arrival_distribution: str = "uniform"
    accrual_rate: float = 1.0
    dlt_window: float = 1.0
    boundary_policy: str = "stop"
    max_total_evaluations: int = 2_000_000
    max_total_work: int = 50_000_000
    accelerated_titration: bool = False
    titration_cap: int | None = None
    true_grade2: ArrayLike | None = None
    grade2_assessment_delay: float | None = None

    def __post_init__(self) -> None:
        doses = _bounded_vector(self.doses, "doses", _MAX_DOSES)
        if np.any(doses <= 0) or np.any(np.diff(doses) <= 0):
            raise ValueError("doses must be strictly increasing and positive")
        if not isinstance(self.prior, BARDLogisticPrior):
            raise TypeError("prior must be a BARDLogisticPrior")
        if np.shape(self.target_interval) != (2,) or np.iscomplexobj(self.target_interval):
            raise ValueError("target_interval must be a real vector of length 2")
        target = finite(self.target_interval, "target_interval")
        if not 0 <= target[0] < target[1] <= 1:
            raise ValueError("target_interval must satisfy 0 <= lower < upper <= 1")
        reference = scalar(self.reference_dose, "reference_dose")
        eta = scalar(self.eta, "eta")
        rate = scalar(self.accrual_rate, "accrual_rate")
        window = scalar(self.dlt_window, "dlt_window")
        if reference <= 0 or not 0 <= eta <= 1 or rate <= 0 or window <= 0:
            raise ValueError(
                "reference_dose, accrual_rate and dlt_window must be positive; eta in [0,1]"
            )
        integers = {
            "cohort_size": _integer(self.cohort_size, "cohort_size", 1, 100_000),
            "max_escalation_patients": _integer(
                self.max_escalation_patients, "max_escalation_patients", 1, 1_000
            ),
            "backfill_evaluable_cap": _integer(
                self.backfill_evaluable_cap, "backfill_evaluable_cap", 1, 1_000_000
            ),
            "draws": _integer(self.draws, "draws", 8, 100_000),
            "warmup": _integer(self.warmup, "warmup", 0, 100_000),
            "chains": _integer(self.chains, "chains", 2, 16),
            "max_arrivals": _integer(self.max_arrivals, "max_arrivals", 1, _MAX_ARRIVALS),
            "max_total_evaluations": _integer(
                self.max_total_evaluations, "max_total_evaluations", 1, 2_000_000
            ),
            "max_total_work": _integer(self.max_total_work, "max_total_work", 1, 50_000_000),
        }
        if integers["chains"] * integers["draws"] * (2 + 3 * doses.size) > 2_000_000:
            raise ValueError("each retained BF-BLRM fit exceeds two million cells")
        if integers["chains"] * integers["draws"] * doses.size > integers["max_total_work"]:
            raise ValueError("work budget cannot cover the minimum prior-only BF-BLRM fit")
        if integers["max_arrivals"] * doses.size > _MAX_OUTCOME_CELLS:
            raise ValueError("generated potential-outcome tapes exceed 200000 cells")
        retained = ((5 if self.accelerated_titration else 4) * integers["max_arrivals"] + 2) * (
            14 * doses.size + 2
        )
        if retained > 2_000_000:
            raise ValueError("replay snapshots exceed two million retained cells")
        if self.arrival_distribution not in ("uniform", "exponential"):
            raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
        if self.boundary_policy not in ("raise", "stop"):
            raise ValueError("boundary_policy must be 'raise' or 'stop'")
        if not isinstance(self.accelerated_titration, (bool, np.bool_)):
            raise ValueError("accelerated_titration must be boolean")
        if self.accelerated_titration:
            cap = (
                doses.size
                if self.titration_cap is None
                else _integer(self.titration_cap, "titration_cap", 1, doses.size)
            )
            if self.true_grade2 is None or self.grade2_assessment_delay is None:
                raise ValueError("titration requires true_grade2 and grade2_assessment_delay")
            grade2 = _bounded_vector(self.true_grade2, "true_grade2", _MAX_DOSES)
            if grade2.shape != doses.shape or np.any((grade2 < 0) | (grade2 > 1)):
                raise ValueError("true_grade2 must have one probability in [0,1] per dose")
            delay = scalar(self.grade2_assessment_delay, "grade2_assessment_delay")
            if delay < 0:
                raise ValueError("grade2_assessment_delay must be nonnegative")
            object.__setattr__(self, "true_grade2", _freeze(grade2))
            object.__setattr__(self, "grade2_assessment_delay", delay)
            object.__setattr__(self, "titration_cap", cap)
        elif (
            self.titration_cap is not None
            or self.true_grade2 is not None
            or self.grade2_assessment_delay is not None
        ):
            raise ValueError("titration settings require accelerated_titration=True")
        object.__setattr__(self, "doses", _freeze(doses))
        object.__setattr__(self, "target_interval", _freeze(target))
        object.__setattr__(self, "reference_dose", reference)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "accrual_rate", rate)
        object.__setattr__(self, "dlt_window", window)
        for name, value in integers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "accelerated_titration", bool(self.accelerated_titration))


@dataclass(frozen=True, slots=True)
class _BARDGeneratedOutcomeTapes:
    arrival_times: NDArray[np.float64]
    profile_indices: NDArray[np.int64]
    factor_profiles: NDArray[np.int64]
    profile_probabilities: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]
    sampled_response_probabilities: NDArray[np.float64]
    potential_toxicities: NDArray[np.bool_]
    potential_responses: NDArray[np.bool_]
    dlt_assessment_delays: NDArray[np.float64]
    response_assessment_delays: NDArray[np.float64]
    potential_grade2_toxicities: NDArray[np.bool_] | None
    grade2_assessment_delays: NDArray[np.float64] | None


def _generate_bard_blrm_outcome_tapes(
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    *,
    arrivals: int,
    accrual_rate: float,
    arrival_distribution: str,
    dlt_window: float,
    joint_toxicity_response_probability: ArrayLike | None = None,
    profile_probabilities: ArrayLike | None = None,
    true_grade2: ArrayLike | None = None,
    grade2_assessment_delay: float | None = None,
    rng: np.random.Generator,
    first_arrival_time: float = 0.0,
) -> _BARDGeneratedOutcomeTapes:
    """Generate bounded tapes; private so full-trial adapters can reuse policy."""
    n = _integer(arrivals, "arrivals", 1, _MAX_ARRIVALS)
    toxicity = _bounded_vector(true_toxicity, "true_toxicity", _MAX_DOSES)
    if np.any((toxicity < 0) | (toxicity > 1)):
        raise ValueError("true_toxicity must contain probabilities in [0,1]")
    if toxicity.size * n > _MAX_OUTCOME_CELLS:
        raise ValueError("generated potential-outcome tapes exceed 200000 cells")
    window = scalar(dlt_window, "dlt_window")
    rate = scalar(accrual_rate, "accrual_rate")
    first = scalar(first_arrival_time, "first_arrival_time")
    if window <= 0 or rate <= 0 or first < 0:
        raise ValueError(
            "dlt_window/accrual_rate must be positive and first_arrival_time nonnegative"
        )
    if arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be a BARDResponseModel")
    if toxicity.shape != response_model.population_response.shape:
        raise ValueError("true_toxicity must have one probability per response-model dose")
    profiles, model_weights, conditional, joint = _validate_bard_response_truth(
        response_model,
        response_model.population_response,
        toxicity,
        joint_toxicity_response_probability,
    )
    if profiles is None or model_weights is None or conditional is None:
        raise RuntimeError("validated response model unexpectedly omitted profiles")
    if profile_probabilities is None:
        weights = np.array(model_weights, copy=True)
    else:
        weights = _bounded_vector(
            profile_probabilities, "profile_probabilities", model_weights.size
        )
        if weights.shape != model_weights.shape:
            raise ValueError("profile_probabilities must match the response model profile rows")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("profile_probabilities must be nonnegative and sum to one")
        weights = weights / weights.sum()
    grade_prob: NDArray[np.float64] | None = None
    grade_delay: float | None = None
    if true_grade2 is not None:
        grade_prob = _bounded_vector(true_grade2, "true_grade2", _MAX_DOSES)
        if grade_prob.shape != toxicity.shape or np.any((grade_prob < 0) | (grade_prob > 1)):
            raise ValueError("true_grade2 must have one probability in [0,1] per dose")
        if grade2_assessment_delay is None:
            raise ValueError("grade2_assessment_delay is required with true_grade2")
        grade_delay = scalar(grade2_assessment_delay, "grade2_assessment_delay")
        if grade_delay < 0:
            raise ValueError("grade2_assessment_delay must be nonnegative")
    elif grade2_assessment_delay is not None:
        raise ValueError("grade2_assessment_delay requires true_grade2")

    # Preflight every Weibull calibration before touching the caller's RNG.
    endpoints = tuple(_weibull_endpoint(float(prob), window) for prob in toxicity)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean_gap = (2.0 if arrival_distribution == "uniform" else 1.0) / rate
    if not np.isfinite(mean_gap) or mean_gap <= 0:
        raise ValueError("accrual_rate makes the arrival interval unrepresentable")
    arrivals_out: NDArray[np.float64] = np.empty(n, dtype=np.float64)
    arrivals_out[0] = first
    if n > 1:
        if arrival_distribution == "uniform":
            increments = rng.uniform(0.0, 2.0 / rate, size=n - 1)
        else:
            increments = rng.exponential(1.0 / rate, size=n - 1)
        if np.any(increments <= 0) or not np.all(np.isfinite(increments)):
            raise ArithmeticError("generated interarrival increments are not representable")
        for i, gap in enumerate(increments, start=1):
            arrivals_out[i] = arrivals_out[i - 1] + gap
            if not np.isfinite(arrivals_out[i]) or arrivals_out[i] <= arrivals_out[i - 1]:
                raise ArithmeticError(
                    "positive arrival increment is below calendar-time resolution"
                )

    # One joint profile per arrival; each dose column is an independently
    # generated potential outcome under the documented counterfactual policy.
    profile_index = rng.choice(profiles.shape[0], size=n, p=weights).astype(np.int64)
    response_by_profile = conditional[:, profile_index].T
    toxicity_matrix = np.empty((n, toxicity.size), dtype=bool)
    response_matrix = np.empty_like(toxicity_matrix)
    sampled_response = np.empty((n, toxicity.size), dtype=np.float64)
    dlt_delays = np.full((n, toxicity.size), window, dtype=np.float64)
    response_delays = np.full((n, toxicity.size), window, dtype=np.float64)
    for j, (prob, (shape, _scale)) in enumerate(zip(toxicity, endpoints, strict=True)):
        if prob == 0:
            dlt: NDArray[np.bool_] = np.zeros(n, dtype=bool)
        elif prob == 1:
            dlt = np.ones(n, dtype=bool)
            dlt_delays[:, j] = window / 2.0
        else:
            u = rng.random(n)
            dlt = u < prob
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                conditional_fraction = np.power(-np.log1p(-u[dlt]) / -np.log1p(-prob), 1.0 / shape)
                dlt_time = window * conditional_fraction
            if (
                not np.all(np.isfinite(dlt_time))
                or np.any(dlt_time < 0)
                or np.any(dlt_time > window)
            ):
                raise ArithmeticError(
                    "generated DLT assessment time is not finite in the DLT window"
                )
            dlt_delays[dlt, j] = dlt_time
        toxicity_matrix[:, j] = dlt
        for i in range(n):
            q = None if joint is None else float(joint[j, profile_index[i]])
            p_response = _bard_conditional_response_probability(
                float(response_by_profile[i, j]), float(prob), q, bool(dlt[i])
            )
            sampled_response[i, j] = p_response
            response_matrix[i, j] = bool(rng.random() < p_response)
    grade_matrix: NDArray[np.bool_] | None = None
    grade_delays: NDArray[np.float64] | None = None
    if grade_prob is not None:
        assert grade_delay is not None
        grade_matrix = np.zeros_like(toxicity_matrix)
        for j, probability in enumerate(grade_prob):
            grade_matrix[:, j] = (~toxicity_matrix[:, j]) & (rng.random(n) < probability)
        grade_delays = np.full((n, toxicity.size), float(grade_delay), dtype=np.float64)
    return _BARDGeneratedOutcomeTapes(
        _owned(arrivals_out),
        _owned(profile_index, np.int64),
        _owned(profiles[profile_index], np.int64),
        _owned(weights),
        _owned(response_by_profile),
        _owned(sampled_response),
        _owned(toxicity_matrix, np.bool_),
        _owned(response_matrix, np.bool_),
        _owned(dlt_delays),
        _owned(response_delays),
        None if grade_matrix is None else _owned(grade_matrix, np.bool_),
        None if grade_delays is None else _owned(grade_delays),
    )


@dataclass(frozen=True, slots=True)
class BARDGeneratedBLRMStageOne:
    design: BARDBLRMSimulationDesign
    outcome_tapes: _BARDGeneratedOutcomeTapes
    trial: BARDBLRMTrial
    seed: int | None


def simulate_bard_blrm_stage_one(
    design: BARDBLRMSimulationDesign,
    true_toxicity: ArrayLike,
    response_model: BARDResponseModel,
    *,
    joint_toxicity_response_probability: ArrayLike | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
) -> BARDGeneratedBLRMStageOne:
    """Generate an explicit stochastic patient tape and run BF-BLRM stage one.

    Integer/omitted seeds are recorded and replayable. A supplied Generator is
    advanced directly; its prior state is intentionally not serialized.
    """
    if not isinstance(design, BARDBLRMSimulationDesign):
        raise TypeError("design must be a BARDBLRMSimulationDesign")
    doses = np.asarray(design.doses)
    toxicity = _bounded_vector(true_toxicity, "true_toxicity", _MAX_DOSES)
    if toxicity.shape != doses.shape or np.any((toxicity < 0) | (toxicity > 1)):
        raise ValueError("true_toxicity must match doses and contain probabilities in [0,1]")
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be a BARDResponseModel")
    if toxicity.shape != response_model.population_response.shape:
        raise ValueError("true_toxicity must match response-model doses")
    # Validate the complete model and joint endpoint law before RNG creation/use.
    _validate_bard_response_truth(
        response_model,
        response_model.population_response,
        toxicity,
        joint_toxicity_response_probability,
    )
    for probability in toxicity:
        _weibull_endpoint(float(probability), design.dlt_window)
    if rng is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        generator = np.random.default_rng(seed)
    elif isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
        seed = int(rng)
        if not 0 <= seed <= _MAX_SEED:
            raise ValueError("rng seed must be an integer in 0..2**64-1")
        generator = np.random.default_rng(seed)
    elif isinstance(rng, np.random.Generator):
        seed = None
        generator = rng
    else:
        raise ValueError("rng must be None, a nonnegative 64-bit seed, or numpy Generator")
    tapes = _generate_bard_blrm_outcome_tapes(
        toxicity,
        response_model,
        arrivals=design.max_arrivals,
        accrual_rate=design.accrual_rate,
        arrival_distribution=design.arrival_distribution,
        dlt_window=design.dlt_window,
        joint_toxicity_response_probability=joint_toxicity_response_probability,
        true_grade2=design.true_grade2,
        grade2_assessment_delay=design.grade2_assessment_delay,
        rng=generator,
    )
    trial = run_bard_blrm_trial(
        doses,
        design.reference_dose,
        design.prior,
        target_interval=design.target_interval,
        eta=design.eta,
        arrival_times=tapes.arrival_times,
        potential_toxicities=tapes.potential_toxicities,
        potential_responses=tapes.potential_responses,
        dlt_assessment_delays=tapes.dlt_assessment_delays,
        response_assessment_delays=tapes.response_assessment_delays,
        dlt_window=design.dlt_window,
        cohort_size=design.cohort_size,
        max_escalation_patients=design.max_escalation_patients,
        backfill_evaluable_cap=design.backfill_evaluable_cap,
        draws=design.draws,
        warmup=design.warmup,
        chains=design.chains,
        rng=generator,
        boundary_policy=design.boundary_policy,
        max_total_evaluations=design.max_total_evaluations,
        max_total_work=design.max_total_work,
        accelerated_titration=design.accelerated_titration,
        titration_cap=design.titration_cap,
        potential_grade2_toxicities=tapes.potential_grade2_toxicities,
        grade2_assessment_delays=tapes.grade2_assessment_delays,
    )
    return BARDGeneratedBLRMStageOne(design, tapes, trial, seed)
