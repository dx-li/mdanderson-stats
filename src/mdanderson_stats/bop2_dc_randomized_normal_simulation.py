"""Bounded operating-characteristic simulation for randomized Normal BOP2-DC."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ._validation import FloatArray
from .bop2_dc_randomized_normal import (
    BOP2DCRandomizedNormalDesign,
    _freeze_float,
    _freeze_int,
    _real_scalar,
)
from .bop2_dc_survival_trial import _replay_seed

IntArray = NDArray[np.int64]
_DECISIONS = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_MAX_TRIALS = 100_000
_MAX_PATIENT_CELLS = 1_000_000
_MAX_REPLAY_WORK = 10_000_000
_MAX_QUADRATURE_WORK = 100_000_000
_MAX_RESULT_CELLS = 100_000


@dataclass(frozen=True)
class BOP2DCRandomizedNormalSimulation:
    """Compact Monte Carlo decision, stopping, and enrollment summaries."""

    n_trials: int
    looks: IntArray
    decision_labels: tuple[str, ...]
    decision_count: IntArray
    decision_probability: FloatArray
    decision_mcse: FloatArray
    look_decision_probability: FloatArray
    look_decision_mcse: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: float
    enrollment_mcse: float
    maximum_quadrature_error: float
    rng_seed: int


def _simulation_size(value: float) -> int:
    scalar_value = _real_scalar(value, "n_trials")
    result = int(scalar_value)
    if scalar_value != result or not 1 <= result <= _MAX_TRIALS:
        raise ValueError(f"n_trials must be an integer in [1,{_MAX_TRIALS}]")
    return result


def simulate_bop2_dc_randomized_normal(
    design: BOP2DCRandomizedNormalDesign,
    control_mean: float,
    control_sd: float,
    treatment_mean: float,
    treatment_sd: float,
    *,
    n_trials: int = 100,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCRandomizedNormalSimulation:
    """Simulate fixed-allocation Normal trials and return aggregate outcomes.

    The arm means and SDs define independent complete Normal observations. A
    single seed generates all trials serially. Outcomes are generated relative
    to the control mean, and prior locations are shifted by that same origin,
    so a common additive shift cannot erase arm-level random variation. The
    trial's fixed assignment tape and configured total-N looks are used without
    modification. The default 100 trials is a Python workload choice, not a
    paper recommendation.
    """
    if not isinstance(design, BOP2DCRandomizedNormalDesign):
        raise ValueError("design must be a BOP2DCRandomizedNormalDesign")
    trials = _simulation_size(n_trials)
    mean_c = _real_scalar(control_mean, "control_mean")
    sd_c = _real_scalar(control_sd, "control_sd")
    mean_t = _real_scalar(treatment_mean, "treatment_mean")
    sd_t = _real_scalar(treatment_sd, "treatment_sd")
    if sd_c <= 0 or sd_t <= 0:
        raise ValueError("control_sd and treatment_sd must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        mean_difference = mean_t - mean_c
        control_prior_location = design.control_prior[0] - mean_c
        treatment_prior_location = design.treatment_prior[0] - mean_c
    if not all(
        np.isfinite(value)
        for value in (mean_difference, control_prior_location, treatment_prior_location)
    ):
        raise ArithmeticError("truth-centered randomized Normal inputs are not representable")

    n = design.max_subjects
    patient_cells = trials * n
    replay_work = trials * n * int(design.looks.size)
    # A finite-interval QAGS call can use up to 21*(2*limit-1) integrand
    # evaluations. Each analysis computes both clinical-margin tails.
    quadrature_work = trials * int(design.looks.size) * 2 * 21 * (2 * design.quadrature_limit - 1)
    result_cells = int(design.looks.size) * (len(_DECISIONS) * 2 + 2) + len(_DECISIONS)
    if patient_cells > _MAX_PATIENT_CELLS:
        raise ValueError("simulation exceeds the one-million patient-cell budget")
    if replay_work > _MAX_REPLAY_WORK:
        raise ValueError("simulation exceeds the ten-million repeated-look budget")
    if quadrature_work > _MAX_QUADRATURE_WORK:
        raise ValueError("simulation exceeds the one-hundred-million quadrature-work budget")
    if result_cells > _MAX_RESULT_CELLS:
        raise ValueError("simulation summary exceeds its retained-cell budget")

    centered_control_prior = (
        control_prior_location,
        design.control_prior[1],
        design.control_prior[2],
        design.control_prior[3],
    )
    centered_treatment_prior = (
        treatment_prior_location,
        design.treatment_prior[1],
        design.treatment_prior[2],
        design.treatment_prior[3],
    )
    centered_design = replace(
        design,
        control_prior=centered_control_prior,
        treatment_prior=centered_treatment_prior,
    )
    seed = _replay_seed(rng)
    generator = np.random.default_rng(seed)
    look_counts = np.zeros((design.looks.size, len(_DECISIONS)), dtype=np.int64)
    mean_n = 0.0
    m2_n = 0.0
    maximum_error = 0.0
    arm_sd = np.where(design.arm_assignments == 0, sd_c, sd_t)
    arm_mean = np.where(design.arm_assignments == 0, 0.0, mean_difference)

    for trial_index in range(1, trials + 1):
        standardized = generator.standard_normal(n)
        with np.errstate(over="ignore", invalid="ignore"):
            outcomes = arm_mean + arm_sd * standardized
        if np.any(~np.isfinite(outcomes)):
            raise ArithmeticError("simulated centered Normal outcomes are not representable")
        replay = centered_design.replay(outcomes)
        terminal = replay.states[-1]
        for state in replay.states:
            maximum_error = max(maximum_error, state.absolute_error_lrv, state.absolute_error_cmv)
        try:
            decision_index = _DECISIONS.index(terminal.decision)
            look_index = int(np.searchsorted(design.looks, terminal.total_n))
        except ValueError as exc:
            raise ArithmeticError(
                "randomized Normal replay returned an unknown terminal state"
            ) from exc
        if look_index >= design.looks.size or int(design.looks[look_index]) != terminal.total_n:
            raise ArithmeticError("randomized Normal replay stopped outside a scheduled look")
        look_counts[look_index, decision_index] += 1
        delta = float(terminal.total_n) - mean_n
        mean_n += delta / trial_index
        m2_n += delta * (float(terminal.total_n) - mean_n)

    counts = np.sum(look_counts, axis=0, dtype=np.int64)
    probability = counts.astype(np.float64) / trials
    look_probability = look_counts.astype(np.float64) / trials
    decision_mcse = np.sqrt(probability * (1 - probability) / trials)
    look_mcse = np.sqrt(look_probability * (1 - look_probability) / trials)
    size_probability = np.sum(look_probability, axis=1)
    enrollment_mcse = np.sqrt(m2_n / (trials - 1) / trials) if trials > 1 else float("nan")
    if int(np.sum(counts)) != trials or not np.isclose(np.sum(size_probability), 1.0):
        raise ArithmeticError("simulated randomized Normal decision masses do not sum to one")
    if np.any(~np.isfinite(probability)) or np.any(~np.isfinite(look_probability)):
        raise ArithmeticError("simulated randomized Normal probabilities are not finite")
    return BOP2DCRandomizedNormalSimulation(
        trials,
        _freeze_int(design.looks),
        _DECISIONS,
        _freeze_int(counts),
        _freeze_float(probability),
        _freeze_float(decision_mcse),
        _freeze_float(look_probability),
        _freeze_float(look_mcse),
        _freeze_float(size_probability),
        mean_n,
        enrollment_mcse,
        maximum_error,
        seed,
    )
