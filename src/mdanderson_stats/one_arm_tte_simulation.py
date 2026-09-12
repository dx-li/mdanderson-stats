"""Bounded Monte Carlo operating characteristics for one-arm TTE designs."""

from dataclasses import dataclass

import numpy as np

from ._cdflib import _freeze
from ._validation import scalar
from .one_arm_tte import OneArmTTEDesign, one_arm_tte_trial


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OneArmTTESimulation:
    repetitions: int
    credible_level: float
    early_inferior_probability: float
    early_superior_probability: float
    final_inferior_probability: float
    final_superior_probability: float
    early_inferior_mcse: float
    early_superior_mcse: float
    final_inferior_mcse: float
    final_superior_mcse: float
    sample_size_quantiles: np.ndarray
    duration_quantiles: np.ndarray
    sample_sizes: np.ndarray
    events: np.ndarray
    exposures: np.ndarray
    accrual_stop_times: np.ndarray
    final_times: np.ndarray
    early_inferior: np.ndarray
    early_superior: np.ndarray
    final_inferior: np.ndarray
    final_superior: np.ndarray


def simulate_one_arm_tte(
    design: OneArmTTEDesign,
    true_tte: float,
    accrual_rate: float,
    repetitions: int,
    *,
    seed: int | None = None,
    credible_level: float = 0.95,
) -> OneArmTTESimulation:
    """Simulate independent exponential durations and Poisson arrivals.

    ``true_tte`` and ``accrual_rate`` use the same caller-selected time unit.
    Results use central credible_level quantiles. Arrays are kept
    per trial only, with a total patient budget of 100,000.
    """
    if not isinstance(design, OneArmTTEDesign):
        raise TypeError("design must be a OneArmTTEDesign")
    mean_parameter = scalar(true_tte, "true_tte")
    rate = scalar(accrual_rate, "accrual_rate")
    reps = scalar(repetitions, "repetitions")
    level = scalar(credible_level, "credible_level")
    if (
        isinstance(repetitions, (bool, np.bool_))
        or mean_parameter <= 0
        or rate <= 0
        or reps < 1
        or int(reps) != reps
    ):
        raise ValueError(
            "true_tte and accrual_rate must be positive; repetitions must be an integer"
        )
    if not 0 < level < 1:
        raise ValueError("credible_level must be in (0,1)")
    reps_int = int(reps)
    if reps_int * design.max_patients > 100_000:
        raise ValueError("total simulated patients cannot exceed 100000")
    duration_mean = (
        mean_parameter if design.parameterization == "mean" else mean_parameter / np.log(2)
    )
    generator = np.random.default_rng(seed)
    sample_sizes: np.ndarray = np.empty(reps_int)
    events: np.ndarray = np.empty(reps_int)
    exposures: np.ndarray = np.empty(reps_int)
    accrual_stop_times: np.ndarray = np.empty(reps_int)
    final_times: np.ndarray = np.empty(reps_int)
    early_inferior: np.ndarray = np.zeros(reps_int, dtype=bool)
    early_superior: np.ndarray = np.zeros(reps_int, dtype=bool)
    final_inferior: np.ndarray = np.zeros(reps_int, dtype=bool)
    final_superior: np.ndarray = np.zeros(reps_int, dtype=bool)
    total_checks = 0
    for i in range(reps_int):
        gaps = generator.exponential(1 / rate, design.max_patients)
        enrollment: np.ndarray = np.cumsum(gaps)
        event_duration = generator.exponential(duration_mean, design.max_patients)
        trial = one_arm_tte_trial(design, enrollment, event_duration)
        total_checks += len(trial.monitor_history)
        if total_checks > 100_000:
            raise ValueError("total simulated monitoring checks cannot exceed 100000")
        sample_sizes[i] = trial.final_monitor.patients
        events[i] = trial.final_monitor.events
        exposures[i] = trial.final_monitor.total_time
        accrual_stop_times[i] = trial.accrual_stop_time
        final_times[i] = trial.final_time
        if trial.early_monitor is not None:
            early_inferior[i] = trial.early_monitor.inferior
            early_superior[i] = trial.early_monitor.superior
        final_inferior[i] = trial.final_monitor.inferior
        final_superior[i] = trial.final_monitor.superior
    levels = [(1 - level) / 2, 0.5, (1 + level) / 2]
    quantiles = np.quantile(np.column_stack((sample_sizes, final_times)), levels, axis=0)

    def proportion(values: np.ndarray) -> tuple[float, float]:
        p = float(np.mean(values))
        return p, float(np.sqrt(p * (1 - p) / reps_int))

    ei, ei_error = proportion(early_inferior)
    es, es_error = proportion(early_superior)
    fi, fi_error = proportion(final_inferior)
    fs, fs_error = proportion(final_superior)
    return OneArmTTESimulation(
        reps_int,
        level,
        ei,
        es,
        fi,
        fs,
        ei_error,
        es_error,
        fi_error,
        fs_error,
        _freeze(quantiles[:, 0]),
        _freeze(quantiles[:, 1]),
        _freeze(sample_sizes),
        _freeze(events),
        _freeze(exposures),
        _freeze(accrual_stop_times),
        _freeze(final_times),
        _readonly(early_inferior),
        _readonly(early_superior),
        _readonly(final_inferior),
        _readonly(final_superior),
    )
