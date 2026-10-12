"""Bounded all-scenario mTPI simulations with auditable seed and precision."""

from dataclasses import dataclass

import numpy as np

from .boin import _owned
from .mtpi_desktop import MTPIDesktopStudy, _integer
from .mtpi_desktop_trial import _capacity, _replay


@dataclass(frozen=True)
class MTPIDesktopSimulation:
    study: MTPIDesktopStudy
    profile: str
    seed: int
    selection_probability: np.ndarray
    selection_mcse: np.ndarray
    mean_patients: np.ndarray
    mean_patients_mcse: np.ndarray
    mean_total_patients: np.ndarray
    pooled_toxicity_rate: np.ndarray
    stopping_probability: np.ndarray
    unsafe_assignment_probability: np.ndarray
    uniforms_consumed: int
    actual_capacity: int


def simulate_mtpi_desktop(study, *, profile="safe", seed=2):
    """Run every scenario in saved order using a single seeded stream.

    Legacy seed 2 reproduces the XLL's fixed MT19937 uint32-midpoint uniforms.
    Other seeds are a Python extension. Safe uses NumPy's default generator.
    Stop columns are safety, excessive underdosing, and capacity. No-selection
    is the last selection column, not a synonym for either stopping event.
    MCSEs describe independent trials conditional on the supplied scenarios.
    """
    capacity = _capacity(study, profile)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    S, D, T = len(study.scenarios), len(study.scenarios[0]), study.trials
    if S * T * D > 2_000_000 or S * T * capacity > 10_000_000:
        raise ValueError("simulation exceeds 2 million trial-dose cells or 10 million uniforms")
    total = S * T * capacity
    if profile == "desktop_legacy":
        tape = (
            np.random.RandomState(seed).randint(0, 2**32, total, dtype=np.uint32).astype(float)
            + 0.5
        ) / 2**32
    else:
        tape = np.random.default_rng(seed).random(total)
    selection = np.zeros((S, D + 1))
    mean = np.zeros((S, D))
    mcse = mean.copy()
    toxicity = np.zeros(S)
    stops = np.zeros((S, 3))
    unsafe = np.zeros(S)
    cursor = 0
    reasons = {"stop_safety": 0, "stop_underdose": 1, "capacity": 2}
    for i in range(S):
        counts = np.empty((T, D))
        dlts = 0
        for j in range(T):
            trial = _replay(study, i, tape[cursor : cursor + capacity], profile)
            cursor += int(trial.patients.sum())
            selection[i, (trial.selected or D + 1) - 1] += 1
            counts[j] = trial.patients
            dlts += int(trial.toxicities.sum())
            stops[i, reasons[trial.stop_reason]] += 1
            unsafe[i] += bool(trial.unsafe_assignments)
        mean[i] = counts.mean(axis=0)
        mcse[i] = counts.std(axis=0, ddof=1) / np.sqrt(T) if T > 1 else np.nan
        toxicity[i] = dlts / counts.sum()
    selection /= T
    return MTPIDesktopSimulation(
        study,
        profile,
        seed,
        _owned(selection),
        _owned(np.sqrt(selection * (1 - selection) / T)),
        _owned(mean),
        _owned(mcse),
        _owned(mean.sum(axis=1)),
        _owned(toxicity),
        _owned(stops / T),
        _owned(unsafe / T),
        cursor,
        capacity,
    )
