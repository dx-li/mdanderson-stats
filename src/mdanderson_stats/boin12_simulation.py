"""Complete binary-cohort simulation for the BOIN12 utility design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import _owned
from .boin12 import BOIN12Design, admissibility


@dataclass(frozen=True)
class BOIN12Simulation:
    """Replicated BOIN12 trial counts, selections, and Monte Carlo summaries."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    efficacies: NDArray[np.int64]
    efficacy_without_toxicity: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    selected_obd: NDArray[np.int64]
    selected_mtd: NDArray[np.int64]
    obd_probability: FloatArray
    obd_mcse: FloatArray
    mtd_probability: FloatArray
    mtd_mcse: FloatArray
    stop_reason: tuple[str, ...]


def _joint_grid(value: ArrayLike, doses: int | None = None) -> FloatArray:
    grid = finite(value, "joint_probability")
    if grid.ndim != 2 or grid.shape[1] != 4 or grid.shape[0] < 1:
        raise ValueError("joint_probability must have shape (doses, 4)")
    if doses is not None and grid.shape[0] != doses:
        raise ValueError("joint_probability has an unexpected number of doses")
    if np.any(grid < 0) or np.any(np.abs(grid.sum(axis=1) - 1.0) > 1e-12):
        raise ValueError("each joint probability row must be nonnegative and sum to one")
    return grid


def _start(value: int, doses: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or not 1 <= int(value) <= doses:
        raise ValueError("start_dose must be a one-based dose index")
    return int(value)


def simulate_boin12(
    design: BOIN12Design,
    joint_probability: ArrayLike,
    *,
    cohorts: int = 12,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> BOIN12Simulation:
    """Simulate BOIN12 using multinomial joint outcomes per dose.

    Each row of ``joint_probability`` is ordered as ``(no toxicity/efficacy,
    no toxicity/no efficacy, toxicity/efficacy, toxicity/no efficacy)``.  This
    preserves the toxicity/efficacy dependence supplied by the caller.
    """
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    grid = _joint_grid(joint_probability)
    doses = grid.shape[0]
    start = _start(start_dose, doses)
    settings = count([cohorts, cohort_size, trials], "simulation settings")
    if settings.shape != (3,) or np.any(settings < 1):
        raise ValueError("cohorts, cohort_size and trials must be positive integers")
    ncohort, size, repetitions = (int(x) for x in settings)
    if ncohort * size > 1000:
        raise ValueError("the trial may enroll at most 1000 patients")
    if repetitions > 1_000_000 or repetitions * doses > 2_000_000:
        raise ValueError("require at most 1000000 trials and 2000000 trial-dose cells")

    generator = np.random.default_rng(rng)
    patients = np.zeros((repetitions, doses), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    efficacies = np.zeros_like(patients)
    efficacy_without_toxicity = np.zeros_like(patients)
    eliminated = np.zeros((repetitions, doses), dtype=bool)
    selected_obd = np.zeros(repetitions, dtype=np.int64)
    selected_mtd = np.zeros(repetitions, dtype=np.int64)
    reasons = np.full(repetitions, "max_cohorts", dtype="U40")
    current = np.full(repetitions, start, dtype=np.int64)
    active = np.ones(repetitions, dtype=bool)

    for _ in range(ncohort):
        active_rows = np.flatnonzero(active)
        if active_rows.size == 0:
            break
        for trial in active_rows:
            dose = int(current[trial]) - 1
            cell = generator.multinomial(size, grid[dose])
            no_tox_eff, no_tox_no_eff, tox_eff, tox_no_eff = (int(x) for x in cell)
            patients[trial, dose] += size
            toxicities[trial, dose] += tox_eff + tox_no_eff
            efficacies[trial, dose] += no_tox_eff + tox_eff
            efficacy_without_toxicity[trial, dose] += no_tox_eff
            posterior = design.posterior(
                patients[trial],
                toxicities[trial],
                efficacies[trial],
                efficacy_without_toxicity=efficacy_without_toxicity[trial],
            )
            allowed = admissibility(
                posterior,
                toxicity_cutoff=design.toxicity_cutoff,
                efficacy_cutoff=design.efficacy_cutoff,
            )
            eliminated[trial] |= ~allowed
            if eliminated[trial, dose]:
                reasons[trial] = "stop_no_admissible_neighbor"
                active[trial] = False
                continue
            decision = design.next_dose(
                patients[trial],
                toxicities[trial],
                efficacies[trial],
                int(current[trial]),
                efficacy_without_toxicity=efficacy_without_toxicity[trial],
            )
            if decision.next_dose is None:
                reasons[trial] = decision.action
                active[trial] = False
            elif eliminated[trial, decision.next_dose - 1]:
                reasons[trial] = "stop_no_admissible_neighbor"
                active[trial] = False
            else:
                current[trial] = decision.next_dose

    for final_trial in range(repetitions):
        result = design.select_obd(
            patients[final_trial],
            toxicities[final_trial],
            efficacies[final_trial],
            efficacy_without_toxicity=efficacy_without_toxicity[final_trial],
        )
        if result.obd is not None:
            selected_obd[final_trial] = result.obd
        if result.mtd is not None:
            selected_mtd[final_trial] = result.mtd

    def summary(selected: NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        frequency = np.bincount(selected, minlength=doses + 1).astype(float) / repetitions
        return frequency, np.sqrt(frequency * (1.0 - frequency) / repetitions)

    obd_probability, obd_mcse = summary(selected_obd)
    mtd_probability, mtd_mcse = summary(selected_mtd)
    return BOIN12Simulation(
        _owned(patients),
        _owned(toxicities),
        _owned(efficacies),
        _owned(efficacy_without_toxicity),
        _owned(eliminated),
        _owned(selected_obd),
        _owned(selected_mtd),
        _owned(obd_probability),
        _owned(obd_mcse),
        _owned(mtd_probability),
        _owned(mtd_mcse),
        tuple(str(reason) for reason in reasons),
    )
